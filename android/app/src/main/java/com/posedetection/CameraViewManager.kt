package com.posedetection

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Choreographer
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.SimpleViewManager
import com.facebook.react.uimanager.ThemedReactContext
import com.facebook.react.uimanager.annotations.ReactProp
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

@ExperimentalGetImage
class CameraViewManager(private val reactContext: ReactApplicationContext) : SimpleViewManager<ConstraintLayout>() {
    
    companion object {
        private const val TAG = "CameraViewManager"
        private const val COMMAND_NAVIGATE_TO_RESULT = 1
    }

    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var currentLensFacing: Int = CameraSelector.LENS_FACING_FRONT
    private var currentView: ConstraintLayout? = null
    
    private var textToSpeech: TextToSpeech? = null
    private var lastSpokenTime = 0L
    private val SPEECH_COOLDOWN = 3000L // 3 seconds between voice instructions
    
    // Modular components
    private val poseValidator = PoseValidator()
    private val sidePoseValidator = SidePoseValidator()
    private val bodyPositionChecker = BodyPositionChecker()
    private var voiceFeedbackProvider: VoiceFeedbackProvider? = null
    
    private val options = PoseDetectorOptions.Builder()
        .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
        .build()

    private val poseDetector: PoseDetector = PoseDetection.getClient(options)
    
    private var isProcessing = false
    private val lock = Any()
    
    private var latestCleanBitmap: Bitmap? = null
    private var shouldCaptureClean = false

    // Countdown and capture state
    private var isCountingDown = false
    private var countdownValue = 5
    private var consecutiveGoodFrames = 0
    private val REQUIRED_GOOD_FRAMES = 10
    private var lastCountdownTime = 0L
    private var smileSaidTime = 0L
    private val SMILE_DELAY = 2000L // 2 seconds delay after saying smile
    
    // Two-stage capture state
    private var currentStage = PoseStage.FRONT_POSE
    private var frontPoseImage: Bitmap? = null
    private var sidePoseImage: Bitmap? = null
    private var frontImageCaptured = false
    private var sideImageCaptured = false

    // Reference pose
    private val referencePose = ReferencePose(
        shoulderAngle = 45.0,
        shoulderAngleTolerance = 15.0,
        elbowAngleLeft = 180.0,
        elbowAngleRight = 180.0,
        elbowAngleTolerance = 20.0,
        spineAngle = 0.0,
        spineAngleTolerance = 10.0,
        hipAngleLeft = 180.0,
        hipAngleRight = 180.0,
        hipAngleTolerance = 15.0,
        shoulderLevelDiff = 0.0,
        shoulderLevelTolerance = 30.0,
        legSeparationAngle = 45.0,
        legSeparationTolerance = 15.0
    )
    
    private val sidePoseReference = SidePoseReference(
        neckHeadAngle = 180.0,
        neckHeadTolerance = 15.0,
        armAngle = 180.0,
        armTolerance = 20.0,
        spineVerticalAngle = 0.0,
        spineTolerance = 15.0,
        legStraightAngle = 180.0,
        legTolerance = 15.0,
        shoulderDepthDiff = 0.0,
        shoulderDepthTolerance = 30.0  // Shoulders should be vertically aligned (small Y difference)
    )
    
    override fun getName(): String {
        return "CameraView"
    }
    
    override fun getCommandsMap(): Map<String, Int> {
        return mapOf(
            "navigateToResult" to COMMAND_NAVIGATE_TO_RESULT
        )
    }
    
    override fun receiveCommand(root: ConstraintLayout, commandId: Int, args: com.facebook.react.bridge.ReadableArray?) {
        when (commandId) {
            COMMAND_NAVIGATE_TO_RESULT -> {
                sendImagesToReactNative()
            }
        }
    }

    override fun createViewInstance(reactContext: ThemedReactContext): ConstraintLayout {
        // Reset detection state to fix re-entry issue
        resetDetectionState()
        
        // Initialize TextToSpeech
        textToSpeech = TextToSpeech(reactContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech?.language = Locale.US
                voiceFeedbackProvider = VoiceFeedbackProvider(textToSpeech)
                Log.d(TAG, "TextToSpeech initialized successfully")
            } else {
                Log.e(TAG, "TextToSpeech initialization failed")
            }
        }
        
        val constraintLayout = ConstraintLayout(reactContext)
        constraintLayout.setBackgroundColor(Color.BLACK)
        constraintLayout.layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
        
        val previewView = PreviewView(reactContext)
        previewView.setBackgroundColor(Color.BLACK)
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        previewView.id = android.view.View.generateViewId()
        
        val previewParams = ConstraintLayout.LayoutParams(
            ConstraintLayout.LayoutParams.MATCH_PARENT,
            ConstraintLayout.LayoutParams.MATCH_PARENT
        )
        previewParams.topToTop = ConstraintLayout.LayoutParams.PARENT_ID
        previewParams.bottomToBottom = ConstraintLayout.LayoutParams.PARENT_ID
        previewParams.startToStart = ConstraintLayout.LayoutParams.PARENT_ID
        previewParams.endToEnd = ConstraintLayout.LayoutParams.PARENT_ID
        
        previewView.layoutParams = previewParams
        constraintLayout.addView(previewView)
        
        // Add overlay view
        val overlayView = PoseOverlayView(reactContext)
        overlayView.id = android.view.View.generateViewId()
        
        val overlayParams = ConstraintLayout.LayoutParams(
            ConstraintLayout.LayoutParams.MATCH_PARENT,
            ConstraintLayout.LayoutParams.MATCH_PARENT
        )
        overlayParams.topToTop = ConstraintLayout.LayoutParams.PARENT_ID
        overlayParams.bottomToBottom = ConstraintLayout.LayoutParams.PARENT_ID
        overlayParams.startToStart = ConstraintLayout.LayoutParams.PARENT_ID
        overlayParams.endToEnd = ConstraintLayout.LayoutParams.PARENT_ID
        
        overlayView.layoutParams = overlayParams
        constraintLayout.addView(overlayView)
        
        // Store views as tags
        constraintLayout.tag = previewView
        constraintLayout.setTag(R.id.pose_overlay_tag, overlayView)
        
        // Install hierarchy fitter AFTER adding views - critical for CameraX
        installHierarchyFitter(constraintLayout)
        
        // Store reference to current view
        currentView = constraintLayout
        
        Log.d(TAG, "ConstraintLayout created with PreviewView and PoseOverlayView")
        
        return constraintLayout
    }

    private fun installHierarchyFitter(view: ViewGroup) {
        // CameraX black screen fix - continuous layout with Choreographer
        // Solution from: https://stackoverflow.com/questions/79664703
        Choreographer.getInstance().postFrameCallback(object : Choreographer.FrameCallback {
            override fun doFrame(frameTimeNanos: Long) {
                manuallyLayoutChildren(view)
                view.viewTreeObserver.dispatchOnGlobalLayout()
                Choreographer.getInstance().postFrameCallback(this)
            }
        })
    }
    
    private fun manuallyLayoutChildren(viewGroup: ViewGroup) {
        for (i in 0 until viewGroup.childCount) {
            val child = viewGroup.getChildAt(i)
            child.measure(
                View.MeasureSpec.makeMeasureSpec(viewGroup.measuredWidth, View.MeasureSpec.EXACTLY),
                View.MeasureSpec.makeMeasureSpec(viewGroup.measuredHeight, View.MeasureSpec.EXACTLY)
            )
            child.layout(0, 0, child.measuredWidth, child.measuredHeight)
        }
    }

    @ReactProp(name = "cameraType")
    fun setCameraType(view: ConstraintLayout, cameraType: String? = "front") {
        currentLensFacing = when (cameraType) {
            "front" -> CameraSelector.LENS_FACING_FRONT
            else -> CameraSelector.LENS_FACING_BACK
        }
        Log.d(TAG, "Setting camera type to: $cameraType (lensFacing: $currentLensFacing)")
        
        // Reset detection state when camera type is set (fixes re-entry detection issue)
        resetDetectionState()
        
        val previewView = view.tag as? PreviewView
        val overlayView = view.getTag(R.id.pose_overlay_tag) as? PoseOverlayView
        
        if (previewView != null && overlayView != null) {
            // Add delay to ensure view is fully ready and screen is active
            view.postDelayed({
                if (view.isAttachedToWindow) {
                    startCamera(previewView, overlayView, currentLensFacing)
                } else {
                    Log.w(TAG, "View not attached to window, skipping camera start")
                }
            }, 300) // 300ms delay
        } else {
            Log.e(TAG, "PreviewView or OverlayView not found")
        }
    }

    private fun startCamera(previewView: PreviewView, overlayView: PoseOverlayView, lensFacing: Int) {
        Log.d(TAG, "Starting camera with lensFacing: $lensFacing")
        
        // Recreate executor if it's been shut down (fixes re-entry detection issue)
        if (cameraExecutor.isShutdown || cameraExecutor.isTerminated) {
            Log.d(TAG, "Recreating camera executor (was shutdown)")
            cameraExecutor = Executors.newSingleThreadExecutor()
        }
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(reactContext)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                
                // Unbind all use cases before rebinding
                cameraProvider.unbindAll()
                Log.d(TAG, "Unbound all previous use cases")
                
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    synchronized(lock) {
                        if (isProcessing) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        isProcessing = true
                    }

                    val image = imageProxy.image
                    if (image == null) {
                        synchronized(lock) { isProcessing = false }
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    try {
                        val buffer: ByteBuffer = image.planes[0].buffer
                        buffer.rewind()
                        val bitmap = Bitmap.createBitmap(
                            imageProxy.width,
                            imageProxy.height,
                            Bitmap.Config.ARGB_8888
                        )
                        bitmap.copyPixelsFromBuffer(buffer)

                        val matrix = Matrix().apply {
                            postRotate(270f)
                            postScale(-1f, 1f, imageProxy.width / 2f, imageProxy.height / 2f)
                        }

                        val rotatedBitmap = Bitmap.createBitmap(
                            bitmap, 0, 0,
                            imageProxy.width, imageProxy.height, matrix, false
                        )
                        
                        // Store clean bitmap for potential capture
                        latestCleanBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, false)

                        val inputImage = InputImage.fromBitmap(rotatedBitmap, 0)
                        poseDetector.process(inputImage)
                            .addOnSuccessListener { pose ->
                                val allLandmarks = pose.allPoseLandmarks
                                if (allLandmarks.isNotEmpty()) {
                                    if (currentStage == PoseStage.FRONT_POSE) {
                                        handleFrontPoseDetection(pose, rotatedBitmap, overlayView)
                                    } else {
                                        handleSidePoseDetection(pose, rotatedBitmap, overlayView)
                                    }
                                }
                                
                                synchronized(lock) {
                                    isProcessing = false
                                }
                            }
                            .addOnFailureListener { e ->
                                Log.e(TAG, "Pose detection failed", e)
                                synchronized(lock) {
                                    isProcessing = false
                                }
                            }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error processing image", e)
                        synchronized(lock) {
                            isProcessing = false
                        }
                    } finally {
                        imageProxy.close()
                    }
                }

                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(lensFacing)
                    .build()

                val activity = reactContext.currentActivity
                if (activity is LifecycleOwner) {
                    Log.d(TAG, "Binding camera to lifecycle")
                    cameraProvider.bindToLifecycle(
                        activity,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                    Log.d(TAG, "Camera started successfully with pose detection")
                    
                    // Send initial status using direct emission
                    android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                        sendStatusEvent("camera_started", "Camera started and ready for detection")
                    }, 500)
                } else {
                    Log.e(TAG, "Activity is not a LifecycleOwner")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error starting camera", e)
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(reactContext))
    }

    override fun onDropViewInstance(view: ConstraintLayout) {
        super.onDropViewInstance(view)
        Log.d(TAG, "Dropping view instance")
        
        // Clear current view reference
        currentView = null
        
        // Cleanup TextToSpeech
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        
        // Only unbind camera, don't shutdown executor or close detector
        // This allows the camera to restart when user returns to screen
        val cameraProviderFuture = ProcessCameraProvider.getInstance(reactContext)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
                Log.d(TAG, "Camera unbound (executor kept alive for re-entry)")
            } catch (e: Exception) {
                Log.e(TAG, "Error unbinding camera", e)
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(reactContext))
    }
    
    private fun handleFrontPoseDetection(pose: Pose, bitmap: Bitmap, overlayView: PoseOverlayView) {
        if (frontImageCaptured) return
        
        val personDetected = poseValidator.isPersonFullyDetected(pose)
        if (!personDetected) {
            voiceFeedbackProvider?.provideFrontPoseFeedback("Please stand in front of camera", null, null)
            val activity = reactContext.currentActivity
            activity?.runOnUiThread {
                overlayView.updatePose(null, null, false, 0, false, currentStage)
                overlayView.updateBitmap(bitmap)
            }
            return
        }
        
        val positionCheck = bodyPositionChecker.checkBodyPosition(pose, bitmap.width, bitmap.height)
        if (!positionCheck.inBox) {
            voiceFeedbackProvider?.provideFrontPoseFeedback(positionCheck.issues.firstOrNull(), null, positionCheck)
            val activity = reactContext.currentActivity
            activity?.runOnUiThread {
                overlayView.updatePose(pose, null, false, 0, false, currentStage)
                overlayView.updateBitmap(bitmap)
            }
            return
        }
        
        val metrics = poseValidator.calculatePostureMetrics(pose)
        val accuracy = poseValidator.compareWithReference(metrics, referencePose)
        val isPerfectPose = poseValidator.isPoseAccurate(accuracy, positionCheck)
        
        if (!isPerfectPose) {
            voiceFeedbackProvider?.provideFrontPoseFeedback(null, accuracy, positionCheck)
        }
        
        handleFrontPoseCapture(isPerfectPose, bitmap, overlayView, pose, accuracy)
    }
    
    private fun handleSidePoseDetection(pose: Pose, bitmap: Bitmap, overlayView: PoseOverlayView) {
        if (sideImageCaptured) return
        
        val personDetected = poseValidator.isPersonFullyDetected(pose)
        if (!personDetected) {
            voiceFeedbackProvider?.speak("Please stand in front of camera")
            val activity = reactContext.currentActivity
            activity?.runOnUiThread {
                overlayView.updatePose(null, null, false, 0, false, currentStage)
                overlayView.updateBitmap(bitmap)
            }
            return
        }
        
        val positionCheck = bodyPositionChecker.checkBodyPosition(pose, bitmap.width, bitmap.height)
        val sideMetrics = sidePoseValidator.calculateSidePoseMetrics(pose)
        val sideAccuracy = sidePoseValidator.compareWithSideReference(sideMetrics, sidePoseReference)
        val isPerfectSidePose = sidePoseValidator.isSidePoseAccurate(sideAccuracy, positionCheck)
        
        if (!isPerfectSidePose) {
            voiceFeedbackProvider?.provideSidePoseFeedback(sideAccuracy, positionCheck)
        }
        
        handleSidePoseCapture(isPerfectSidePose, bitmap, overlayView, pose, sideAccuracy)
    }
    
    private fun handleFrontPoseCapture(isPerfectPose: Boolean, bitmap: Bitmap, overlayView: PoseOverlayView, pose: Pose, accuracy: PostureAccuracy) {
        if (isPerfectPose) {
            consecutiveGoodFrames++
            if (consecutiveGoodFrames >= REQUIRED_GOOD_FRAMES && !isCountingDown) {
                isCountingDown = true
                countdownValue = 5
                lastCountdownTime = System.currentTimeMillis()
                smileSaidTime = 0L
                textToSpeech?.speak("Perfect posture! Hold still", TextToSpeech.QUEUE_FLUSH, null, null)
                
                // Emit ready to capture status
                sendStatusEvent("ready_to_capture", "Front pose ready, countdown started")
            } else if (isCountingDown) {
                val currentTime = System.currentTimeMillis()
                if (smileSaidTime > 0) {
                    if (currentTime - smileSaidTime >= SMILE_DELAY && !frontImageCaptured) {
                        frontPoseImage = bitmap.copy(Bitmap.Config.ARGB_8888, false)
                        frontImageCaptured = true
                        isCountingDown = false
                        consecutiveGoodFrames = 0
                        countdownValue = 5
                        smileSaidTime = 0L
                        
                        // Emit front pose captured status
                        sendStatusEvent("front_pose_captured", "Front pose captured successfully")
                        
                        // Transition to side pose stage
                        currentStage = PoseStage.SIDE_POSE
                        
                        // Give clear instruction to turn sideways
                        val activity = reactContext.currentActivity
                        activity?.runOnUiThread {
                            overlayView.updatePose(null, null, false, 0, false, currentStage)
                        }
                        
                        // Use handler to delay voice instruction so user hears it clearly
                        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                            textToSpeech?.speak("Great! Now turn sideways. Face left and show your side to the camera", TextToSpeech.QUEUE_FLUSH, null, null)
                        }, 1500)
                    }
                } else if (currentTime - lastCountdownTime >= 1000) {
                    countdownValue--
                    lastCountdownTime = currentTime
                    if (countdownValue > 0) {
                        textToSpeech?.speak(countdownValue.toString(), TextToSpeech.QUEUE_FLUSH, null, null)
                    } else {
                        textToSpeech?.speak("Smile!", TextToSpeech.QUEUE_FLUSH, null, null)
                        smileSaidTime = currentTime
                    }
                }
            }
        } else {
            if (isCountingDown) {
                textToSpeech?.speak("Hold your position", TextToSpeech.QUEUE_FLUSH, null, null)
                isCountingDown = false
                countdownValue = 5
                smileSaidTime = 0L
            }
            consecutiveGoodFrames = 0
        }
        
        val activity = reactContext.currentActivity
        activity?.runOnUiThread {
            overlayView.updatePose(pose, accuracy, isPerfectPose, countdownValue, isCountingDown, currentStage)
            overlayView.updateBitmap(bitmap)
        }
    }
    
    private fun handleSidePoseCapture(isPerfectPose: Boolean, bitmap: Bitmap, overlayView: PoseOverlayView, pose: Pose, accuracy: SidePoseAccuracy) {
        if (isPerfectPose) {
            consecutiveGoodFrames++
            if (consecutiveGoodFrames >= REQUIRED_GOOD_FRAMES && !isCountingDown) {
                isCountingDown = true
                countdownValue = 5
                lastCountdownTime = System.currentTimeMillis()
                smileSaidTime = 0L
                textToSpeech?.speak("Perfect! Hold still", TextToSpeech.QUEUE_FLUSH, null, null)
                
                // Emit ready to capture status for side pose
                sendStatusEvent("ready_to_capture_side", "Side pose ready, countdown started")
            } else if (isCountingDown) {
                val currentTime = System.currentTimeMillis()
                if (smileSaidTime > 0) {
                    if (currentTime - smileSaidTime >= SMILE_DELAY && !sideImageCaptured) {
                        sidePoseImage = bitmap.copy(Bitmap.Config.ARGB_8888, false)
                        sideImageCaptured = true
                        isCountingDown = false
                        consecutiveGoodFrames = 0
                        
                        Log.d(TAG, "Side pose captured! Front image exists: ${frontPoseImage != null}, Side image exists: ${sidePoseImage != null}")
                        textToSpeech?.speak("Perfect! Both poses captured!", TextToSpeech.QUEUE_FLUSH, null, null)
                        
                        // Emit both poses captured status
                        sendStatusEvent("both_poses_captured", "Both front and side poses captured successfully")
                        
                        // Send both images to React Native
                        Log.d(TAG, "About to call sendImagesToReactNative()")
                        sendImagesToReactNative()
                        
                        val activity = reactContext.currentActivity
                        activity?.runOnUiThread {
                            overlayView.updatePose(null, null, false, 0, false, currentStage)
                            overlayView.updateBitmap(null)
                        }
                    }
                } else if (currentTime - lastCountdownTime >= 1000) {
                    countdownValue--
                    lastCountdownTime = currentTime
                    if (countdownValue > 0) {
                        textToSpeech?.speak(countdownValue.toString(), TextToSpeech.QUEUE_FLUSH, null, null)
                    } else {
                        textToSpeech?.speak("Smile!", TextToSpeech.QUEUE_FLUSH, null, null)
                        smileSaidTime = currentTime
                    }
                }
            }
        } else {
            if (isCountingDown) {
                textToSpeech?.speak("Hold your position", TextToSpeech.QUEUE_FLUSH, null, null)
                isCountingDown = false
                countdownValue = 5
                smileSaidTime = 0L
            }
            consecutiveGoodFrames = 0
        }
        
        val activity = reactContext.currentActivity
        activity?.runOnUiThread {
            overlayView.updateSidePose(pose, accuracy, isPerfectPose, countdownValue, isCountingDown, currentStage)
            overlayView.updateBitmap(bitmap)
        }
    }
    
    private fun sendImagesToReactNative() {
        Log.d(TAG, "sendImagesToReactNative called")
        try {
            Log.d(TAG, "Front image null? ${frontPoseImage == null}, Side image null? ${sidePoseImage == null}")
            
            val frontUri = saveBitmapToCache(frontPoseImage, "front_pose.jpg")
            Log.d(TAG, "Front URI: $frontUri")
            
            val sideUri = saveBitmapToCache(sidePoseImage, "side_pose.jpg")
            Log.d(TAG, "Side URI: $sideUri")
            
            if (frontUri != null && sideUri != null) {
                Log.d(TAG, "Both URIs are valid, sending event directly")
                // Send event directly from CameraViewManager
                val params = com.facebook.react.bridge.Arguments.createMap()
                params.putString("frontUri", frontUri)
                params.putString("sideUri", sideUri)
                
                reactContext
                    .getJSModule(com.facebook.react.modules.core.DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                    .emit("onBothImagesCaptured", params)
                    
                Log.d(TAG, "Both images event emitted successfully - Front: $frontUri, Side: $sideUri")
            } else {
                Log.e(TAG, "Failed to save images - Front: $frontUri, Side: $sideUri")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending images to React Native", e)
            e.printStackTrace()
        }
    }
    
    private fun saveBitmapToCache(bitmap: Bitmap?, filename: String): String? {
        return try {
            bitmap?.let {
                val cacheDir = reactContext.cacheDir
                val file = java.io.File(cacheDir, filename)
                java.io.FileOutputStream(file).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
                "file://${file.absolutePath}"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error saving bitmap to cache: $filename", e)
            null
        }
    }
    
    private fun resetDetectionState() {
        Log.d(TAG, "Resetting detection state")
        isProcessing = false
        isCountingDown = false
        countdownValue = 5
        consecutiveGoodFrames = 0
        currentStage = PoseStage.FRONT_POSE
        frontPoseImage = null
        sidePoseImage = null
        frontImageCaptured = false
        sideImageCaptured = false
        shouldCaptureClean = false
        lastCountdownTime = 0L
        smileSaidTime = 0L
        lastSpokenTime = 0L
    }
    
    private fun sendStatusEvent(status: String, message: String = "") {
        try {
            val params = com.facebook.react.bridge.Arguments.createMap()
            params.putString("status", status)
            params.putString("message", message)
            Log.d(TAG, "Sending status event - Status: $status, Message: $message")
            reactContext
                .getJSModule(com.facebook.react.modules.core.DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("onCaptureStatus", params)
            Log.d(TAG, "Status event emitted successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error sending status event", e)
        }
    }
    
    private fun getImageCaptureModule(): ImageCaptureModule? {
        return try {
            reactContext.getNativeModule(ImageCaptureModule::class.java)
        } catch (e: Exception) {
            Log.e(TAG, "Error getting ImageCaptureModule", e)
            null
        }
    }
}
