import Foundation
import React
import AVFoundation
import Vision
import ImageIO
import MLKitPoseDetection
import MLKitVision
import CoreGraphics
 

@objc(CameraViewManager)
class CameraViewManager: RCTViewManager {

    
    override static func requiresMainQueueSetup() -> Bool {
        return true
    }
    
    override func view() -> UIView! {
        let view = CameraView()
        // Pass bridge to the view so it can emit DeviceEventEmitter events
        view.reactBridge = self.bridge
        return view
    }

    override class func moduleName() -> String! {
        return "CameraView"
    }
    
    @objc func setCameraType(_ node: NSNumber, cameraType: String) {
        DispatchQueue.main.async {
            if let component = self.bridge.uiManager.view(forReactTag: node) as? CameraView {
                component.updateCameraType(cameraType)
            }
        }
    }
}

class CameraView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate {
    // Bridge reference for emitting JS device events (matches Android behavior)
    weak var reactBridge: RCTBridge?
    
    private var captureSession: AVCaptureSession?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var currentCamera: AVCaptureDevice?
    
    private let poseValidator = PoseValidator()
    private let sidePoseValidator = SidePoseValidator()
    private let bodyPositionChecker = BodyPositionChecker()
    private let voiceFeedback = VoiceFeedbackProvider()
    
    private var countdownTimer: Timer?
    private var countdownValue = 3
    private var isCapturing = false
    private var latestSampleBuffer: CMSampleBuffer?
    private let processingQueue = DispatchQueue(label: "com.posedetection.processing", qos: .userInitiated)
    // Reuse a single CIContext to avoid per-frame allocations
    private let ciContext = CIContext()
    // Throttled logging to avoid spamming console and impacting performance
    private var frameCounter: Int = 0
    private let logEveryNFrames: Int = 10
    private let poseDetector: PoseDetector = {
        let options = PoseDetectorOptions()
        options.detectorMode = .stream
        return PoseDetector.poseDetector(options: options)
    }()
    
    private enum PoseStage {
        case frontPose
        case sidePose
    }
    
    private var currentStage: PoseStage = .frontPose
    private var frontImagePath: String?
    private var sideImagePath: String?
    private var initialCameraType: String = "front"
    
    @objc var cameraType: NSString = "front" {
        didSet {
            initialCameraType = cameraType as String
            if captureSession == nil {
                setupCamera()
            }
        }
    }
    
    @objc var onCaptureStatus: RCTDirectEventBlock?
    @objc var onBothCaptured: RCTDirectEventBlock?
    
    private let poseOverlayView = PoseOverlayView()
    
    private let countdownLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.font = UIFont.boldSystemFont(ofSize: 72)
        label.textAlignment = .center
        label.isHidden = true
        return label
    }()
    
    private var latestImageSize: CGSize? = nil
    // Keep last-known overlay state so timer ticks can refresh the overlay independently
    private var lastAccuracy: PoseValidator.PostureAccuracy? = nil
    private var lastGuidance: [JointName: String]? = nil
    private var lastMirrored: Bool = false
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupCamera()
        setupUI()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupUI() {
        // Add pose overlay
        addSubview(poseOverlayView)
        poseOverlayView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            poseOverlayView.leadingAnchor.constraint(equalTo: leadingAnchor),
            poseOverlayView.trailingAnchor.constraint(equalTo: trailingAnchor),
            poseOverlayView.topAnchor.constraint(equalTo: topAnchor),
            poseOverlayView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
        
        addSubview(countdownLabel)
        countdownLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            countdownLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            countdownLabel.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }
    
    private func setupCamera() {
        // Check camera permission
        let cameraAuthStatus = AVCaptureDevice.authorizationStatus(for: .video)
        if cameraAuthStatus != .authorized {
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    DispatchQueue.main.async {
                        self?.setupCamera()
                    }
                } else {
                    print("Camera permission denied")
                }
            }
            return
        }
        
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .high
        
        // Use camera based on prop (default: front)
        let position: AVCaptureDevice.Position = initialCameraType == "front" ? .front : .back
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            print("Unable to access \(initialCameraType) camera")
            sendStatusEvent(status: "error", message: "Camera not available")
            return
        }
        
        currentCamera = camera
        
        do {
            let input = try AVCaptureDeviceInput(device: camera)
            
            if captureSession?.canAddInput(input) == true {
                captureSession?.addInput(input)
            }
            
            videoOutput = AVCaptureVideoDataOutput()
            videoOutput?.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput?.alwaysDiscardsLateVideoFrames = true
            videoOutput?.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            
            if captureSession?.canAddOutput(videoOutput!) == true {
                captureSession?.addOutput(videoOutput!)
            }

            if let connection = videoOutput?.connection(with: .video) {
                connection.videoOrientation = .portrait
                connection.isVideoMirrored = position == .front
            }
            
            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
            previewLayer?.videoGravity = .resizeAspectFill
            previewLayer?.frame = bounds
            
            if let previewLayer = previewLayer {
                layer.insertSublayer(previewLayer, at: 0)
            }
            
            // Ensure overlay is on top and visible
            bringSubviewToFront(poseOverlayView)
            bringSubviewToFront(countdownLabel)
            poseOverlayView.isHidden = false
            poseOverlayView.alpha = 1.0
            
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession?.startRunning()
                DispatchQueue.main.async {
                    // Keep screen on during pose detection
                    UIApplication.shared.isIdleTimerDisabled = true
                    self?.sendStatusEvent(status: "camera_started", message: "Camera started and ready!")
                }
            }
            
        } catch {
            print("Error setting up camera: \(error)")
        }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
        
        // Don't set frame manually - using Auto Layout constraints
        // Ensure overlay stays on top after layout changes
        bringSubviewToFront(poseOverlayView)
        bringSubviewToFront(countdownLabel)
    }
    
    @objc func updateCameraType(_ type: String) {
        guard let session = captureSession else { return }
        
        session.beginConfiguration()
        
        // Remove existing inputs
        if let currentInput = session.inputs.first as? AVCaptureDeviceInput {
            session.removeInput(currentInput)
        }
        
        let position: AVCaptureDevice.Position = type == "front" ? .front : .back
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            session.commitConfiguration()
            return
        }
        
        currentCamera = camera
        
        do {
            let input = try AVCaptureDeviceInput(device: camera)
            if session.canAddInput(input) {
                session.addInput(input)
            }
        } catch {
            print("Error switching camera: \(error)")
        }
        
        session.commitConfiguration()
        
        // Reset detection state
        resetDetectionState()
    }
    
    private func resetDetectionState() {
        isCapturing = false
        countdownTimer?.invalidate()
        countdownTimer = nil
        countdownValue = 3
        DispatchQueue.main.async {
            self.countdownLabel.isHidden = true
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Guard: prevent processing if view is being deallocated
        guard self.captureSession != nil else { return }
        
        // Store latest sample buffer for capture
        latestSampleBuffer = sampleBuffer
        
        // Continue pose detection even during countdown for overlay updates
        // The countdown timer validates the pose separately
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Snapshot values needed inside async block
        let isMirrored = connection.isVideoMirrored
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        processingQueue.async { [weak self] in
            guard let self = self else { return }
            autoreleasepool {
                // ML Kit Pose Detection (streaming mode per docs)
                let visionImage = VisionImage(buffer: sampleBuffer)
                visionImage.orientation = self.mlkitImageOrientation(
                    deviceOrientation: UIDevice.current.orientation,
                    cameraPosition: self.currentCamera?.position ?? AVCaptureDevice.Position.back
                )

                self.poseDetector.process(visionImage) { [weak self] poses, error in
                    guard let self = self else { return }
                    if let error = error {
                        print("⚠️ ML Kit error: \(error.localizedDescription)")
                        DispatchQueue.main.async {
                            self.poseOverlayView.updatePose(nil, imageSize: CGSize(width: width, height: height))
                        }
                        return
                    }

                    print("🔍 ML Kit callback: \(poses?.count ?? 0) poses detected")
                    
                    guard let pose = poses?.first else {
                        print("⚠️ No pose in frame (poses=\(poses == nil ? "nil" : "empty"))")
                        DispatchQueue.main.async {
                            self.poseOverlayView.updatePose(nil, imageSize: CGSize(width: width, height: height))
                        }
                        return
                    }
                    
                    print("✅ Pose found with \(pose.landmarks.count) landmarks")

                    // Map ML Kit landmarks to shared dictionary (keep pixel coordinates, not normalized)
                    var points: PoseLandmarks = [:]
                    func add(_ type: PoseLandmarkType, _ joint: JointName) {
                        let lm = pose.landmark(ofType: type)
                        let location = CGPoint(x: CGFloat(lm.position.x), y: CGFloat(lm.position.y))
                        points[joint] = RecognizedPointCompat(location: location, confidence: lm.inFrameLikelihood)
                    }
                    let leftShoulderLM = pose.landmark(ofType: .leftShoulder)
                    let rightShoulderLM = pose.landmark(ofType: .rightShoulder)
                    add(.leftShoulder, .leftShoulder)
                    add(.rightShoulder, .rightShoulder)
                    add(.leftElbow, .leftElbow)
                    add(.rightElbow, .rightElbow)
                    add(.leftWrist, .leftWrist)
                    add(.rightWrist, .rightWrist)
                    add(.leftHip, .leftHip)
                    add(.rightHip, .rightHip)
                    add(.leftKnee, .leftKnee)
                    add(.rightKnee, .rightKnee)
                    add(.leftAnkle, .leftAnkle)
                    add(.rightAnkle, .rightAnkle)
                    // Foot landmarks
                    add(.leftHeel, .leftHeel)
                    add(.rightHeel, .rightHeel)
                    add(.leftToe, .leftFootIndex)
                    add(.rightToe, .rightFootIndex)
                    // Hand landmarks
                    add(.leftPinkyFinger, .leftPinky)
                    add(.rightPinkyFinger, .rightPinky)
                    add(.leftIndexFinger, .leftIndex)
                    add(.rightIndexFinger, .rightIndex)
                    add(.leftThumb, .leftThumb)
                    add(.rightThumb, .rightThumb)
                    
                    // Throttled landmark logging to confirm ankle/feet visibility without spamming
                    frameCounter &+= 1
                    if frameCounter % logEveryNFrames == 0 {
                        var summary: [String] = []
                        let ordered: [JointName] = [
                            .nose, .leftEye, .rightEye, .leftEar, .rightEar,
                            .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
                            .leftWrist, .rightWrist, .leftHip, .rightHip,
                            .leftKnee, .rightKnee, .leftAnkle, .rightAnkle
                        ]
                        for joint in ordered {
                            if let pt = points[joint] {
                                summary.append("\(joint): (\(Int(pt.location.x)), \(Int(pt.location.y))) conf=\(String(format: "%.2f", pt.confidence))")
                            } else {
                                summary.append("\(joint): missing")
                            }
                        }
                        print("📍 Landmarks [frame #\(frameCounter)]:")
                        summary.forEach { print("   \($0)") }
                    }
                    
                    // Add neck (average of shoulders if not available)
                  _ = pose.landmark(ofType: .nose) // ML Kit doesn't have neck, approximate
                    let neckLocation = CGPoint(
                        x: (CGFloat(leftShoulderLM.position.x) + CGFloat(rightShoulderLM.position.x)) / 2,
                        y: (CGFloat(leftShoulderLM.position.y) + CGFloat(rightShoulderLM.position.y)) / 2
                    )
                    points[.neck] = RecognizedPointCompat(location: neckLocation, confidence: min(leftShoulderLM.inFrameLikelihood, rightShoulderLM.inFrameLikelihood))
                    
                    let nose = pose.landmark(ofType: .nose)
                    let noseLocation = CGPoint(x: CGFloat(nose.position.x), y: CGFloat(nose.position.y))
                    points[.nose] = RecognizedPointCompat(location: noseLocation, confidence: nose.inFrameLikelihood)
                    
                    // Add eye landmarks for drawing
                    add(.leftEye, .leftEye)
                    add(.rightEye, .rightEye)
                    add(.leftEyeInner, .leftEyeInner)
                    add(.leftEyeOuter, .leftEyeOuter)
                    add(.rightEyeInner, .rightEyeInner)
                    add(.rightEyeOuter, .rightEyeOuter)
                    
                    // Add ear landmarks (matching Android)
                    add(.leftEar, .leftEar)
                    add(.rightEar, .rightEar)
                    
                    // Add mouth landmarks (matching Android)
                    add(.mouthLeft, .mouthLeft)
                    add(.mouthRight, .mouthRight)

                    // Calculate accuracy for overlay coloring
                    // Build accuracy differently depending on current stage
                    let metrics = self.poseValidator.calculatePostureMetrics(points)
                    var accuracy: PoseValidator.PostureAccuracy? = nil
                    // Build an imageSize to pass to side-pose checks (normalize x coordinates)
                    let currentImageSize = CGSize(width: width, height: height)
                    if self.currentStage == .frontPose {
                        accuracy = metrics.map { self.poseValidator.compareWithReference($0) }

                        // If user is accidentally sideways while we expect front pose, mark as inaccurate
                        if self.sidePoseValidator.isSidewaysPose(points, imageSize: currentImageSize) {
                            accuracy = PoseValidator.PostureAccuracy(
                                shoulderAccurateLeft: false,
                                shoulderAccurateRight: false,
                                elbowAccurateLeft: false,
                                elbowAccurateRight: false,
                                spineAccurate: false,
                                hipAccurateLeft: false,
                                hipAccurateRight: false
                            )
                        }
                    } else {
                        // Side pose stage - derive an accuracy-like struct from side checks
                        let isSide = self.sidePoseValidator.isSidewaysPose(points, imageSize: currentImageSize)
                        let armsSide = self.sidePoseValidator.areArmsSideways(points, imageSize: currentImageSize)
                        let legsSide = self.sidePoseValidator.areLegsSideways(points, imageSize: currentImageSize)
                        accuracy = PoseValidator.PostureAccuracy(
                            shoulderAccurateLeft: isSide && armsSide,
                            shoulderAccurateRight: isSide && armsSide,
                            elbowAccurateLeft: true,
                            elbowAccurateRight: true,
                            spineAccurate: isSide,
                            hipAccurateLeft: legsSide,
                            hipAccurateRight: legsSide
                        )
                    }

                    // Build guidance messages per joint when not accurate (front vs side)
                    var guidance: [JointName: String]? = nil
                    if self.currentStage == .frontPose {
                        // Use checkPoseDetailed to get per-arm directional guidance
                        let detailedResult = self.bodyPositionChecker.checkPoseDetailed(points)
                        if let gj = detailedResult.guidanceJoints, !gj.isEmpty {
                            guidance = gj
                        } else if let acc = accuracy {
                            // Fallback: use accuracy-based guidance for legs/spine
                            var g: [JointName: String] = [:]
                            if !acc.spineAccurate { g[.neck] = "Stand up straight" }
                            if !acc.hipAccurateLeft  { g[.leftHip] = "Keep legs straight"; g[.leftKnee] = "straighten" }
                            if !acc.hipAccurateRight { g[.rightHip] = "Keep legs straight"; g[.rightKnee] = "straighten" }
                            if !g.isEmpty { guidance = g }
                        }
                    } else {
                        // Side stage guidance
                        let isSide = self.sidePoseValidator.isSidewaysPose(points, imageSize: currentImageSize)
                        let armsSide = self.sidePoseValidator.areArmsSideways(points, imageSize: currentImageSize)
                        let legsSide = self.sidePoseValidator.areLegsSideways(points, imageSize: currentImageSize)
                        var g: [JointName: String] = [:]
                        if !isSide {
                            if let left = points[.leftShoulder], let right = points[.rightShoulder] {
                                let diff = left.location.x - right.location.x
                                if diff < 0 {
                                    g[.neck] = "Rotate slightly to your right"
                                } else {
                                    g[.neck] = "Rotate slightly to your left"
                                }
                            } else {
                                g[.neck] = "Rotate so one shoulder is behind the other"
                            }
                        }
                        if !armsSide {
                            g[.leftElbow] = "Bring arms in line with shoulders"
                            g[.rightElbow] = "Bring arms in line with shoulders"
                        }
                        if !legsSide {
                            g[.leftHip] = "Align legs sideways"
                            g[.rightHip] = "Align legs sideways"
                        }
                        if !g.isEmpty { guidance = g }
                    }

                    // Process pose with image size for validators that expect pixel scale
                    self.processPose(points, imageSize: currentImageSize)

                    // Save the computed state so the countdown timer can refresh visuals even when ML callbacks are sparse
                    self.latestImageSize = currentImageSize
                    self.lastAccuracy = accuracy
                    self.lastGuidance = guidance
                    self.lastMirrored = isMirrored
                    print("🎨 Updating overlay with \(points.count) landmarks, imageSize: \(currentImageSize), mirrored: \(isMirrored)")
                    DispatchQueue.main.async {
                        self.poseOverlayView.updatePose(points,
                            imageSize: currentImageSize,
                            accuracy: accuracy,
                            perfect: false,
                            countdown: self.countdownValue,
                            counting: self.isCapturing,
                            mirrored: isMirrored,
                            guidance: guidance,
                            stage: self.currentStage == .frontPose ? "front" : "side")
                    }
                }
            }
        }
    }
    
    private func processPose(_ landmarks: PoseLandmarks, imageSize: CGSize) {
        switch currentStage {
        case .frontPose:
            processFrontPose(landmarks, imageSize: imageSize)
        case .sidePose:
            processSidePose(landmarks, imageSize: imageSize)
        }
    }
    
    private func processFrontPose(_ landmarks: PoseLandmarks, imageSize: CGSize) {
        let result = bodyPositionChecker.checkPoseDetailed(landmarks)

        // Voice feedback — only when not already capturing
        if !isCapturing {
            voiceFeedback.provideFeedback(result.feedback)
        }

        // Feet must be in-frame AND full-body check must pass before triggering countdown
        if result.isValid && !isCapturing && feetInFrame(landmarks, imageSize: imageSize) {
            sendStatusEvent(status: "ready_to_capture", message: "Ready to capture front pose!")
            voiceFeedback.provideFeedback("Capturing in 3 seconds")
            startCountdown(for: .frontPose)
        }
    }
    
    private func processSidePose(_ landmarks: PoseLandmarks, imageSize: CGSize) {
        // Ensure person is detected (in frame) before side checks
        if !poseValidator.isValidPose(landmarks) {
            // Only provide feedback if not capturing to avoid spam
            if !isCapturing {
                voiceFeedback.provideFeedback("Please move into the frame")
            }
            return
        }

        // Require feet visible/in-frame before proceeding
        guard feetInFrame(landmarks, imageSize: imageSize) else {
            if !isCapturing {
                voiceFeedback.provideFeedback("Keep your feet in the frame")
            }
            return
        }

        let isSide = sidePoseValidator.isSidewaysPose(landmarks, imageSize: imageSize)
        let armsSide = sidePoseValidator.areArmsSideways(landmarks, imageSize: imageSize)
        let legsSide = sidePoseValidator.areLegsSideways(landmarks, imageSize: imageSize)

        if isSide && armsSide && legsSide && !isCapturing {
            sendStatusEvent(status: "ready_to_capture_side", message: "Ready to capture side pose!")
            voiceFeedback.provideFeedback("Capturing side pose in 3 seconds")
            startCountdown(for: .sidePose)
            return
        }
    }

    private func cancelCountdown(reason: String?) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            if let timer = self.countdownTimer {
                timer.invalidate()
                self.countdownTimer = nil
            }
            self.isCapturing = false
            self.countdownValue = 3
            self.countdownLabel.isHidden = true

            if let reason = reason {
                print("⛔ Countdown cancelled: \(reason)")
            }
            // Notify JS and voice
            self.sendStatusEvent(status: "capture_cancelled", message: "Pose lost, please hold still")
            self.voiceFeedback.provideFeedback("Pose lost, please hold still")
        }
    }

    private func startCountdown(for stage: PoseStage) {
        // If already counting down, do nothing
        guard countdownTimer == nil else { return }

        isCapturing = true
        countdownValue = 3

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            self.countdownLabel.isHidden = false
            self.countdownLabel.text = "\(self.countdownValue)"
            // Speak the initial countdown
            self.voiceFeedback.provideFeedback("\(self.countdownValue)")

            // 2 second intervals give voice feedback time to complete before next count
            self.countdownTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] timer in
                guard let self = self else {
                    timer.invalidate()
                    return
                }

                // Before decrementing, ensure pose still meets requirements for the stage
                var stillValid = false
                if self.currentStage == .frontPose {
                    // Re-evaluate using BodyPositionChecker logic
                    let (isValid, _) = self.bodyPositionChecker.checkPose(self.poseOverlayView.getCurrentLandmarks() ?? [:])
                    stillValid = isValid
                } else {
                    // Side pose: ensure person in frame and side pose validity
                    let currentLandmarks = self.poseOverlayView.getCurrentLandmarks() ?? [:]
                    if let size = self.latestImageSize, self.poseValidator.isValidPose(currentLandmarks) {
                        stillValid = self.sidePoseValidator.isValidSidePose(currentLandmarks, imageSize: size)
                    } else {
                        stillValid = false
                    }
                }

                if !stillValid {
                    // Cancel countdown and notify - will restart automatically when pose is valid again
                    timer.invalidate()
                    self.countdownTimer = nil
                    self.isCapturing = false
                    self.countdownLabel.isHidden = true
                    self.countdownValue = 3
                    print("⚠️ Countdown cancelled - pose became invalid")
                    self.sendStatusEvent(status: "capture_cancelled", message: "Hold your pose steady")
                    self.voiceFeedback.provideFeedback("Hold your pose steady")
                    return
                }

                self.countdownValue -= 1

                if self.countdownValue > 0 {
                    self.countdownLabel.text = "\(self.countdownValue)"
                    // Note: No need to refresh overlay here - captureOutput() already updates it continuously
                    // with the latest landmarks from the camera, even during countdown
                    
                    // Speak remaining second (will respect min interval)
                    self.voiceFeedback.provideFeedback("\(self.countdownValue)")
                } else {
                    // Reached zero - hold briefly before capture so user sees final frame
                    timer.invalidate()
                    self.countdownTimer = nil
                    self.countdownLabel.isHidden = true

                    // Small pause so the camera/overlay stabilizes and user sees final pose
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.7) {
                        self.captureImage(for: stage)
                    }
                }
            }
        }
    }
    
    private func captureImage(for stage: PoseStage) {
        // Guard: prevent access if object is deallocating
        guard captureSession != nil else {
            print("❌ Capture aborted: object deallocating")
            isCapturing = false
            return
        }
        guard let sampleBuffer = latestSampleBuffer else {
            print("No sample buffer available")
            isCapturing = false
            return
        }
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get image buffer")
            isCapturing = false
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            print("Failed to create CGImage")
            isCapturing = false
            return
        }
        
        // CVPixelBuffer from camera is already in correct orientation when connection.videoOrientation = .portrait
        // Just apply mirroring for front camera
        let orientation: UIImage.Orientation = currentCamera?.position == .front ? .upMirrored : .up
        let image = UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
        
        saveImage(image, for: stage)
    }
    
    private func saveImage(_ image: UIImage, for stage: PoseStage) {
        guard let data = image.jpegData(compressionQuality: 0.65) else {
            isCapturing = false
            return
        }
        
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let fileName = stage == .frontPose ? "front_pose_\(Date().timeIntervalSince1970).jpg" : "side_pose_\(Date().timeIntervalSince1970).jpg"
        let fileURL = cacheDir.appendingPathComponent(fileName)
        
        do {
            try data.write(to: fileURL)
            
            if stage == .frontPose {
                frontImagePath = fileURL.path
                sendStatusEvent(status: "front_pose_captured", message: "Front pose captured! Turn sideways...")

                // Voice feedback will handle main thread dispatch internally
                voiceFeedback.provideFeedback("Front pose captured")

                currentStage = .sidePose
                isCapturing = false
            } else {
                 sideImagePath = fileURL.path
                 print("📸 SIDE IMAGE SAVED: \(fileURL.path)")
                 print("📸 FRONT IMAGE PATH: \(frontImagePath ?? "nil")")
                 sendStatusEvent(status: "both_poses_captured", message: "Both poses captured! Processing...")
                 
                 // Also emit DeviceEventEmitter event like Android so JS listeners using DeviceEventEmitter receive it
                 if let bridge = self.reactBridge {
                     let params = ["frontUri": "file://\(frontImagePath ?? "")", "sideUri": "file://\(sideImagePath ?? "")"]
                     print("🔔 EMITTING DeviceEventEmitter with params: \(params)")
                     bridge.enqueueJSCall("RCTDeviceEventEmitter", method: "emit", args: ["onBothImagesCaptured", params], completion: nil)
                     print("✅ DeviceEventEmitter.emit() called successfully")
                 } else {
                     print("❌ ERROR: reactBridge is nil, cannot emit DeviceEventEmitter event")
                 }
                 
                 // Notify RN via prop callback - This triggers navigation to ResultScreen
                 print("🚀 CALLING sendImagesToReactNative()...")
                 sendImagesToReactNative()
                 print("✅ sendImagesToReactNative() completed")

                 // CRITICAL: Stop capture session immediately to prevent camera pipeline crash
                 self.captureSession?.stopRunning()
                 print("🛑 Capture session stopped")
                 // Restore idle timer now that capture is done
                 DispatchQueue.main.async {
                     UIApplication.shared.isIdleTimerDisabled = false
                 }

                 // Aggressive memory cleanup before navigation to prevent Result screen hang
                 autoreleasepool {
                     self.latestSampleBuffer = nil
                 }
                 
                 // Clear overlay and reset detection state so RN preview can take over
                 DispatchQueue.main.async {
                     print("🧹 Clearing overlay and resetting state...")
                     self.poseOverlayView.updatePose(nil, imageSize: .zero, accuracy: nil, perfect: false, countdown: 0, counting: false, mirrored: false, guidance: nil, stage: nil)
                     self.poseOverlayView.setNeedsDisplay()
                     self.isCapturing = false
                     
                     // Reset to front pose for next capture session
                     self.currentStage = .frontPose
                     self.frontImagePath = nil
                     self.sideImagePath = nil
                     print("✅ State reset completed - ready for next capture")
                 }

                 DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                     self.resetDetectionState()
                 }
              }
            
        } catch {
            print("Error saving image: \(error)")
            isCapturing = false
        }
    }
    
    private func sendStatusEvent(status: String, message: String) {
        guard let onCaptureStatus = onCaptureStatus else { 
            print("Warning: onCaptureStatus is nil")
            return 
        }
        
        DispatchQueue.main.async {
            onCaptureStatus([
                "status": status,
                "message": message
            ])
        }
    }
    
    private func sendImagesToReactNative() {
        print("📤 sendImagesToReactNative() started")
        print("   frontImagePath: \(frontImagePath ?? "nil")")
        print("   sideImagePath: \(sideImagePath ?? "nil")")
        
        guard let frontPath = frontImagePath,
              let sidePath = sideImagePath else {
            print("❌ ERROR: Missing image paths - front: \(frontImagePath ?? "nil"), side: \(sideImagePath ?? "nil")")
            sendStatusEvent(status: "capture_incomplete", message: "Images not saved, please retake")
            return
        }
        
        print("✅ Image paths validated")
        print("   Front: \(frontPath)")
        print("   Side: \(sidePath)")
        
        guard let onBothCapturedHandler = onBothCaptured else {
            print("❌ ERROR: onBothCaptured callback is nil!")
            print("   This means the callback was not set by React Native component")
            return
        }
        
        print("✅ onBothCaptured callback exists")
        
        let fileManager = FileManager.default
        let frontExists = fileManager.fileExists(atPath: frontPath)
        let sideExists = fileManager.fileExists(atPath: sidePath)
        
        print("📁 File existence check:")
        print("   Front exists: \(frontExists)")
        print("   Side exists: \(sideExists)")
        
        guard frontExists, sideExists else {
            print("❌ ERROR: Image files missing on disk")
            sendStatusEvent(status: "capture_incomplete", message: "Images not saved, please retake")
            return
        }
        
        // Match ResultScreen param expectations: imageUri for front, sideImageUri for side
        let params = [
            "imageUri": "file://\(frontPath)",
            "sideImageUri": "file://\(sidePath)"
        ]
        
        print("🎯 INVOKING onBothCaptured callback with params:")
        print("   \(params)")
        
        DispatchQueue.main.async {
            onBothCapturedHandler(params)
            print("✅ onBothCaptured callback invoked successfully on main thread")
        }
    }

    // Ensure required ankle landmarks are in frame with adequate confidence
    private func feetInFrame(_ landmarks: PoseLandmarks, imageSize: CGSize, minConfidence: Float = 0.5) -> Bool {
        let keys: [JointName] = [.leftAnkle, .rightAnkle]
        for key in keys {
            guard let pt = landmarks[key], pt.confidence >= minConfidence else { return false }
            let x = pt.location.x
            let y = pt.location.y
            if x < 0 || y < 0 || x > imageSize.width || y > imageSize.height {
                return false
            }
        }
        return true
    }

    // ML Kit orientation helper per docs
    private func mlkitImageOrientation(deviceOrientation: UIDeviceOrientation, cameraPosition: AVCaptureDevice.Position) -> UIImage.Orientation {
        switch deviceOrientation {
        case .portrait:
            return cameraPosition == .front ? .leftMirrored : .right
        case .landscapeLeft:
            return cameraPosition == .front ? .downMirrored : .up
        case .portraitUpsideDown:
            return cameraPosition == .front ? .rightMirrored : .left
        case .landscapeRight:
            return cameraPosition == .front ? .upMirrored : .down
        case .faceDown, .faceUp, .unknown:
            return .up
        @unknown default:
            return .up
        }
    }
    
    override func willMove(toWindow newWindow: UIWindow?) {
        super.willMove(toWindow: newWindow)
        if newWindow == nil {
            // View is being removed — restore idle timer
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    deinit {
        captureSession?.stopRunning()
        latestSampleBuffer = nil
        countdownTimer?.invalidate()
        countdownTimer = nil
        voiceFeedback.stop()
        // Ensure idle timer is always restored
        DispatchQueue.main.async {
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }
}
