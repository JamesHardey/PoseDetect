package com.posedetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark
import kotlin.math.abs

class PoseOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var bitmap: Bitmap? = null
    private var pose: Pose? = null
    private var accuracy: PostureAccuracy? = null
    private var sideAccuracy: SidePoseAccuracy? = null
    private var isPerfectPose: Boolean = false
    private var countdownValue: Int = 0
    private var isCountingDown: Boolean = false
    private var currentStage: PoseStage = PoseStage.FRONT_POSE

    private val goodPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL_AND_STROKE
        strokeWidth = 5f
    }

    private val badPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL_AND_STROKE
        strokeWidth = 5f
    }

    private val goodLinePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val badLinePaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val boxPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 3f
        pathEffect = DashPathEffect(floatArrayOf(20f, 10f), 0f)
    }

    private val boxFillPaint = Paint().apply {
        color = Color.argb(30, 255, 255, 0)
        style = Paint.Style.FILL
    }

    private val countdownPaint = Paint().apply {
        color = Color.WHITE
        textSize = 120f
        textAlign = Paint.Align.CENTER
        style = Paint.Style.FILL
        setShadowLayer(10f, 0f, 0f, Color.BLACK)
    }

    private val statusPaint = Paint().apply {
        color = Color.GREEN
        textSize = 50f
        textAlign = Paint.Align.CENTER
        style = Paint.Style.FILL
        setShadowLayer(8f, 0f, 0f, Color.BLACK)
    }

    // Target box for user positioning (relative coordinates 0-1)
    private val targetBox = TargetBox(
        left = 0.05f,
        top = 0.05f,
        right = 0.95f,
        bottom = 0.95f
    )

    fun updatePose(
        newPose: Pose?,
        newAccuracy: PostureAccuracy?,
        perfect: Boolean,
        countdown: Int,
        counting: Boolean,
        stage: PoseStage = PoseStage.FRONT_POSE
    ) {
        pose = newPose
        accuracy = newAccuracy
        sideAccuracy = null
        isPerfectPose = perfect
        countdownValue = countdown
        isCountingDown = counting
        currentStage = stage
        invalidate()
    }
    
    fun updateSidePose(
        newPose: Pose?,
        newSideAccuracy: SidePoseAccuracy?,
        perfect: Boolean,
        countdown: Int,
        counting: Boolean,
        stage: PoseStage = PoseStage.SIDE_POSE
    ) {
        pose = newPose
        accuracy = null
        sideAccuracy = newSideAccuracy
        isPerfectPose = perfect
        countdownValue = countdown
        isCountingDown = counting
        currentStage = stage
        invalidate()
    }

    fun updateBitmap(newBitmap: Bitmap?) {
        bitmap = newBitmap
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw camera preview bitmap if available
        bitmap?.let { bmp ->
            val srcRect = Rect(0, 0, bmp.width, bmp.height)
            val dstRect = Rect(0, 0, width, height)
            canvas.drawBitmap(bmp, srcRect, dstRect, null)
        }

        val canvasWidth = width.toFloat()
        val canvasHeight = height.toFloat()

        // Draw target box
        drawTargetBox(canvas, canvasWidth, canvasHeight)

        // Draw pose skeleton
        pose?.let { p ->
            val bitmapWidth = bitmap?.width ?: 1
            val bitmapHeight = bitmap?.height ?: 1
            
            // Draw connections even without accuracy data
            if (accuracy != null) {
                drawPoseConnectionsWithAccuracy(canvas, p, accuracy!!, canvasWidth, canvasHeight)
                
                // Draw landmarks with accuracy colors
                for (poseLandmark in p.allPoseLandmarks) {
                    val paint = if (isLandmarkAccurate(poseLandmark, accuracy!!)) goodPaint else badPaint
                    val scaledX = poseLandmark.position.x * canvasWidth / bitmapWidth
                    val scaledY = poseLandmark.position.y * canvasHeight / bitmapHeight
                    canvas.drawCircle(scaledX, scaledY, 8f, paint)
                }
            } else {
                // Draw connections with default color when no accuracy data
                drawPoseConnections(canvas, p, canvasWidth, canvasHeight)
                
                // Draw landmarks with default color
                for (poseLandmark in p.allPoseLandmarks) {
                    val scaledX = poseLandmark.position.x * canvasWidth / bitmapWidth
                    val scaledY = poseLandmark.position.y * canvasHeight / bitmapHeight
                    canvas.drawCircle(scaledX, scaledY, 8f, goodPaint)
                }
            }
        }

        // Draw countdown or status text
        if (isCountingDown && countdownValue > 0) {
            canvas.drawText(
                countdownValue.toString(),
                canvasWidth / 2f,
                canvasHeight / 2f,
                countdownPaint
            )
        } else if (isPerfectPose) {
            canvas.drawText(
                "PERFECT POSE!",
                canvasWidth / 2f,
                100f,
                statusPaint
            )
        }
        
        // Draw stage instruction
        val stageText = when (currentStage) {
            PoseStage.FRONT_POSE -> "Front Pose: Stand facing camera"
            PoseStage.SIDE_POSE -> "Side Pose: Turn sideways to camera"
        }
        statusPaint.color = if (currentStage == PoseStage.FRONT_POSE) Color.GREEN else Color.CYAN
        canvas.drawText(
            stageText,
            canvasWidth / 2f,
            canvasHeight - 100f,
            statusPaint
        )
    }
    
    private fun drawPoseConnections(
        canvas: Canvas,
        pose: Pose,
        canvasWidth: Float,
        canvasHeight: Float
    ) {
        val bitmapWidth = bitmap?.width ?: 1
        val bitmapHeight = bitmap?.height ?: 1

        val connections = listOf(
            Triple(PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER, "shoulder_level"),
            Triple(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, "left_arm"),
            Triple(PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST, "left_forearm"),
            Triple(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, "right_arm"),
            Triple(PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST, "right_forearm"),
            Triple(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP, "left_torso"),
            Triple(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP, "right_torso"),
            Triple(PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP, "hips"),
            Triple(PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, "left_thigh"),
            Triple(PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE, "left_shin"),
            Triple(PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, "right_thigh"),
            Triple(PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE, "right_shin")
        )

        for ((startType, endType, _) in connections) {
            val startLandmark = pose.getPoseLandmark(startType)
            val endLandmark = pose.getPoseLandmark(endType)

            if (startLandmark != null && endLandmark != null) {
                val startX = startLandmark.position.x * canvasWidth / bitmapWidth
                val startY = startLandmark.position.y * canvasHeight / bitmapHeight
                val endX = endLandmark.position.x * canvasWidth / bitmapWidth
                val endY = endLandmark.position.y * canvasHeight / bitmapHeight

                canvas.drawLine(startX, startY, endX, endY, goodLinePaint)
            }
        }
    }

    private fun drawTargetBox(canvas: Canvas, width: Float, height: Float) {
        val left = targetBox.left * width
        val top = targetBox.top * height
        val right = targetBox.right * width
        val bottom = targetBox.bottom * height

        canvas.drawRect(left, top, right, bottom, boxFillPaint)
        canvas.drawRect(left, top, right, bottom, boxPaint)
    }

    private fun isLandmarkAccurate(landmark: PoseLandmark, accuracy: PostureAccuracy): Boolean {
        return when (landmark.landmarkType) {
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW -> 
                accuracy.shoulderAccurateLeft && accuracy.elbowAccurateLeft
            PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW -> 
                accuracy.shoulderAccurateRight && accuracy.elbowAccurateRight
            PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE -> accuracy.hipAccurateLeft
            PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE -> accuracy.hipAccurateRight
            else -> true
        }
    }

    private fun drawPoseConnectionsWithAccuracy(
        canvas: Canvas,
        pose: Pose,
        accuracy: PostureAccuracy,
        canvasWidth: Float,
        canvasHeight: Float
    ) {
        val bitmapWidth = bitmap?.width ?: 1
        val bitmapHeight = bitmap?.height ?: 1

        val connections = listOf(
            Triple(PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER, "shoulder_level"),
            Triple(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, "left_arm"),
            Triple(PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST, "left_forearm"),
            Triple(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, "right_arm"),
            Triple(PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST, "right_forearm"),
            Triple(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP, "left_torso"),
            Triple(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP, "right_torso"),
            Triple(PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP, "hips"),
            Triple(PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, "left_thigh"),
            Triple(PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE, "left_shin"),
            Triple(PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, "right_thigh"),
            Triple(PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE, "right_shin")
        )

        for ((startType, endType, part) in connections) {
            val startLandmark = pose.getPoseLandmark(startType)
            val endLandmark = pose.getPoseLandmark(endType)

            if (startLandmark != null && endLandmark != null) {
                val isAccurate = when (part) {
                    "left_arm", "left_forearm" -> accuracy.shoulderAccurateLeft && accuracy.elbowAccurateLeft
                    "right_arm", "right_forearm" -> accuracy.shoulderAccurateRight && accuracy.elbowAccurateRight
                    "left_torso", "right_torso" -> accuracy.spineAccurate
                    "left_thigh", "left_shin" -> accuracy.hipAccurateLeft
                    "right_thigh", "right_shin" -> accuracy.hipAccurateRight
                    "shoulder_level" -> accuracy.shoulderLevelAccurate
                    else -> true
                }

                val paint = if (isAccurate) goodLinePaint else badLinePaint

                val startX = startLandmark.position.x * canvasWidth / bitmapWidth
                val startY = startLandmark.position.y * canvasHeight / bitmapHeight
                val endX = endLandmark.position.x * canvasWidth / bitmapWidth
                val endY = endLandmark.position.y * canvasHeight / bitmapHeight

                canvas.drawLine(startX, startY, endX, endY, paint)
            }
        }
    }
}

// Data classes
data class PostureMetrics(
    val shoulderAngleLeft: Double,
    val shoulderAngleRight: Double,
    val elbowAngleLeft: Double,
    val elbowAngleRight: Double,
    val spineAngle: Double,
    val hipAngleLeft: Double,
    val hipAngleRight: Double,
    val shoulderLevelDiff: Double,
    val legSeparationAngle: Double
)

data class ReferencePose(
    val shoulderAngle: Double,
    val shoulderAngleTolerance: Double,
    val elbowAngleLeft: Double,
    val elbowAngleRight: Double,
    val elbowAngleTolerance: Double,
    val spineAngle: Double,
    val spineAngleTolerance: Double,
    val hipAngleLeft: Double,
    val hipAngleRight: Double,
    val hipAngleTolerance: Double,
    val shoulderLevelDiff: Double,
    val shoulderLevelTolerance: Double,
    val legSeparationAngle: Double,
    val legSeparationTolerance: Double
)

data class PostureAccuracy(
    val shoulderAccurateLeft: Boolean,
    val shoulderAccurateRight: Boolean,
    val elbowAccurateLeft: Boolean,
    val elbowAccurateRight: Boolean,
    val spineAccurate: Boolean,
    val hipAccurateLeft: Boolean,
    val hipAccurateRight: Boolean,
    val shoulderLevelAccurate: Boolean,
    val legSeparationAccurate: Boolean,

    val shoulderDiffLeft: Double,
    val shoulderDiffRight: Double,
    val elbowDiffLeft: Double,
    val elbowDiffRight: Double,
    val spineDiff: Double,
    val hipDiffLeft: Double,
    val hipDiffRight: Double,
    val legSeparationDiff: Double
)

data class TargetBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

data class PositionCheck(
    val inBox: Boolean,
    val issues: List<String>
)

enum class PoseStage {
    FRONT_POSE,
    SIDE_POSE
}

data class SidePoseReference(
    val neckHeadAngle: Double,
    val neckHeadTolerance: Double,
    val armAngle: Double,
    val armTolerance: Double,
    val spineVerticalAngle: Double,
    val spineTolerance: Double,
    val legStraightAngle: Double,
    val legTolerance: Double,
    val shoulderDepthDiff: Double,
    val shoulderDepthTolerance: Double
)

data class SidePoseMetrics(
    val neckHeadAngle: Double,
    val leftArmAngle: Double,
    val rightArmAngle: Double,
    val spineAngle: Double,
    val leftLegAngle: Double,
    val rightLegAngle: Double,
    val shoulderDepthDiff: Double,
    val shoulderHorizontalAlignment: Double,
    val armsOverlapping: Double,
    val legsOverlapping: Double
)

data class SidePoseAccuracy(
    val neckHeadAccurate: Boolean,
    val leftArmAccurate: Boolean,
    val rightArmAccurate: Boolean,
    val spineAccurate: Boolean,
    val leftLegAccurate: Boolean,
    val rightLegAccurate: Boolean,
    val isSideView: Boolean,
    val neckHeadDiff: Double,
    val leftArmDiff: Double,
    val rightArmDiff: Double,
    val spineDiff: Double,
    val leftLegDiff: Double,
    val rightLegDiff: Double
)
