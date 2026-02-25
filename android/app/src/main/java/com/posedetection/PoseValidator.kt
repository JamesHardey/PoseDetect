package com.posedetection

import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark
import kotlin.math.*

class PoseValidator {
    
    companion object {
        private const val TAG = "PoseValidator"
    }
    
    fun calculateAngle(firstPoint: PoseLandmark, midPoint: PoseLandmark, lastPoint: PoseLandmark): Double {
        val result = Math.toDegrees(
            atan2(lastPoint.position.y - midPoint.position.y.toDouble(),
                lastPoint.position.x - midPoint.position.x.toDouble()) -
                    atan2(firstPoint.position.y - midPoint.position.y.toDouble(),
                        firstPoint.position.x - midPoint.position.x.toDouble())
        )
        var finalResult = abs(result)
        if (finalResult > 180) {
            finalResult = 360 - finalResult
        }
        return finalResult
    }
    
    fun calculatePostureMetrics(pose: Pose): PostureMetrics {
        val leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)
        val rightShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)
        val leftElbow = pose.getPoseLandmark(PoseLandmark.LEFT_ELBOW)
        val rightElbow = pose.getPoseLandmark(PoseLandmark.RIGHT_ELBOW)
        val leftWrist = pose.getPoseLandmark(PoseLandmark.LEFT_WRIST)
        val rightWrist = pose.getPoseLandmark(PoseLandmark.RIGHT_WRIST)
        val leftHip = pose.getPoseLandmark(PoseLandmark.LEFT_HIP)
        val rightHip = pose.getPoseLandmark(PoseLandmark.RIGHT_HIP)
        val leftKnee = pose.getPoseLandmark(PoseLandmark.LEFT_KNEE)
        val rightKnee = pose.getPoseLandmark(PoseLandmark.RIGHT_KNEE)
        val leftAnkle = pose.getPoseLandmark(PoseLandmark.LEFT_ANKLE)
        val rightAnkle = pose.getPoseLandmark(PoseLandmark.RIGHT_ANKLE)
        
        val shoulderAngleLeft = if (leftShoulder != null && leftElbow != null && leftHip != null) {
            calculateAngle(leftElbow, leftShoulder, leftHip)
        } else 999.0

        val shoulderAngleRight = if (rightShoulder != null && rightElbow != null && rightHip != null) {
            calculateAngle(rightElbow, rightShoulder, rightHip)
        } else 999.0

        val elbowAngleLeft = if (leftShoulder != null && leftElbow != null && leftWrist != null) {
            calculateAngle(leftShoulder, leftElbow, leftWrist)
        } else 999.0

        val elbowAngleRight = if (rightShoulder != null && rightElbow != null && rightWrist != null) {
            calculateAngle(rightShoulder, rightElbow, rightWrist)
        } else 999.0

        val spineAngle = if (leftShoulder != null && rightShoulder != null && leftHip != null && rightHip != null) {
            val shoulderMidX = (leftShoulder.position.x + rightShoulder.position.x) / 2
            val shoulderMidY = (leftShoulder.position.y + rightShoulder.position.y) / 2
            val hipMidX = (leftHip.position.x + rightHip.position.x) / 2
            val hipMidY = (leftHip.position.y + rightHip.position.y) / 2
            
            val angleFromVertical = atan2(
                (hipMidX - shoulderMidX).toDouble(),
                (hipMidY - shoulderMidY).toDouble()
            ) * 180 / PI
            
            abs(angleFromVertical)
        } else 999.0

        val hipAngleLeft = if (leftHip != null && leftKnee != null && leftAnkle != null) {
            calculateAngle(leftShoulder ?: leftHip, leftHip, leftKnee)
        } else 999.0

        val hipAngleRight = if (rightHip != null && rightKnee != null && rightAnkle != null) {
            calculateAngle(rightShoulder ?: rightHip, rightHip, rightKnee)
        } else 999.0

        val shoulderLevelDiff = if (leftShoulder != null && rightShoulder != null) {
            abs(leftShoulder.position.y - rightShoulder.position.y).toDouble()
        } else 999.0
        
        val legSeparationAngle = if (leftKnee != null && rightKnee != null && leftHip != null && rightHip != null) {
            val hipMidX = (leftHip.position.x + rightHip.position.x) / 2
            val hipMidY = (leftHip.position.y + rightHip.position.y) / 2
            
            val leftLegAngle = atan2(
                (leftKnee.position.y - hipMidY).toDouble(),
                (leftKnee.position.x - hipMidX).toDouble()
            ) * 180 / PI
            
            val rightLegAngle = atan2(
                (rightKnee.position.y - hipMidY).toDouble(),
                (rightKnee.position.x - hipMidX).toDouble()
            ) * 180 / PI
            
            abs(leftLegAngle - rightLegAngle)
        } else {
            999.0
        }

        return PostureMetrics(
            shoulderAngleLeft,
            shoulderAngleRight,
            elbowAngleLeft,
            elbowAngleRight,
            spineAngle,
            hipAngleLeft,
            hipAngleRight,
            shoulderLevelDiff,
            legSeparationAngle
        )
    }
    
    fun compareWithReference(current: PostureMetrics, reference: ReferencePose): PostureAccuracy {
        val shoulderDiffL = abs(current.shoulderAngleLeft - reference.shoulderAngle)
        val shoulderDiffR = abs(current.shoulderAngleRight - reference.shoulderAngle)
        val elbowDiffL = abs(current.elbowAngleLeft - reference.elbowAngleLeft)
        val elbowDiffR = abs(current.elbowAngleRight - reference.elbowAngleRight)
        val spineDiff = current.spineAngle
        val hipDiffL = abs(current.hipAngleLeft - reference.hipAngleLeft)
        val hipDiffR = abs(current.hipAngleRight - reference.hipAngleRight)
        val legSeparationDiff = abs(current.legSeparationAngle - reference.legSeparationAngle)

        return PostureAccuracy(
            shoulderAccurateLeft = shoulderDiffL <= reference.shoulderAngleTolerance,
            shoulderAccurateRight = shoulderDiffR <= reference.shoulderAngleTolerance,
            elbowAccurateLeft = elbowDiffL <= reference.elbowAngleTolerance,
            elbowAccurateRight = elbowDiffR <= reference.elbowAngleTolerance,
            spineAccurate = spineDiff <= reference.spineAngleTolerance,
            hipAccurateLeft = hipDiffL <= reference.hipAngleTolerance,
            hipAccurateRight = hipDiffR <= reference.hipAngleTolerance,
            shoulderLevelAccurate = current.shoulderLevelDiff <= reference.shoulderLevelTolerance,
            legSeparationAccurate = legSeparationDiff <= reference.legSeparationTolerance,
            shoulderDiffLeft = shoulderDiffL,
            shoulderDiffRight = shoulderDiffR,
            elbowDiffLeft = elbowDiffL,
            elbowDiffRight = elbowDiffR,
            spineDiff = spineDiff,
            hipDiffLeft = hipDiffL,
            hipDiffRight = hipDiffR,
            legSeparationDiff = legSeparationDiff
        )
    }
    
    fun isPoseAccurate(accuracy: PostureAccuracy, positionCheck: PositionCheck): Boolean {
        return accuracy.shoulderAccurateLeft &&
               accuracy.shoulderAccurateRight &&
               accuracy.elbowAccurateLeft &&
               accuracy.elbowAccurateRight &&
               accuracy.spineAccurate &&
               accuracy.hipAccurateLeft &&
               accuracy.hipAccurateRight &&
               accuracy.shoulderLevelAccurate &&
               accuracy.legSeparationAccurate &&
               positionCheck.inBox
    }
    
    fun isPersonFullyDetected(pose: Pose): Boolean {
        val criticalLandmarks = listOf(
            PoseLandmark.NOSE,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.LEFT_WRIST,
            PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.RIGHT_KNEE
        )
        
        return criticalLandmarks.all { landmarkType ->
            pose.getPoseLandmark(landmarkType) != null
        }
    }
}
