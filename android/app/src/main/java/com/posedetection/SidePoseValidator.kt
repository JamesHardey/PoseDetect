package com.posedetection

import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark
import kotlin.math.*

class SidePoseValidator {
    
    private val poseValidator = PoseValidator()
    
    fun calculateSidePoseMetrics(pose: Pose): SidePoseMetrics {
        val nose = pose.getPoseLandmark(PoseLandmark.NOSE)
        val leftEar = pose.getPoseLandmark(PoseLandmark.LEFT_EAR)
        val rightEar = pose.getPoseLandmark(PoseLandmark.RIGHT_EAR)
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
        
        // Use the ear that's more visible
        val ear = if (leftEar != null && rightEar != null) {
            if (leftEar.inFrameLikelihood > rightEar.inFrameLikelihood) leftEar else rightEar
        } else {
            leftEar ?: rightEar
        }
        
        // Calculate shoulder midpoint for neck position
        val shoulderMidPoint = if (leftShoulder != null && rightShoulder != null) {
            android.graphics.PointF(
                (leftShoulder.position.x + rightShoulder.position.x) / 2f,
                (leftShoulder.position.y + rightShoulder.position.y) / 2f
            )
        } else {
            leftShoulder?.position ?: rightShoulder?.position
        }
        
        val neckHeadAngle = if (ear != null && shoulderMidPoint != null && nose != null) {
            val neckToEar = atan2(
                (ear.position.y - shoulderMidPoint.y).toDouble(),
                (ear.position.x - shoulderMidPoint.x).toDouble()
            ) * 180 / PI
            abs(neckToEar - 90.0)
        } else {
            999.0
        }
        
        val leftArmAngle = if (leftShoulder != null && leftElbow != null && leftWrist != null) {
            poseValidator.calculateAngle(leftShoulder, leftElbow, leftWrist)
        } else {
            999.0
        }
        
        val rightArmAngle = if (rightShoulder != null && rightElbow != null && rightWrist != null) {
            poseValidator.calculateAngle(rightShoulder, rightElbow, rightWrist)
        } else {
            999.0
        }
        
        val spineAngle = if (leftShoulder != null && rightShoulder != null && leftHip != null && rightHip != null) {
            val shoulderMid = android.graphics.PointF(
                (leftShoulder.position.x + rightShoulder.position.x) / 2f,
                (leftShoulder.position.y + rightShoulder.position.y) / 2f
            )
            val hipMid = android.graphics.PointF(
                (leftHip.position.x + rightHip.position.x) / 2f,
                (leftHip.position.y + rightHip.position.y) / 2f
            )
            
            val angleFromVertical = atan2(
                (hipMid.x - shoulderMid.x).toDouble(),
                (hipMid.y - shoulderMid.y).toDouble()
            ) * 180 / PI
            abs(angleFromVertical)
        } else {
            999.0
        }
        
        val leftLegAngle = if (leftHip != null && leftKnee != null && leftAnkle != null) {
            poseValidator.calculateAngle(leftHip, leftKnee, leftAnkle)
        } else {
            999.0
        }
        
        val rightLegAngle = if (rightHip != null && rightKnee != null && rightAnkle != null) {
            poseValidator.calculateAngle(rightHip, rightKnee, rightAnkle)
        } else {
            999.0
        }
        
        val shoulderDepthDiff = if (leftShoulder != null && rightShoulder != null) {
            // For side pose: shoulders should be vertically aligned (small Y difference)
            // and horizontally close (small X difference means one behind the other)
            abs(leftShoulder.position.y - rightShoulder.position.y).toDouble()
        } else {
            999.0
        }
        
        // Check if shoulders are aligned horizontally (one directly behind the other)
        val shoulderHorizontalAlignment = if (leftShoulder != null && rightShoulder != null) {
            abs(leftShoulder.position.x - rightShoulder.position.x).toDouble()
        } else {
            999.0
        }
        
        // Check if arms are overlapping (side view)
        val armsOverlapping = if (leftElbow != null && rightElbow != null && leftWrist != null && rightWrist != null) {
            val elbowXDiff = abs(leftElbow.position.x - rightElbow.position.x)
            val wristXDiff = abs(leftWrist.position.x - rightWrist.position.x)
            (elbowXDiff + wristXDiff) / 2.0
        } else {
            999.0
        }
        
        // Check if legs are overlapping (side view)
        val legsOverlapping = if (leftKnee != null && rightKnee != null && leftAnkle != null && rightAnkle != null) {
            val kneeXDiff = abs(leftKnee.position.x - rightKnee.position.x)
            val ankleXDiff = abs(leftAnkle.position.x - rightAnkle.position.x)
            (kneeXDiff + ankleXDiff) / 2.0
        } else {
            999.0
        }
        
        return SidePoseMetrics(
            neckHeadAngle = neckHeadAngle,
            leftArmAngle = leftArmAngle,
            rightArmAngle = rightArmAngle,
            spineAngle = spineAngle,
            leftLegAngle = leftLegAngle,
            rightLegAngle = rightLegAngle,
            shoulderDepthDiff = shoulderDepthDiff,
            shoulderHorizontalAlignment = shoulderHorizontalAlignment,
            armsOverlapping = armsOverlapping,
            legsOverlapping = legsOverlapping
        )
    }
    
    fun compareWithSideReference(current: SidePoseMetrics, reference: SidePoseReference): SidePoseAccuracy {
        val neckHeadDiff = abs(current.neckHeadAngle - reference.neckHeadAngle)
        val leftArmDiff = abs(current.leftArmAngle - reference.armAngle)
        val rightArmDiff = abs(current.rightArmAngle - reference.armAngle)
        val spineDiff = abs(current.spineAngle - reference.spineVerticalAngle)
        val leftLegDiff = abs(current.leftLegAngle - reference.legStraightAngle)
        val rightLegDiff = abs(current.rightLegAngle - reference.legStraightAngle)
        
        // For true side view: shoulders should be vertically aligned (small Y difference)
        // and horizontally close/overlapping (small X difference means side-on)
        // AND both arms should be overlapping AND both legs should be overlapping
        val isSideView = current.shoulderDepthDiff <= reference.shoulderDepthTolerance &&
                         current.shoulderHorizontalAlignment <= 100.0 && // Shoulders should overlap horizontally
                         current.armsOverlapping <= 120.0 && // Arms should be close together (overlapping)
                         current.legsOverlapping <= 120.0 // Legs should be close together (overlapping)
        
        return SidePoseAccuracy(
            neckHeadAccurate = neckHeadDiff <= reference.neckHeadTolerance,
            leftArmAccurate = leftArmDiff <= reference.armTolerance,
            rightArmAccurate = rightArmDiff <= reference.armTolerance,
            spineAccurate = spineDiff <= reference.spineTolerance,
            leftLegAccurate = leftLegDiff <= reference.legTolerance,
            rightLegAccurate = rightLegDiff <= reference.legTolerance,
            isSideView = isSideView,
            neckHeadDiff = neckHeadDiff,
            leftArmDiff = leftArmDiff,
            rightArmDiff = rightArmDiff,
            spineDiff = spineDiff,
            leftLegDiff = leftLegDiff,
            rightLegDiff = rightLegDiff
        )
    }
    
    fun isSidePoseAccurate(accuracy: SidePoseAccuracy, positionCheck: PositionCheck): Boolean {
        return accuracy.isSideView &&
               accuracy.neckHeadAccurate &&
               (accuracy.leftArmAccurate || accuracy.rightArmAccurate) &&
               accuracy.spineAccurate &&
               (accuracy.leftLegAccurate || accuracy.rightLegAccurate) &&
               positionCheck.inBox
    }
}
