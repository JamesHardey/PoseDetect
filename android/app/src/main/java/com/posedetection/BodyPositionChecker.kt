package com.posedetection

import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark

class BodyPositionChecker {
    
    fun checkBodyPosition(pose: Pose, width: Int, height: Int): PositionCheck {
        val issues = mutableListOf<String>()
        val targetBox = TargetBox(0.05f, 0.05f, 0.95f, 0.95f)

        // Get all landmarks
        val leftAnkle = pose.getPoseLandmark(PoseLandmark.LEFT_ANKLE)
        val rightAnkle = pose.getPoseLandmark(PoseLandmark.RIGHT_ANKLE)
        val leftHeel = pose.getPoseLandmark(PoseLandmark.LEFT_HEEL)
        val rightHeel = pose.getPoseLandmark(PoseLandmark.RIGHT_HEEL)
        val leftFootIndex = pose.getPoseLandmark(PoseLandmark.LEFT_FOOT_INDEX)
        val rightFootIndex = pose.getPoseLandmark(PoseLandmark.RIGHT_FOOT_INDEX)
        val leftWrist = pose.getPoseLandmark(PoseLandmark.LEFT_WRIST)
        val rightWrist = pose.getPoseLandmark(PoseLandmark.RIGHT_WRIST)
        val nose = pose.getPoseLandmark(PoseLandmark.NOSE)
        val leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)
        val rightShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)
        val leftKnee = pose.getPoseLandmark(PoseLandmark.LEFT_KNEE)
        val rightKnee = pose.getPoseLandmark(PoseLandmark.RIGHT_KNEE)

        // Check for foot visibility first (most important)
        val footLandmarks = listOfNotNull(
            leftAnkle, rightAnkle,
            leftHeel, rightHeel,
            leftFootIndex, rightFootIndex
        )
        
        val kneeLandmarks = listOfNotNull(leftKnee, rightKnee)
        val headLandmarks = listOfNotNull(nose)
        
        // Priority 1: Check if feet are visible
        val feetVisible = footLandmarks.isNotEmpty()
        val kneesVisible = kneeLandmarks.isNotEmpty()
        val headVisible = headLandmarks.isNotEmpty()
        
        if (!feetVisible && kneesVisible) {
            // Knees visible but feet are not - user is too close
            issues.add("Move back so your feet are visible")
            return PositionCheck(false, issues)
        }
        
        if (!headVisible && feetVisible) {
            // Feet visible but head is not - user is too far
            issues.add("Move forward so your head is visible")
            return PositionCheck(false, issues)
        }
        
        if (!feetVisible && !headVisible) {
            // Neither feet nor head visible
            issues.add("Please stand in front of camera")
            return PositionCheck(false, issues)
        }

        // Include all important landmarks for positioning
        val landmarks = listOfNotNull(
            leftAnkle, rightAnkle,
            leftHeel, rightHeel,
            leftFootIndex, rightFootIndex,
            leftWrist, rightWrist,
            nose, leftShoulder, rightShoulder
        )

        // Check if all visible landmarks are within the box
        var allInBox = true
        var tooFarLeft = false
        var tooFarRight = false
        var tooHigh = false
        var tooLow = false

        for (landmark in landmarks) {
            val x = landmark.position.x / width
            val y = landmark.position.y / height

            if (x < targetBox.left) {
                tooFarLeft = true
                allInBox = false
            }
            if (x > targetBox.right) {
                tooFarRight = true
                allInBox = false
            }
            if (y < targetBox.top) {
                tooHigh = true
                allInBox = false
            }
            if (y > targetBox.bottom) {
                tooLow = true
                allInBox = false
            }
        }

        // Provide specific feedback
        if (!allInBox) {
            when {
                tooLow && tooHigh -> issues.add("Move back to fit in frame")
                tooLow -> issues.add("Move back so your feet are in frame")
                tooHigh -> issues.add("Move forward so your head is in frame")
                tooFarLeft && tooFarRight -> issues.add("Step back to fit in frame")
                tooFarLeft -> issues.add("Move to your right")
                tooFarRight -> issues.add("Move to your left")
            }
        }

        return PositionCheck(allInBox, issues)
    }
}
