import Foundation
import Vision
import CoreGraphics

public class BodyPositionChecker {
    private let poseValidator = PoseValidator()
    
    public init() {}
    
    public func checkPose(_ landmarks: PoseLandmarks) -> (isValid: Bool, feedback: String) {
        // Priority 1: Check if feet are visible (matching Android's foot visibility check)
        let hasLeftAnkle = landmarks[.leftAnkle]?.confidence ?? 0 > 0.3
        let hasRightAnkle = landmarks[.rightAnkle]?.confidence ?? 0 > 0.3
        let feetVisible = hasLeftAnkle || hasRightAnkle
        
        // Check if head is visible
        let headVisible = (landmarks[.nose]?.confidence ?? 0) > 0.3
        
        // Check if knees are visible
        let kneesVisible = (landmarks[.leftKnee]?.confidence ?? 0 > 0.3) || (landmarks[.rightKnee]?.confidence ?? 0 > 0.3)
        
        if !feetVisible && kneesVisible {
            // Knees visible but feet are not - user is too close
            return (false, "Move back so your feet are visible")
        }
        
        if !headVisible && feetVisible {
            // Feet visible but head is not - user is too far
            return (false, "Move forward so your head is visible")
        }
        
        if !feetVisible && !headVisible {
            // Neither feet nor head visible
            return (false, "Please stand in front of camera")
        }
        
        // First check if basic pose is valid (matches Android isPersonFullyDetected)
        if !poseValidator.isValidPose(landmarks) {
            return (false, "Please stand in front of camera")
        }

        // NOTE: Body-centered check removed to avoid blocking users who cannot perfectly center.

        // Calculate posture metrics (matches Android calculatePostureMetrics)
        guard let metrics = poseValidator.calculatePostureMetrics(landmarks) else {
            return (false, "Cannot calculate posture metrics")
        }

        // Compare with reference (matches Android compareWithReference + isPoseAccurate)
        let accuracy = poseValidator.compareWithReference(metrics)

        // Check shoulder angles (arms should be at ~45° from body, matching Android)
        if !accuracy.shoulderAccurateLeft || !accuracy.shoulderAccurateRight {
            return (false, "Extend your arms away from your body")
        }

        // Check if elbows are straight (matching Android)
        if !accuracy.elbowAccurateLeft || !accuracy.elbowAccurateRight {
            return (false, "Keep your arms straight")
        }

        // Check spine alignment (matching Android)
        // if !accuracy.spineAccurate {
        //     return (false, "Stand up straight")
        // }

        // Check hip/leg angles (matching Android)
        if !accuracy.hipAccurateLeft || !accuracy.hipAccurateRight {
            return (false, "Keep your legs straight")
        }

        // Check feet separation (matching Android leg separation check)
        if !poseValidator.areFeetApart(landmarks) {
            return (false, "Spread your feet shoulder-width apart")
        }

        return (true, "Perfect! Hold still...")
    }
}
