import Foundation
import Vision
import CoreGraphics

public class PoseValidator {
    // Posture accuracy matching Android
    public struct PostureAccuracy {
        public let shoulderAccurateLeft: Bool
        public let shoulderAccurateRight: Bool
        public let elbowAccurateLeft: Bool
        public let elbowAccurateRight: Bool
        public let spineAccurate: Bool
        public let hipAccurateLeft: Bool
        public let hipAccurateRight: Bool
        
        public init(shoulderAccurateLeft: Bool, shoulderAccurateRight: Bool, elbowAccurateLeft: Bool, elbowAccurateRight: Bool, spineAccurate: Bool, hipAccurateLeft: Bool, hipAccurateRight: Bool) {
            self.shoulderAccurateLeft = shoulderAccurateLeft
            self.shoulderAccurateRight = shoulderAccurateRight
            self.elbowAccurateLeft = elbowAccurateLeft
            self.elbowAccurateRight = elbowAccurateRight
            self.spineAccurate = spineAccurate
            self.hipAccurateLeft = hipAccurateLeft
            self.hipAccurateRight = hipAccurateRight
        }
    }
    
    // Posture metrics matching Android implementation
    public struct PostureMetrics {
        public let shoulderAngleLeft: Double
        public let shoulderAngleRight: Double
        public let elbowAngleLeft: Double
        public let elbowAngleRight: Double
        public let spineAngle: Double
        public let hipAngleLeft: Double
        public let hipAngleRight: Double
        public let shoulderLevelDiff: Double
        public let legSeparationAngle: Double
    }
    
    // Reference pose values matching Android exactly
    struct ReferencePose {
        let shoulderAngle: Double = 45.0          // Arms at 45° from body (Android value)
        let shoulderTolerance: Double = 30.0      // Android: shoulderAngleTolerance (allows 45-75 degree range)
        let elbowAngle: Double = 180.0            // Straight arms (Android: elbowAngleLeft/Right)
        let elbowTolerance: Double = 20.0         // Android: elbowAngleTolerance
        let spineAngle: Double = 0.0              // Vertical spine (Android value)
        let spineTolerance: Double = 10.0         // Android: spineAngleTolerance
        let hipAngle: Double = 180.0              // Straight legs (Android: hipAngleLeft/Right)
        let hipTolerance: Double = 15.0           // Android: hipAngleTolerance
        let shoulderLevelTolerance: Double = 30.0 // Android: shoulderLevelTolerance
        let legSeparationAngle: Double = 45.0     // Android value
        let legSeparationTolerance: Double = 15.0 // Android value
    }
    
    let referencePose = ReferencePose()
    
    public init() {}
    
    // Calculate angle between three points (matching Android's calculateAngle)
    private func calculateAngle(first: CGPoint, mid: CGPoint, last: CGPoint) -> Double {
        let radians = atan2(last.y - mid.y, last.x - mid.x) -
                     atan2(first.y - mid.y, first.x - mid.x)
        
        var angle = abs(radians * 180.0 / .pi)
        
        if angle > 180.0 {
            angle = 360.0 - angle
        }
        
        return angle
    }
    
    public func isValidPose(_ landmarks: PoseLandmarks, 
                     confidenceThreshold: Float = 0.3) -> Bool {
        let criticalLandmarks: [MLKitJointName] = [
            .nose, .leftShoulder, .rightShoulder,
            .leftElbow, .rightElbow, .leftWrist, .rightWrist,
            .leftHip, .rightHip, .leftKnee, .rightKnee,
            .leftAnkle, .rightAnkle  // Feet must be visible!
        ]
        
        for landmark in criticalLandmarks {
            guard let point = landmarks[landmark],
                  point.confidence > confidenceThreshold else {
                return false
            }
        }
        
        return true
    }
    
    // Calculate posture metrics (matching Android implementation)
    public func calculatePostureMetrics(_ landmarks: PoseLandmarks) -> PostureMetrics? {
        guard let leftShoulder = landmarks[.leftShoulder],
              let rightShoulder = landmarks[.rightShoulder],
              let leftElbow = landmarks[.leftElbow],
              let rightElbow = landmarks[.rightElbow],
              let leftWrist = landmarks[.leftWrist],
              let rightWrist = landmarks[.rightWrist],
              let leftHip = landmarks[.leftHip],
              let rightHip = landmarks[.rightHip],
              let leftKnee = landmarks[.leftKnee],
              let rightKnee = landmarks[.rightKnee],
              let neck = landmarks[.neck] else {
            return nil
        }
        
        // Calculate shoulder angles: angle between vertical body line and arm
        // For left arm: angle at shoulder between hip-shoulder line and shoulder-elbow line
        let shoulderAngleLeft = calculateAngle(
            first: leftHip.location,
            mid: leftShoulder.location,
            last: leftElbow.location
        )
        // For right arm: angle at shoulder between hip-shoulder line and shoulder-elbow line
        let shoulderAngleRight = calculateAngle(
            first: rightHip.location,
            mid: rightShoulder.location,
            last: rightElbow.location
        )
        let elbowAngleLeft = calculateAngle(
            first: leftShoulder.location,
            mid: leftElbow.location,
            last: leftWrist.location
        )
        let elbowAngleRight = calculateAngle(
            first: rightShoulder.location,
            mid: rightElbow.location,
            last: rightWrist.location
        )
        
        // Calculate spine angle (neck to hip center)
        let hipCenter = CGPoint(
            x: (leftHip.location.x + rightHip.location.x) / 2,
            y: (leftHip.location.y + rightHip.location.y) / 2
        )
        let spineAngle = abs(atan2(hipCenter.y - neck.location.y, hipCenter.x - neck.location.x) * 180.0 / .pi)
        
        // Hip angles
        let hipAngleLeft = calculateAngle(
            first: leftShoulder.location,
            mid: leftHip.location,
            last: leftKnee.location
        )
        let hipAngleRight = calculateAngle(
            first: rightShoulder.location,
            mid: rightHip.location,
            last: rightKnee.location
        )
        
        // Shoulder level difference
        let shoulderLevelDiff = abs(leftShoulder.location.y - rightShoulder.location.y)
        
        // Leg separation angle
        let legSeparationAngle = calculateAngle(
            first: leftKnee.location,
            mid: hipCenter,
            last: rightKnee.location
        )
        
        return PostureMetrics(
            shoulderAngleLeft: shoulderAngleLeft,
            shoulderAngleRight: shoulderAngleRight,
            elbowAngleLeft: elbowAngleLeft,
            elbowAngleRight: elbowAngleRight,
            spineAngle: spineAngle,
            hipAngleLeft: hipAngleLeft,
            hipAngleRight: hipAngleRight,
            shoulderLevelDiff: Double(shoulderLevelDiff),
            legSeparationAngle: legSeparationAngle
        )
    }
    
    // Compare with reference and return detailed accuracy (matching Android exactly)
    public func compareWithReference(_ metrics: PostureMetrics) -> PostureAccuracy {
        let shoulderAccurateLeft = abs(metrics.shoulderAngleLeft - referencePose.shoulderAngle) <= referencePose.shoulderTolerance
        let shoulderAccurateRight = abs(metrics.shoulderAngleRight - referencePose.shoulderAngle) <= referencePose.shoulderTolerance
        let elbowAccurateLeft = abs(metrics.elbowAngleLeft - referencePose.elbowAngle) <= referencePose.elbowTolerance
        let elbowAccurateRight = abs(metrics.elbowAngleRight - referencePose.elbowAngle) <= referencePose.elbowTolerance
        let spineAccurate = metrics.spineAngle <= referencePose.spineTolerance
        let hipAccurateLeft = abs(metrics.hipAngleLeft - referencePose.hipAngle) <= referencePose.hipTolerance
        let hipAccurateRight = abs(metrics.hipAngleRight - referencePose.hipAngle) <= referencePose.hipTolerance
        
        return PostureAccuracy(
            shoulderAccurateLeft: shoulderAccurateLeft,
            shoulderAccurateRight: shoulderAccurateRight,
            elbowAccurateLeft: elbowAccurateLeft,
            elbowAccurateRight: elbowAccurateRight,
            spineAccurate: spineAccurate,
            hipAccurateLeft: hipAccurateLeft,
            hipAccurateRight: hipAccurateRight
        )
    }
    
    // Compare with reference pose (matching Android's compareWithReference)
    func isPoseAccurate(_ metrics: PostureMetrics) -> Bool {
        let shouldersLevel = abs(metrics.shoulderAngleLeft - referencePose.shoulderAngle) <= referencePose.shoulderTolerance &&
                            abs(metrics.shoulderAngleRight - referencePose.shoulderAngle) <= referencePose.shoulderTolerance

        let armsRelaxed = abs(metrics.elbowAngleLeft - referencePose.elbowAngle) <= referencePose.elbowTolerance &&
                         abs(metrics.elbowAngleRight - referencePose.elbowAngle) <= referencePose.elbowTolerance

        let spineErect = metrics.spineAngle <= referencePose.spineTolerance

        let hipsLevel = abs(metrics.hipAngleLeft - referencePose.hipAngle) <= referencePose.hipTolerance &&
                       abs(metrics.hipAngleRight - referencePose.hipAngle) <= referencePose.hipTolerance

        return shouldersLevel && armsRelaxed && spineErect && hipsLevel
    }
    
    // Legacy helper methods for backward compatibility
    func areShouldersLevel(_ landmarks: PoseLandmarks, 
                          tolerance: Float = 0.15) -> Bool {
        guard let leftShoulder = landmarks[.leftShoulder],
              let rightShoulder = landmarks[.rightShoulder] else {
            return false
        }
        
        let yDifference = abs(leftShoulder.location.y - rightShoulder.location.y)
        return yDifference < CGFloat(tolerance)
    }
    
    func areArmsDown(_ landmarks: PoseLandmarks, 
                    tolerance: Float = 0.2) -> Bool {
        guard let leftShoulder = landmarks[.leftShoulder],
              let rightShoulder = landmarks[.rightShoulder],
              let leftWrist = landmarks[.leftWrist],
              let rightWrist = landmarks[.rightWrist] else {
            return false
        }
        
        let leftArmDown = leftWrist.location.y < leftShoulder.location.y - CGFloat(tolerance)
        let rightArmDown = rightWrist.location.y < rightShoulder.location.y - CGFloat(tolerance)
        
        return leftArmDown && rightArmDown
    }
    
    func areFeetApart(_ landmarks: PoseLandmarks, 
                     minDistance: Float = 0.15) -> Bool {
        guard let leftAnkle = landmarks[.leftAnkle],
              let rightAnkle = landmarks[.rightAnkle] else {
            return false
        }
        
        let distance = abs(leftAnkle.location.x - rightAnkle.location.x)
        return distance >= CGFloat(minDistance)
    }
    
    func isBodyCentered(_ landmarks: PoseLandmarks, 
                       centerTolerance: Float = 0.2) -> Bool {
        guard let leftShoulder = landmarks[.leftShoulder],
              let rightShoulder = landmarks[.rightShoulder] else {
            return false
        }
        
        let shoulderCenter = (leftShoulder.location.x + rightShoulder.location.x) / 2
        let screenCenter: CGFloat = 0.5
        
        let distanceFromCenter = abs(shoulderCenter - screenCenter)
        return distanceFromCenter < CGFloat(centerTolerance)
    }
}
