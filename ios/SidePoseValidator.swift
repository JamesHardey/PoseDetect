import Foundation
import Vision
import CoreGraphics

public class SidePoseValidator {
    
    public init() {}
    
    public func isSidewaysPose(_ landmarks: PoseLandmarks, imageSize: CGSize, tolerance: CGFloat = 0.08) -> Bool {
        // Check if shoulders are aligned vertically (indicating sideways position)
        guard let leftShoulder = landmarks[.leftShoulder],
              let rightShoulder = landmarks[.rightShoulder] else {
            return false
        }
        
        // Normalize X coordinates to [0..1]
        let leftX = leftShoulder.location.x / imageSize.width
        let rightX = rightShoulder.location.x / imageSize.width
        let shoulderXDiff = abs(leftX - rightX)
        return shoulderXDiff < tolerance
    }
    
    public func areArmsSideways(_ landmarks: PoseLandmarks, imageSize: CGSize, tolerance: CGFloat = 0.12) -> Bool {
        guard let leftShoulder = landmarks[.leftShoulder],
              let rightShoulder = landmarks[.rightShoulder],
              let leftWrist = landmarks[.leftWrist],
              let rightWrist = landmarks[.rightWrist] else {
            return false
        }
        
        // Normalize X coordinates
        let leftShoulderX = leftShoulder.location.x / imageSize.width
        let rightShoulderX = rightShoulder.location.x / imageSize.width
        let leftWristX = leftWrist.location.x / imageSize.width
        let rightWristX = rightWrist.location.x / imageSize.width
        
        // Check if wrists are close to shoulders in X (line up vertically when sideways)
        let leftArmXDiff = abs(leftWristX - leftShoulderX)
        let rightArmXDiff = abs(rightWristX - rightShoulderX)
        
        return leftArmXDiff < tolerance && rightArmXDiff < tolerance
    }
    
    public func areLegsSideways(_ landmarks: PoseLandmarks, imageSize: CGSize, tolerance: CGFloat = 0.12) -> Bool {
        guard let leftHip = landmarks[.leftHip],
              let rightHip = landmarks[.rightHip],
              let leftAnkle = landmarks[.leftAnkle],
              let rightAnkle = landmarks[.rightAnkle] else {
            return false
        }
        
        let leftHipX = leftHip.location.x / imageSize.width
        let rightHipX = rightHip.location.x / imageSize.width
        let leftAnkleX = leftAnkle.location.x / imageSize.width
        let rightAnkleX = rightAnkle.location.x / imageSize.width
        
        let leftLegXDiff = abs(leftAnkleX - leftHipX)
        let rightLegXDiff = abs(rightAnkleX - rightHipX)
        
        return leftLegXDiff < tolerance && rightLegXDiff < tolerance
    }
    
    public func isValidSidePose(_ landmarks: PoseLandmarks, imageSize: CGSize) -> Bool {
        return isSidewaysPose(landmarks, imageSize: imageSize) &&
               areArmsSideways(landmarks, imageSize: imageSize) &&
               areLegsSideways(landmarks, imageSize: imageSize)
    }
}
