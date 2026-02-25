import CoreGraphics
import Vision

// Common point type used across overlay and validators when not using Vision's VNRecognizedPoint directly
public struct RecognizedPointCompat {
    public let location: CGPoint
    public let confidence: Float
}

// Custom enum for ML Kit landmarks (all 33 landmarks matching Android)
public enum MLKitJointName: Hashable {
    // Face landmarks (5)
    case nose
    case leftEye
    case rightEye
    case leftEar
    case rightEar
    
    // Upper body (10)
    case leftShoulder
    case rightShoulder
    case leftElbow
    case rightElbow
    case leftWrist
    case rightWrist
    case leftPinky
    case rightPinky
    case leftIndex
    case rightIndex
    case leftThumb
    case rightThumb
    
    // Lower body (12)
    case leftHip
    case rightHip
    case leftKnee
    case rightKnee
    case leftAnkle
    case rightAnkle
    case leftHeel
    case rightHeel
    case leftFootIndex
    case rightFootIndex
    
    // Computed (2)
    case neck
    case mouthLeft
    case mouthRight
    
    // Convert to Vision JointName for backward compatibility
    var visionJointName: VNHumanBodyPoseObservation.JointName? {
        switch self {
        case .nose: return .nose
        case .leftEye: return .leftEye
        case .rightEye: return .rightEye
        case .leftEar: return .leftEar
        case .rightEar: return .rightEar
        case .leftShoulder: return .leftShoulder
        case .rightShoulder: return .rightShoulder
        case .leftElbow: return .leftElbow
        case .rightElbow: return .rightElbow
        case .leftWrist: return .leftWrist
        case .rightWrist: return .rightWrist
        case .leftHip: return .leftHip
        case .rightHip: return .rightHip
        case .leftKnee: return .leftKnee
        case .rightKnee: return .rightKnee
        case .leftAnkle: return .leftAnkle
        case .rightAnkle: return .rightAnkle
        case .neck: return .neck
        // ML Kit specific landmarks (no Vision equivalent)
        default: return nil
        }
    }
}

public typealias JointName = MLKitJointName
public typealias PoseLandmarks = [JointName: RecognizedPointCompat]
