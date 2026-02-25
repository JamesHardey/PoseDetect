import UIKit
import Vision
import CoreGraphics

// Shared pose types for ML Kit mapping
public struct RecognizedPointCompat {
    public let location: CGPoint
    public let confidence: Float
}

// Use custom ML Kit joint names (supports all 33 landmarks)
public enum MLKitJointName: Hashable {
    case nose, leftEye, rightEye, leftEar, rightEar
    case leftEyeInner, leftEyeOuter, rightEyeInner, rightEyeOuter
    case leftShoulder, rightShoulder, leftElbow, rightElbow
    case leftWrist, rightWrist, leftPinky, rightPinky
    case leftIndex, rightIndex, leftThumb, rightThumb
    case leftHip, rightHip, leftKnee, rightKnee
    case leftAnkle, rightAnkle, leftHeel, rightHeel
    case leftFootIndex, rightFootIndex
    case neck, mouthLeft, mouthRight
}

public typealias JointName = MLKitJointName
public typealias PoseLandmarks = [JointName: RecognizedPointCompat]


class PoseOverlayView: UIView {
    // Landmarks need module access for countdown/validation checks from CameraViewManager
    var landmarks: PoseLandmarks?
    // Public getter for safe access from other classes
    public func getCurrentLandmarks() -> PoseLandmarks? {
        return self.landmarks
    }
    private var imageSize: CGSize = .zero
    private var accuracy: PoseValidator.PostureAccuracy?
    private var isPerfectPose: Bool = false
    private var countdownValue: Int = 0
    private var isCountingDown: Bool = false
    private var isMirrored: Bool = false
    private var guidance: [JointName: String]? = nil
    private var poseStage: String? = nil

    // Target box similar to Android (5% inset with dashed yellow stroke)
    private let targetBoxInsets: CGFloat = 0.05
    private let targetStrokeColor = UIColor.yellow.cgColor
    private let targetFillColor = UIColor(red: 1, green: 1, blue: 0, alpha: 0.15).cgColor
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque = false
        isUserInteractionEnabled = false
        contentMode = .redraw
        clearsContextBeforeDrawing = true
        // Force the layer to be transparent and non-blocking
        layer.isOpaque = false
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func updatePose(_ newLandmarks: PoseLandmarks?, 
                    imageSize: CGSize,
                    accuracy: PoseValidator.PostureAccuracy? = nil,
                    perfect: Bool = false,
                    countdown: Int = 0,
                    counting: Bool = false,
                    mirrored: Bool = false,
                    guidance: [JointName: String]? = nil,
                    stage: String? = nil) {
        self.landmarks = newLandmarks
        self.imageSize = imageSize
        self.accuracy = accuracy
        self.isPerfectPose = perfect
        self.countdownValue = countdown
        self.isCountingDown = counting
        self.isMirrored = mirrored
        self.guidance = guidance
        self.poseStage = stage
        
        print("🖼️ PoseOverlayView.updatePose called - landmarks: \(newLandmarks?.count ?? 0), imageSize: \(imageSize), bounds: \(bounds), guidance: \(guidance?.count ?? 0)")
        
        setNeedsDisplay()
    }
    
    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else {
            print("❌ No graphics context!")
            return
        }
        
        print("🎨 PoseOverlayView.draw called - rect: \(rect), bounds: \(bounds)")
        
        guard let landmarks = landmarks, imageSize.width > 0, imageSize.height > 0 else {
            print("⚠️ No landmarks or invalid imageSize - landmarks: \(landmarks?.count ?? 0), imageSize: \(imageSize)")
            // Still draw target box so user knows framing
            drawTargetBox()
            return
        }
        
        print("✅ Drawing \(landmarks.count) landmarks")

        // ML Kit provides pixel coordinates; scale to view bounds
        let scaleX = bounds.width / imageSize.width
        let scaleY = bounds.height / imageSize.height

        print("📐 Scale factors - scaleX: \(scaleX), scaleY: \(scaleY)")

        // Draw target box first
        drawTargetBox()
        
        // Draw landmarks and connections directly - we'll scale points manually
        // (No CGAffineTransform needed as we scale each point individually)
        drawConnections(context: context, landmarks: landmarks, scaleX: scaleX, scaleY: scaleY)
        drawLandmarks(context: context, landmarks: landmarks, scaleX: scaleX, scaleY: scaleY)
        
        // Draw countdown if active
        if isCountingDown && countdownValue > 0 {
            drawCountdown(context: context)
        } else if isPerfectPose {
            drawStatus(context: context, text: "PERFECT POSE!")
        }

        // Draw guidance overlay last so it's visible
        drawGuidanceOverlay()
    }

    private func drawTargetBox() {
        guard let context = UIGraphicsGetCurrentContext() else { return }
        let insetX = bounds.width * targetBoxInsets
        let insetY = bounds.height * targetBoxInsets
        let rect = bounds.insetBy(dx: insetX, dy: insetY)

        context.saveGState()
        context.setFillColor(targetFillColor)
        context.fill(rect)

        context.setStrokeColor(targetStrokeColor)
        context.setLineWidth(3.0)
        context.setLineDash(phase: 0, lengths: [12, 6])
        context.stroke(rect)
        context.restoreGState()
    }
    
    private func drawConnections(context: CGContext, landmarks: PoseLandmarks, scaleX: CGFloat, scaleY: CGFloat) {
        let connections: [(MLKitJointName, MLKitJointName)] = [
            // Face connections
            (.leftEyeInner, .leftEye),
            (.leftEye, .leftEyeOuter),
            (.rightEyeInner, .rightEye),
            (.rightEye, .rightEyeOuter),
            (.leftEye, .rightEye),
            (.leftEye, .nose),
            (.rightEye, .nose),
            (.leftEye, .leftEar),
            (.rightEye, .rightEar),
            // Body connections
            (.leftShoulder, .rightShoulder),
            (.leftShoulder, .leftElbow),
            (.leftElbow, .leftWrist),
            (.rightShoulder, .rightElbow),
            (.rightElbow, .rightWrist),
            (.leftShoulder, .leftHip),
            (.rightShoulder, .rightHip),
            (.leftHip, .rightHip),
            (.leftHip, .leftKnee),
            (.leftKnee, .leftAnkle),
            (.rightHip, .rightKnee),
            (.rightKnee, .rightAnkle),
            // Foot connections
            (.leftAnkle, .leftHeel),
            (.leftHeel, .leftFootIndex),
            (.rightAnkle, .rightHeel),
            (.rightHeel, .rightFootIndex),
            // Hand connections
            (.leftWrist, .leftThumb),
            (.leftWrist, .leftIndex),
            (.leftWrist, .leftPinky),
            (.rightWrist, .rightThumb),
            (.rightWrist, .rightIndex),
            (.rightWrist, .rightPinky),
            // Mouth connection
            (.mouthLeft, .mouthRight)
        ]
        
        for (start, end) in connections {
            // Default to green unless accuracy indicates otherwise
            var lineColor = UIColor.green.cgColor
            var lineWidth: CGFloat = 4.0
            
            // Make ankle/foot connections more prominent
            let footConnections: [(MLKitJointName, MLKitJointName)] = [
                (.leftKnee, .leftAnkle), (.rightKnee, .rightAnkle),
                (.leftAnkle, .leftHeel), (.rightAnkle, .rightHeel),
                (.leftHeel, .leftFootIndex), (.rightHeel, .rightFootIndex)
            ]
            if footConnections.contains(where: { $0 == start && $1 == end }) {
                lineWidth = 6.0
                lineColor = UIColor.cyan.cgColor
            }
            
            context.setLineWidth(lineWidth)
            if let acc = accuracy {
                let isAccurate: Bool
                switch (start, end) {
                // Face landmarks are always considered accurate
                case (.leftEye, .rightEye), (.leftEye, .nose), (.rightEye, .nose),
                     (.leftEye, .leftEar), (.rightEye, .rightEar):
                    isAccurate = true
                // Arm connections - use elbow accuracy
                case (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
                     (.leftWrist, .leftThumb), (.leftWrist, .leftIndex), (.leftWrist, .leftPinky):
                    isAccurate = acc.elbowAccurateLeft
                case (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
                     (.rightWrist, .rightThumb), (.rightWrist, .rightIndex), (.rightWrist, .rightPinky):
                    isAccurate = acc.elbowAccurateRight
                // Body/leg and shoulder/hip lines — keep green (not validated here)
                case (.leftShoulder, .leftHip), (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
                     (.rightShoulder, .rightHip), (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
                     (.leftShoulder, .rightShoulder), (.leftHip, .rightHip), (.leftHip, .rightHip):
                    isAccurate = true
                default:
                    isAccurate = true
                }
                lineColor = isAccurate ? UIColor.green.cgColor : UIColor.red.cgColor
            }

            // If guidance indicates this connection's joints, mark red
            if let guidance = guidance, guidance[start] != nil || guidance[end] != nil {
                lineColor = UIColor.red.cgColor
            }
            context.setStrokeColor(lineColor)
            
            guard let startPoint = landmarks[start],
                  let endPoint = landmarks[end],
                  startPoint.confidence > 0.3,
                  endPoint.confidence > 0.3 else { continue }
            
            // Scale from image coordinates to view coordinates
            let scaledStart = CGPoint(x: startPoint.location.x * scaleX, y: startPoint.location.y * scaleY)
            let scaledEnd = CGPoint(x: endPoint.location.x * scaleX, y: endPoint.location.y * scaleY)
            
            context.move(to: scaledStart)
            context.addLine(to: scaledEnd)
            context.strokePath()
        }
    }
    
    private func drawLandmarks(context: CGContext, landmarks: PoseLandmarks, scaleX: CGFloat, scaleY: CGFloat) {
        for (joint, point) in landmarks {
            // Default to green unless accuracy indicates otherwise
            var dotColor = UIColor.green.cgColor
            var radius: CGFloat = 8.0
            
            if let acc = accuracy {
                let isAccurate: Bool
                switch joint {
                // Face landmarks always green
                case .nose, .leftEye, .rightEye, .leftEar, .rightEar, .mouthLeft, .mouthRight:
                    isAccurate = true
                // Left arm landmarks - use left elbow accuracy as proxy
                case .leftShoulder, .leftElbow, .leftWrist, .leftThumb, .leftIndex, .leftPinky:
                    isAccurate = acc.elbowAccurateLeft
                // Right arm landmarks - use right elbow accuracy as proxy
                case .rightShoulder, .rightElbow, .rightWrist, .rightThumb, .rightIndex, .rightPinky:
                    isAccurate = acc.elbowAccurateRight
                // Legs / hips - keep green (not validated in this view)
                case .leftHip, .leftKnee, .rightHip, .rightKnee:
                    isAccurate = true
                // Ankle/foot landmarks - make them more prominent with larger radius and cyan color
                case .leftAnkle, .rightAnkle, .leftHeel, .leftFootIndex, .rightHeel, .rightFootIndex:
                    isAccurate = true
                    radius = 12.0  // Larger radius for feet
                    dotColor = UIColor.cyan.cgColor  // Distinct color for feet
                default:
                    isAccurate = true
                }
                
                // Only apply red/green logic if not a foot landmark
                if joint != .leftAnkle && joint != .rightAnkle && 
                   joint != .leftHeel && joint != .rightHeel && 
                   joint != .leftFootIndex && joint != .rightFootIndex {
                    dotColor = isAccurate ? UIColor.green.cgColor : UIColor.red.cgColor
                }
            }

            // Override color to red for any joint with guidance
            if let guidance = guidance, guidance[joint] != nil {
                dotColor = UIColor.red.cgColor
            }
            context.setFillColor(dotColor)
            
            guard point.confidence > 0.3 else { continue }
            
            // Scale from image coordinates to view coordinates
            let x = point.location.x * scaleX
            let y = point.location.y * scaleY
            
            let rect = CGRect(x: x - radius, y: y - radius, width: radius * 2, height: radius * 2)
            context.fillEllipse(in: rect)
        }
    }

    private func drawCountdown(context: CGContext) {
        let text = "\(countdownValue)" as NSString
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 120),
            .foregroundColor: UIColor.white
        ]
        
        let textSize = text.size(withAttributes: attributes)
        let x = (bounds.width - textSize.width) / 2
        let y = (bounds.height - textSize.height) / 2
        
        text.draw(at: CGPoint(x: x, y: y), withAttributes: attributes)
    }
    
    private func drawStatus(context: CGContext, text: String) {
        let statusText = text as NSString
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 50),
            .foregroundColor: UIColor.green
        ]
        
        let textSize = statusText.size(withAttributes: attributes)
        let x = (bounds.width - textSize.width) / 2
        let y: CGFloat = 100
        
        statusText.draw(at: CGPoint(x: x, y: y), withAttributes: attributes)
    }

    // Draw guidance messages on top-left to help user
    private func drawGuidanceOverlay() {
        guard let guidance = guidance, guidance.count > 0 else { return }
        var messages: [String] = []
        for (joint, msg) in guidance {
            messages.append("\(joint): \(msg)")
        }
        let text = messages.joined(separator: "  •  ") as NSString
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 16),
            .foregroundColor: UIColor.white,
            .backgroundColor: UIColor(red: 0, green: 0, blue: 0, alpha: 0.5)
        ]
        let textSize = text.size(withAttributes: attributes)
        let padding: CGFloat = 8
        let rect = CGRect(x: 10, y: 10, width: min(bounds.width - 20, textSize.width + padding * 2), height: textSize.height + padding)

        // Draw background rounded rect
        if let context = UIGraphicsGetCurrentContext() {
            context.saveGState()
            let path = UIBezierPath(roundedRect: rect, cornerRadius: 8)
            context.setFillColor(UIColor(white: 0, alpha: 0.5).cgColor)
            context.addPath(path.cgPath)
            context.fillPath()
            context.restoreGState()
        }

        let textPoint = CGPoint(x: rect.minX + padding, y: rect.minY + (padding / 2))
        text.draw(at: textPoint, withAttributes: attributes)
    }
}
