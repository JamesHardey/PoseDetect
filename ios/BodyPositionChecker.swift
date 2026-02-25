import Foundation
import Vision
import CoreGraphics

public class BodyPositionChecker {
    private let poseValidator = PoseValidator()

    public init() {}

    // ── Full result including per-arm guidance for the overlay ──────────────
    public struct CheckResult {
        public let isValid: Bool
        public let feedback: String
        /// Joints to highlight red on the overlay  (nil = nothing to highlight)
        public let guidanceJoints: [JointName: String]?
    }

    // ── Main entry point ────────────────────────────────────────────────────
    public func checkPoseDetailed(_ landmarks: PoseLandmarks) -> CheckResult {

        // ── PRIORITY 1: Full body must be in frame ──────────────────────────
        // Head (nose) AND both ankles must be visible with adequate confidence.
        let headConf   = landmarks[.nose]?.confidence       ?? 0
        let lAnkleConf = landmarks[.leftAnkle]?.confidence  ?? 0
        let rAnkleConf = landmarks[.rightAnkle]?.confidence ?? 0
        let headVisible      = headConf   > 0.4
        // BOTH ankles must be visible — one foot hidden means the frame is too tight
        let bothFeetVisible  = lAnkleConf > 0.4 && rAnkleConf > 0.4
        let oneFeetVisible   = lAnkleConf > 0.4 || rAnkleConf > 0.4
        let kneesVisible     = (landmarks[.leftKnee]?.confidence  ?? 0) > 0.3
                            || (landmarks[.rightKnee]?.confidence ?? 0) > 0.3

        // Hard gate: head AND both feet must be in frame before anything else is evaluated
        if !headVisible && !oneFeetVisible {
            return CheckResult(isValid: false,
                               feedback: "Step into the frame — show your full body",
                               guidanceJoints: nil)
        }
        if !headVisible {
            return CheckResult(isValid: false,
                               feedback: "Move back — your head must be visible",
                               guidanceJoints: nil)
        }
        if !bothFeetVisible && kneesVisible {
            return CheckResult(isValid: false,
                               feedback: "Move back — both feet must be fully visible",
                               guidanceJoints: nil)
        }
        if !bothFeetVisible {
            return CheckResult(isValid: false,
                               feedback: "Step back so your head and both feet are in frame",
                               guidanceJoints: nil)
        }

        // ── PRIORITY 2: Critical skeleton detected ──────────────────────────
        if !poseValidator.isValidPose(landmarks) {
            return CheckResult(isValid: false,
                               feedback: "Stand in front of the camera so your full body is visible",
                               guidanceJoints: nil)
        }

        // ── PRIORITY 3: Legs straight + feet apart ──────────────────────────
        guard let metrics = poseValidator.calculatePostureMetrics(landmarks) else {
            return CheckResult(isValid: false,
                               feedback: "Cannot read your pose — ensure good lighting",
                               guidanceJoints: nil)
        }
        let accuracy = poseValidator.compareWithReference(metrics)

        if !accuracy.hipAccurateLeft || !accuracy.hipAccurateRight {
            return CheckResult(isValid: false,
                               feedback: "Stand straight — keep both legs straight",
                               guidanceJoints: [.leftHip: "straighten", .rightHip: "straighten",
                                               .leftKnee: "straighten", .rightKnee: "straighten"])
        }
        if !poseValidator.areFeetApart(landmarks) {
            return CheckResult(isValid: false,
                               feedback: "Spread your feet roughly shoulder-width apart",
                               guidanceJoints: [.leftAnkle: "spread", .rightAnkle: "spread"])
        }

        // ── PRIORITY 4: Arm position (most detail) ──────────────────────────
        guard let armResult = poseValidator.checkBothArms(landmarks) else {
            return CheckResult(isValid: false,
                               feedback: "Show your arms clearly",
                               guidanceJoints: nil)
        }

        var armGuidance: [JointName: String] = [:]

        if !armResult.left.isAccurate {
            // armResult.left == ML Kit's leftShoulder == person's RIGHT arm (front-camera mirror)
            let msg = armResult.left.message ?? "Adjust your right arm"
            armGuidance[.leftShoulder] = msg
            armGuidance[.leftElbow]    = msg
            armGuidance[.leftWrist]    = msg
        }
        if !armResult.right.isAccurate {
            // armResult.right == ML Kit's rightShoulder == person's LEFT arm (front-camera mirror)
            let msg = armResult.right.message ?? "Adjust your left arm"
            armGuidance[.rightShoulder] = msg
            armGuidance[.rightElbow]    = msg
            armGuidance[.rightWrist]    = msg
        }

        if !armResult.left.isAccurate || !armResult.right.isAccurate {
            // Provide the most specific feedback message available
            let feedback: String
            if !armResult.left.isAccurate && !armResult.right.isAccurate {
                // Both arms wrong — use left arm message but prefix with "Both arms: "
                if armResult.left.message == armResult.right.message,
                   let msg = armResult.left.message {
                    feedback = "Both arms: \(msg.lowercased())"
                } else {
                    feedback = armResult.left.message ?? armResult.right.message ?? "Adjust both arms"
                }
            } else if !armResult.left.isAccurate {
                // ML Kit left == person's right arm (mirrored front camera)
                feedback = armResult.left.message ?? "Adjust your right arm"
            } else {
                // ML Kit right == person's left arm (mirrored front camera)
                feedback = armResult.right.message ?? "Adjust your left arm"
            }
            return CheckResult(isValid: false,
                               feedback: feedback,
                               guidanceJoints: armGuidance.isEmpty ? nil : armGuidance)
        }

        // ── All checks passed ────────────────────────────────────────────────
        return CheckResult(isValid: true,
                           feedback: "Perfect! Hold still...",
                           guidanceJoints: nil)
    }

    // ── Legacy simple wrapper kept for backward-compat ───────────────────────
    public func checkPose(_ landmarks: PoseLandmarks) -> (isValid: Bool, feedback: String) {
        let r = checkPoseDetailed(landmarks)
        return (r.isValid, r.feedback)
    }
}
