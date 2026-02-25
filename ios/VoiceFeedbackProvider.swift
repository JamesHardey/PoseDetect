import AVFoundation
import UIKit

class VoiceFeedbackProvider: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var lastSpokenMessage: String = ""
    private var lastSpeakTime: Date = Date.distantPast
    private let minimumSpeakInterval: TimeInterval = 2.0
    private var audioSessionObserver: NSObjectProtocol?
    
    override init() {
        super.init()
        synthesizer.delegate = self
        setupAudioSession()
        setupNotifications()
    }
    
    private func setupNotifications() {
        // Observe audio session interruptions
        audioSessionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance(),
            queue: .main
        ) { [weak self] notification in
            guard let self = self else { return }
            guard let userInfo = notification.userInfo,
                  let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
                return
            }
            
            if type == .ended {
                // Re-setup audio session after interruption
                print("🔄 Audio session interruption ended, reactivating...")
                self.setupAudioSession()
            }
        }
    }
    
    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            // Use .ambient category to allow mixing with other audio and override silent mode
            try audioSession.setCategory(.playback, mode: .spokenAudio, options: [.mixWithOthers])
            try audioSession.setActive(true, options: [])
            print("✅ Audio session configured for voice feedback")
        } catch {
            print("❌ Failed to setup audio session: \(error.localizedDescription)")
        }
    }
    
    func provideFeedback(_ message: String) {
        // Avoid repeating the same message too frequently
        let currentTime = Date()
        if message == lastSpokenMessage && 
           currentTime.timeIntervalSince(lastSpeakTime) < minimumSpeakInterval {
            return
        }
        
        // Check if already on main thread to avoid nested dispatch
        if Thread.isMainThread {
            self.performSpeech(message, currentTime: currentTime)
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.performSpeech(message, currentTime: currentTime)
            }
        }
    }
    
    private func performSpeech(_ message: String, currentTime: Date) {
        // Ensure audio session is active before speaking (avoid reactivating if already active)
        do {
            let audioSession = AVAudioSession.sharedInstance()
            if !audioSession.isOtherAudioPlaying && !audioSession.isOtherAudioPlaying {
                if !audioSession.isInputAvailable || audioSession.outputVolume >= 0 {
                    try audioSession.setActive(true, options: [])
                }
            }
        } catch {
            print("❌ Failed to activate audio session: \(error.localizedDescription)")
            setupAudioSession()
        }
        
        // If already speaking, skip rather than interrupt to reduce lag
        if synthesizer.isSpeaking {
            return
        }
        
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.9
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        utterance.preUtteranceDelay = 0.1
        
        print("🔊 Speaking: \"\(message)\"")
        synthesizer.speak(utterance)
        
        self.lastSpokenMessage = message
        self.lastSpeakTime = currentTime
    }
    
    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
    
    // MARK: - AVSpeechSynthesizerDelegate
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        print("🗣️ Speech started: \"\(utterance.speechString)\"")
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        print("✅ Speech finished: \"\(utterance.speechString)\"")
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        print("⏹️ Speech cancelled: \"\(utterance.speechString)\"")
    }
    
    deinit {
        if let observer = audioSessionObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        synthesizer.stopSpeaking(at: .immediate)
    }
}
