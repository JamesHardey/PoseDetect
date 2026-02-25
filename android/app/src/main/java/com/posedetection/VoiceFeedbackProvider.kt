package com.posedetection

import android.speech.tts.TextToSpeech
import android.util.Log

class VoiceFeedbackProvider(private val textToSpeech: TextToSpeech?) {
    
    companion object {
        private const val TAG = "VoiceFeedbackProvider"
        private const val SPEECH_COOLDOWN = 3000L // 3 seconds
    }
    
    private var lastSpokenTime = 0L
    
    fun provideFrontPoseFeedback(
        customMessage: String?,
        accuracy: PostureAccuracy?,
        positionCheck: PositionCheck?
    ) {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastSpokenTime < SPEECH_COOLDOWN) return

        val feedback = customMessage ?: run {
            if (positionCheck != null && !positionCheck.inBox && positionCheck.issues.isNotEmpty()) {
                positionCheck.issues.first()
            } else if (accuracy != null) {
                when {
                    !accuracy.shoulderAccurateLeft && accuracy.shoulderDiffLeft > 20 ->
                        "Raise your right arm higher"
                    !accuracy.shoulderAccurateRight && accuracy.shoulderDiffRight > 20 ->
                        "Raise your left arm higher"
                    !accuracy.elbowAccurateLeft ->
                        "Straighten your right arm"
                    !accuracy.elbowAccurateRight ->
                        "Straighten your left arm"
                    !accuracy.spineAccurate ->
                        "Stand up straight"
                    !accuracy.shoulderLevelAccurate ->
                        "Level your shoulders"
                    !accuracy.legSeparationAccurate ->
                        "Spread your legs apart"
                    !accuracy.hipAccurateLeft || !accuracy.hipAccurateRight ->
                        "Keep your legs straight"
                    else -> null
                }
            } else {
                null
            }
        }

        feedback?.let {
            speak(it)
        }
    }

    fun provideSidePoseFeedback(
        accuracy: SidePoseAccuracy,
        positionCheck: PositionCheck?
    ) {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastSpokenTime < SPEECH_COOLDOWN) return

        val feedback = run {
            if (positionCheck != null && !positionCheck.inBox && positionCheck.issues.isNotEmpty()) {
                positionCheck.issues.first()
            } else {
                when {
                    !accuracy.isSideView ->
                        "Please turn to your side, stand sideways to the camera"
                    !accuracy.neckHeadAccurate ->
                        "Keep your head straight, align with your spine"
                    !accuracy.spineAccurate ->
                        "Stand up straight, keep your spine vertical"
                    !accuracy.leftArmAccurate && !accuracy.rightArmAccurate ->
                        "Relax your arms by your sides"
                    !accuracy.leftLegAccurate && !accuracy.rightLegAccurate ->
                        "Keep your legs straight"
                    else -> null
                }
            }
        }

        feedback?.let {
            speak(it)
        }
    }
    
    fun speak(message: String) {
        textToSpeech?.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
        lastSpokenTime = System.currentTimeMillis()
        Log.d(TAG, "Voice feedback: $message")
    }
    
    fun speakWithoutCooldown(message: String) {
        textToSpeech?.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
        Log.d(TAG, "Voice feedback (no cooldown): $message")
    }
}
