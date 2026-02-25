package com.posedetection

import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.io.File
import java.io.FileOutputStream

class ImageCaptureModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String {
        return "ImageCaptureModule"
    }

    @ReactMethod
    fun saveImage(imageUri: String, promise: Promise) {
        try {
            // For now, just resolve with the URI
            // The actual saving will be handled in CameraViewManager
            promise.resolve(imageUri)
        } catch (e: Exception) {
            promise.reject("SAVE_ERROR", "Failed to save image", e)
        }
    }

    fun sendCaptureEvent(imageUri: String) {
        try {
            val params = com.facebook.react.bridge.Arguments.createMap()
            params.putString("uri", imageUri)
            android.util.Log.d("ImageCaptureModule", "Sending capture event with URI: $imageUri")
            reactApplicationContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("onImageCaptured", params)
            android.util.Log.d("ImageCaptureModule", "Event emitted successfully")
        } catch (e: Exception) {
            android.util.Log.e("ImageCaptureModule", "Error sending capture event", e)
        }
    }
    
    fun sendBothImagesEvent(frontUri: String, sideUri: String) {
        try {
            val params = com.facebook.react.bridge.Arguments.createMap()
            params.putString("frontUri", frontUri)
            params.putString("sideUri", sideUri)
            android.util.Log.d("ImageCaptureModule", "Sending both images event - Front: $frontUri, Side: $sideUri")
            reactApplicationContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("onBothImagesCaptured", params)
            android.util.Log.d("ImageCaptureModule", "Both images event emitted successfully")
        } catch (e: Exception) {
            android.util.Log.e("ImageCaptureModule", "Error sending both images event", e)
        }
    }
    
    fun sendStatusEvent(status: String, message: String = "") {
        try {
            val params = com.facebook.react.bridge.Arguments.createMap()
            params.putString("status", status)
            params.putString("message", message)
            android.util.Log.d("ImageCaptureModule", "Sending status event - Status: $status, Message: $message")
            reactApplicationContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("onCaptureStatus", params)
            android.util.Log.d("ImageCaptureModule", "Status event emitted successfully")
        } catch (e: Exception) {
            android.util.Log.e("ImageCaptureModule", "Error sending status event", e)
        }
    }
}
