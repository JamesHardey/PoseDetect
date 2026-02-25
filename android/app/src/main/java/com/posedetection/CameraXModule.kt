package com.posedetection

import android.Manifest
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.PermissionAwareActivity
import com.facebook.react.modules.core.PermissionListener

class CameraXModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 100
    }

    override fun getName(): String {
        return "CameraXModule"
    }

    @ReactMethod
    fun checkCameraPermission(promise: Promise) {
        val context = reactApplicationContext
        val permission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        )
        promise.resolve(permission == PackageManager.PERMISSION_GRANTED)
    }

    @ReactMethod
    fun requestCameraPermission(promise: Promise) {
        val activity = reactApplicationContext.currentActivity
        
        if (activity == null) {
            promise.reject("E_ACTIVITY_DOES_NOT_EXIST", "Activity doesn't exist")
            return
        }

        val permissionAwareActivity = activity as? PermissionAwareActivity
        if (permissionAwareActivity == null) {
            promise.reject("E_ACTIVITY_NOT_PERMISSION_AWARE", "Activity is not PermissionAwareActivity")
            return
        }

        val permissionListener = object : PermissionListener {
            override fun onRequestPermissionsResult(
                requestCode: Int,
                permissions: Array<String>,
                grantResults: IntArray
            ): Boolean {
                if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
                    if (grantResults.isNotEmpty() && 
                        grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                        promise.resolve(true)
                    } else {
                        promise.resolve(false)
                    }
                    return true
                }
                return false
            }
        }

        permissionAwareActivity.requestPermissions(
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST_CODE,
            permissionListener
        )
    }

    @ReactMethod
    fun isCameraAvailable(promise: Promise) {
        val context = reactApplicationContext
        val hasCamera = context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)
        promise.resolve(hasCamera)
    }
}
