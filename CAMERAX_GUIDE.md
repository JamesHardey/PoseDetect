# CameraX Implementation Guide

This project includes a native CameraX implementation for Android.

## Features

- ✅ Native CameraX integration
- ✅ Front and back camera support
- ✅ Permission handling
- ✅ React Native bridge
- ✅ TypeScript support

## What Was Implemented

### Native Android Components

1. **CameraXModule.kt** - Native module providing:
   - `checkCameraPermission()` - Check if camera permission is granted
   - `requestCameraPermission()` - Request camera permission from user
   - `isCameraAvailable()` - Check if device has a camera

2. **CameraViewManager.kt** - Native view manager providing:
   - Camera preview display
   - Front/back camera switching
   - Lifecycle management

3. **CameraXPackage.kt** - Package registration for React Native

### React Native Components

1. **src/modules/CameraXModule.ts** - TypeScript bridge to native module
2. **src/components/CameraView.tsx** - React component wrapper with:
   - Automatic permission handling
   - Camera availability checking
   - Error handling
   - User-friendly UI

### Dependencies Added

```gradle
androidx.camera:camera-core:1.3.1
androidx.camera:camera-camera2:1.3.1
androidx.camera:camera-lifecycle:1.3.1
androidx.camera:camera-view:1.3.1
androidx.camera:camera-extensions:1.3.1
```

### Permissions Added

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" android:required="false" />
<uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />
```

## Usage

The `App.tsx` file demonstrates basic usage:

```tsx
import { CameraView } from './src/components/CameraView';

function App() {
  const [cameraType, setCameraType] = useState<'front' | 'back'>('back');

  return (
    <CameraView style={styles.camera} cameraType={cameraType} />
  );
}
```

## Running the App

1. Install dependencies:
```bash
npm install
```

2. Build and run on Android:
```bash
npm run android
```

## Next Steps for Pose Detection

To add pose detection capabilities, you can:

1. Add ML Kit Pose Detection dependency:
```gradle
implementation 'com.google.mlkit:pose-detection:18.0.0-beta3'
```

2. Integrate pose detection in `CameraViewManager.kt`:
   - Add ImageAnalysis use case
   - Process frames with PoseDetector
   - Send results back to React Native

3. Create overlay component to draw skeleton on detected poses

## API Reference

### CameraView Component

Props:
- `style?: ViewStyle` - Style for the camera view
- `cameraType?: 'front' | 'back'` - Which camera to use (default: 'back')

### CameraXModule

Methods:
- `checkCameraPermission(): Promise<boolean>` - Returns true if permission granted
- `requestCameraPermission(): Promise<boolean>` - Requests permission, returns true if granted
- `isCameraAvailable(): Promise<boolean>` - Returns true if device has camera

## Troubleshooting

If you encounter build errors:
1. Clean the build: `cd android && ./gradlew clean`
2. Rebuild: `cd .. && npm run android`

If camera doesn't start:
1. Check that camera permission is granted in device settings
2. Verify device has a camera
3. Check logcat for errors: `adb logcat | grep -i camera`
