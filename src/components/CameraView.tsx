import React, { useEffect, useState, useRef } from 'react';
import {
  requireNativeComponent,
  ViewStyle,
  StyleSheet,
  View,
  Text,
  Button,
  TouchableOpacity,
  Platform,
  PermissionsAndroid,
  UIManager,
  findNodeHandle,
} from 'react-native';
import CameraXModule from '../modules/CameraXModule';

interface CameraViewProps {
  style?: ViewStyle;
  cameraType?: 'front' | 'back';
  onBothCaptured?: (params: { imageUri: string; sideImageUri: string }) => void;
}

const NativeCameraView = requireNativeComponent<CameraViewProps>('CameraView');

export const CameraView: React.FC<CameraViewProps> = ({
  style,
  cameraType = 'front',
  onBothCaptured,
}) => {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const cameraViewRef = useRef(null);

  useEffect(() => {
    checkPermission();
  }, []);

  useEffect(() => {
    // Re-check permission when component mounts
    if (hasPermission === null) {
      checkPermission();
    }
  }, [hasPermission]);

  const checkPermission = async () => {
    try {
      // iOS handles permissions natively, just set to true
      if (Platform.OS === 'ios') {
        setHasPermission(true);
        setIsChecking(false);
        return;
      }

      // Android permission checks
      if (Platform.OS === 'android') {
        // Quick permission check first
        const granted = await CameraXModule.checkCameraPermission();

        if (granted) {
          // Permission already granted, start camera immediately
          setHasPermission(true);
          setIsChecking(false);
          return;
        }

        // Check if camera is available
        const hasCamera = await CameraXModule.isCameraAvailable();
        if (!hasCamera) {
          setError('No camera available on this device');
          setIsChecking(false);
          return;
        }

        setHasPermission(false);
        setError(
          'Camera permission denied. Please grant permission in settings.',
        );
        setIsChecking(false);
      }
    } catch (error) {
      console.error('Error checking camera permission:', error);
      setError(`Error: ${error}`);
      setIsChecking(false);
    }
  };

  const requestPermission = async () => {
    try {
      if (Platform.OS === 'ios') {
        // iOS permissions are handled natively through Info.plist
        setHasPermission(true);
        setError('');
        return;
      }

      // Android permission request
      if (Platform.OS === 'android') {
        const result = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.CAMERA,
          {
            title: 'Camera Permission',
            message: 'This app needs access to your camera for pose detection',
            buttonNeutral: 'Ask Me Later',
            buttonNegative: 'Cancel',
            buttonPositive: 'OK',
          },
        );

        if (result === PermissionsAndroid.RESULTS.GRANTED) {
          setHasPermission(true);
          setError('');
        } else {
          setError('Camera permission denied');
        }
      }
    } catch (error) {
      console.error('Error requesting camera permission:', error);
      setError(`Error: ${error}`);
    }
  };

  const handleViewResults = () => {
    if (onBothCaptured) {
      const viewId = findNodeHandle(cameraViewRef.current);
      if (viewId) {
        UIManager.dispatchViewManagerCommand(
          viewId,
          (
            UIManager.getViewManagerConfig('CameraView').Commands
              .navigateToResult as any
          ).toString(),
          [],
        );
      }
      onBothCaptured();
    }
  };

  if (isChecking) {
    return (
      <View style={[styles.container, style]}>
        <Text style={styles.text}>Checking camera permission...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={[styles.container, style]}>
        <Text style={styles.errorText}>{error}</Text>
        {!hasPermission && !error.includes('only available') && (
          <Button title="Grant Permission" onPress={requestPermission} />
        )}
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={[styles.container, style]}>
        <Text style={styles.text}>Camera permission is required</Text>
        <Button title="Grant Permission" onPress={requestPermission} />
      </View>
    );
  }

  console.log('Rendering NativeCameraView with cameraType:', cameraType);
  console.log('🔗 onBothCaptured callback exists:', !!onBothCaptured);
  
  return (
    <NativeCameraView
      ref={cameraViewRef}
      style={style}
      cameraType={cameraType}
      onBothCaptured={(event: any) => {
        console.log('🎯 NativeCameraView onBothCaptured event received:', event);
        const native = event?.nativeEvent ?? {};
        console.log('   imageUri:', native.imageUri);
        console.log('   sideImageUri:', native.sideImageUri);

        if (onBothCaptured) {
          console.log('✅ Calling parent onBothCaptured callback with params');
          onBothCaptured({
            imageUri: native.imageUri || '',
            sideImageUri: native.sideImageUri || '',
          });
        } else {
          console.log('❌ No parent onBothCaptured callback set');
        }
      }}
    />
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  text: {
    color: '#fff',
    marginBottom: 16,
    fontSize: 16,
  },
  errorText: {
    color: '#ff6b6b',
    marginBottom: 16,
    fontSize: 16,
    textAlign: 'center',
    paddingHorizontal: 20,
  },
});
