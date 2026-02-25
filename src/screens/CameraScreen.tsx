import React, { useEffect, useState } from 'react';
import {
  View,
  StyleSheet,
  DeviceEventEmitter,
  TouchableOpacity,
  Text,
} from 'react-native';
import { CameraView } from '../components/CameraView';
import { useNavigation, useIsFocused } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../../App';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Camera'>;

const CameraScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp>();
  const isFocused = useIsFocused();
  const [bothCaptured, setBothCaptured] = useState(false);
  const [capturedUris, setCapturedUris] = useState<{
    frontUri: string;
    sideUri: string;
  } | null>(null);
  const [status, setStatus] = useState<any>('Waiting...');

  useEffect(() => {
    console.log('CameraScreen: Setting up event listeners');

    // Reset state when screen is focused
    if (isFocused) {
      setBothCaptured(false);
      setCapturedUris(null);
      setStatus('Waiting...');
    }

    // Listen for capture status updates
    const statusSubscription = DeviceEventEmitter.addListener(
      'onCaptureStatus',
      (event: { status: string; message: string }) => {
        console.log('CameraScreen: Status event received:', event);

        switch (event.status) {
          case 'camera_started':
            setStatus('Camera started and ready!');
            break;
          case 'ready_to_capture':
            setStatus('Ready to capture front pose!');
            break;
          case 'front_pose_captured':
            setStatus('Front pose captured! Turn sideways...');
            break;
          case 'ready_to_capture_side':
            setStatus('Ready to capture side pose!');
            break;
          case 'both_poses_captured':
            setStatus('Both poses captured! Processing...');
            break;
          default:
            setStatus(event.message || event.status);
        }
      },
    );

    // Listen for the event when both images are captured
    const imagesSubscription = DeviceEventEmitter.addListener(
      'onBothImagesCaptured',
      (event: { frontUri: string; sideUri: string }) => {
        console.log('🔔 CameraScreen: onBothImagesCaptured event received!');
        console.log('📸 Event data:', event);
        console.log('   Front URI:', event.frontUri);
        console.log('   Side URI:', event.sideUri);

        setStatus('Images ready! Click to view results');

        // Store URIs and show button
        setCapturedUris(event);
        setBothCaptured(true);

        console.log('✅ State updated - bothCaptured: true, capturedUris set');
      },
    );

    return () => {
      console.log('CameraScreen: Removing event listeners');
      statusSubscription.remove();
      imagesSubscription.remove();
    };
  }, [navigation, isFocused]);

  const handleViewResults = () => {
    console.log('🚀 handleViewResults called');
    console.log('📸 capturedUris:', capturedUris);

    if (capturedUris) {
      console.log('✅ Navigating to Result screen with:');
      console.log('   Front URI:', capturedUris.frontUri);
      console.log('   Side URI:', capturedUris.sideUri);

      navigation.navigate('Result', {
        imageUri: capturedUris.frontUri || '',
        sideImageUri: capturedUris.sideUri || '',
      });

      console.log('✅ Navigation.navigate() called');
    } else {
      console.log('❌ ERROR: capturedUris is null, cannot navigate');
    }
  };

  return (
    <View style={styles.container}>
      {isFocused && (
        <CameraView
          style={styles.camera}
          cameraType="front"
          onBothCaptured={handleViewResults}
        />
      )}

      {bothCaptured && (
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={handleViewResults}>
            <Text style={styles.buttonText}>View Results</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  statusContainer: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  statusText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  buttonContainer: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  button: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 40,
    paddingVertical: 15,
    borderRadius: 25,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default CameraScreen;
