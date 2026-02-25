import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  Image,
  PermissionsAndroid,
  Platform,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';

type IntroScreenNavigationProp = NativeStackNavigationProp<
  RootStackParamList,
  'Intro'
>;

interface Props {
  navigation: IntroScreenNavigationProp;
}

const IntroScreen: React.FC<Props> = ({ navigation }) => {
  const [permissionGranted, setPermissionGranted] = useState(false);

  useEffect(() => {
    checkCameraPermission();
  }, []);

  const checkCameraPermission = async () => {
    if (Platform.OS !== 'android') {
      setPermissionGranted(true);
      return;
    }

    try {
      const granted = await PermissionsAndroid.check(
        PermissionsAndroid.PERMISSIONS.CAMERA
      );
      setPermissionGranted(granted);
    } catch (err) {
      console.warn(err);
      setPermissionGranted(false);
    }
  };

  const requestPermission = async () => {
    if (Platform.OS !== 'android') {
      setPermissionGranted(true);
      return;
    }

    try {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.CAMERA,
        {
          title: 'Camera Permission Required',
          message: 'This app needs access to your camera for pose detection',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        }
      );

      if (granted === PermissionsAndroid.RESULTS.GRANTED) {
        setPermissionGranted(true);
      }
    } catch (err) {
      console.warn(err);
    }
  };

  const handleButtonPress = () => {
    if (permissionGranted) {
      navigation.navigate('Camera');
    } else {
      requestPermission();
    }
  };

  const handleViewHistory = () => {
    navigation.navigate('History');
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.emoji}>📸</Text>
          <Text style={styles.title}>Pose Detection</Text>
          <Text style={styles.subtitle}>
            AI-Powered Real-time Pose Detection
          </Text>
        </View>

        <View style={styles.featuresContainer}>
          <View style={styles.feature}>
            <Text style={styles.featureIcon}>🎯</Text>
            <Text style={styles.featureTitle}>Real-time Detection</Text>
            <Text style={styles.featureDescription}>
              Detect body poses in real-time using your device camera
            </Text>
          </View>

          {/* <View style={styles.feature}>
            <Text style={styles.featureIcon}>🔄</Text>
            <Text style={styles.featureTitle}>Front & Back Camera</Text>
            <Text style={styles.featureDescription}>
              Switch between front and back camera seamlessly
            </Text>
          </View> */}

          <View style={styles.feature}>
            <Text style={styles.featureIcon}>⚡</Text>
            <Text style={styles.featureTitle}>Fast & Accurate</Text>
            <Text style={styles.featureDescription}>
              Powered by CameraX and ML Kit for optimal performance
            </Text>
          </View>
        </View>

        <TouchableOpacity
          style={[styles.button, !permissionGranted && styles.buttonDisabled]}
          onPress={handleButtonPress}>
          <Text style={styles.buttonText}>
            {permissionGranted ? 'Start Detection' : 'Grant Camera Permission'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.historyButton}
          onPress={handleViewHistory}>
          <Text style={styles.historyButtonText}>📁 View Saved Poses</Text>
        </TouchableOpacity>

        <Text style={styles.footer}>
          {permissionGranted 
            ? 'Ready to detect pose'
            : 'Please grant camera permission to continue'}
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  content: {
    flex: 1,
    padding: 24,
    justifyContent: 'space-between',
  },
  header: {
    alignItems: 'center',
    marginTop: 40,
  },
  emoji: {
    fontSize: 80,
    marginBottom: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
  },
  featuresContainer: {
    flex: 1,
    justifyContent: 'center',
    gap: 24,
  },
  feature: {
    backgroundColor: '#1a1a1a',
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
  },
  featureIcon: {
    fontSize: 40,
    marginBottom: 8,
  },
  featureTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 4,
  },
  featureDescription: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 12,
  },
  buttonDisabled: {
    backgroundColor: '#333',
    opacity: 0.5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  historyButton: {
    backgroundColor: '#FF9800',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 16,
  },
  historyButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  footer: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
});

export default IntroScreen;
