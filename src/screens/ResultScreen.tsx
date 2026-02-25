import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  ActivityIndicator,
  Platform,
  ScrollView,
  Modal,
  Alert,
  PermissionsAndroid,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import AsyncStorage from '@react-native-async-storage/async-storage';
import RNFS from 'react-native-fs';

type ResultScreenNavigationProp = NativeStackNavigationProp<
  RootStackParamList,
  'Result'
>;

type ResultScreenRouteProp = RouteProp<RootStackParamList, 'Result'>;

interface Props {
  navigation: ResultScreenNavigationProp;
  route: ResultScreenRouteProp;
}

interface SavedPose {
  id: string;
  frontUri: string;
  sideUri: string;
  timestamp: number;
  date: string;
}

const ResultScreen: React.FC<Props> = ({ navigation, route }) => {
  console.log('🎯 ResultScreen mounted');
  console.log('📋 Route object:', JSON.stringify(route, null, 2));
  console.log('📋 Route params:', JSON.stringify(route.params, null, 2));
  console.log('📋 Route params type:', typeof route.params);

  // Safely extract params with fallback
  const { imageUri, sideImageUri } = route.params || {};

  console.log('✅ Extracted params:');
  console.log('   imageUri:', imageUri);
  console.log('   imageUri type:', typeof imageUri);
  console.log('   sideImageUri:', sideImageUri);
  console.log('   sideImageUri type:', typeof sideImageUri);

  // State hooks must be at top level
  const [frontImageError, setFrontImageError] = useState(false);
  const [sideImageError, setSideImageError] = useState(false);
  const [frontLoading, setFrontLoading] = useState(true);
  const [sideLoading, setSideLoading] = useState(!!sideImageUri);
  const [fullScreenImage, setFullScreenImage] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  // Validate required params after state declarations
  if (!imageUri) {
    console.error('❌ CRITICAL ERROR: No imageUri provided to ResultScreen!');
    console.error('   This means navigation params are invalid or missing');
    console.error('   Full route.params:', route.params);

    // Show error UI instead of crashing
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorScreen}>
          <Text style={styles.errorTitle}>❌ Error</Text>
          <Text style={styles.errorMessage}>No image data received</Text>
          <TouchableOpacity
            style={styles.retryButton}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.retryButtonText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // For Android content URIs, we need to use them directly
  const displayFrontUri =
    imageUri &&
    (Platform.OS === 'android' && imageUri.startsWith('content://')
      ? imageUri
      : imageUri.startsWith('file://')
      ? imageUri
      : `file://${imageUri}`);

  const displaySideUri = sideImageUri
    ? Platform.OS === 'android' && sideImageUri.startsWith('content://')
      ? sideImageUri
      : sideImageUri.startsWith('file://')
      ? sideImageUri
      : `file://${sideImageUri}`
    : null;

  // useEffect(() => {
  //   console.log('📱 Result screen - Front Image URI:', imageUri);
  //   console.log('📱 Result screen - Side Image URI:', sideImageUri);
  //   console.log('📱 Result screen - Display Front URI:', displayFrontUri);
  //   console.log('📱 Result screen - Display Side URI:', displaySideUri);
  // }, [imageUri, sideImageUri, displayFrontUri, displaySideUri]);

  const requestStoragePermission = async (): Promise<boolean> => {
    if (Platform.OS !== 'android') return true;

    try {
      if (Platform.Version >= 33) {
        // Android 13+, no permission needed for app-specific storage
        return true;
      }

      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
        {
          title: 'Storage Permission',
          message: 'This app needs access to storage to save your poses',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        },
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    } catch (err) {
      console.error('Permission error:', err);
      return false;
    }
  };

  const savePose = async () => {
    try {
      setIsSaving(true);

      const hasPermission = await requestStoragePermission();
      if (!hasPermission) {
        Alert.alert(
          'Permission Denied',
          'Storage permission is required to save poses',
        );
        setIsSaving(false);
        return;
      }

      // Create permanent storage directory
      const poseDir = `${RNFS.DocumentDirectoryPath}/poses`;
      await RNFS.mkdir(poseDir);

      // Generate unique ID
      const id = Date.now().toString();
      const frontFileName = `front_${id}.jpg`;
      const sideFileName = `side_${id}.jpg`;
      const frontDestPath = `${poseDir}/${frontFileName}`;
      const sideDestPath = `${poseDir}/${sideFileName}`;

      // Copy from cache to permanent storage
      const frontSource = imageUri.replace('file://', '');
      const sideSource = sideImageUri?.replace('file://', '') || '';

      await RNFS.copyFile(frontSource, frontDestPath);
      if (sideSource) {
        await RNFS.copyFile(sideSource, sideDestPath);
      }

      // Save to history
      const savedPose: SavedPose = {
        id,
        frontUri: `file://${frontDestPath}`,
        sideUri: `file://${sideDestPath}`,
        timestamp: Date.now(),
        date: new Date().toLocaleDateString(),
      };

      // Get existing history
      const historyJson = await AsyncStorage.getItem('poseHistory');
      const history: SavedPose[] = historyJson ? JSON.parse(historyJson) : [];

      // Add new pose to history
      history.unshift(savedPose);

      // Keep only last 50 poses
      const trimmedHistory = history.slice(0, 50);

      // Save updated history
      await AsyncStorage.setItem('poseHistory', JSON.stringify(trimmedHistory));

      Alert.alert('Success', 'Pose saved successfully!', [
        { text: 'OK', onPress: () => setIsSaving(false) },
      ]);
    } catch (error) {
      console.error('Error saving pose:', error);
      Alert.alert('Error', 'Failed to save pose. Please try again.');
      setIsSaving(false);
    }
  };

  const handleRetake = () => {
    navigation.navigate('Camera');
  };

  const handleDone = () => {
    navigation.navigate('Intro');
  };

  const handleViewHistory = () => {
    navigation.navigate('History' as any);
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.content}
        contentContainerStyle={styles.scrollContent}
      >
        <Text style={styles.title}>Perfect Shots! 📸</Text>

        {/* Both Images Side by Side */}
        <View style={styles.imagesRow}>
          {/* Front Pose Image */}
          <TouchableOpacity
            style={styles.imageSection}
            onPress={() => setFullScreenImage(displayFrontUri)}
            activeOpacity={0.8}
          >
            <Text style={styles.imageLabel}>Front Pose</Text>
            <View style={styles.imageContainer}>
              {frontLoading && (
                <View style={styles.loadingContainer}>
                  <ActivityIndicator size="large" color="#007AFF" />
                </View>
              )}

              {frontImageError ? (
                <View style={styles.errorContainer}>
                  <Text style={styles.errorText}>Failed to load</Text>
                </View>
              ) : (
                <Image
                  source={{ uri: displayFrontUri }}
                  style={styles.image}
                  resizeMode="cover"
                  onLoad={() => {
                    setFrontLoading(false);
                    console.log('Front image loaded successfully');
                  }}
                  onError={error => {
                    setFrontLoading(false);
                    setFrontImageError(true);
                    console.error(
                      'Front image load error:',
                      error.nativeEvent.error,
                    );
                  }}
                />
              )}
            </View>
            <Text style={styles.tapHint}>Tap to view fullscreen</Text>
          </TouchableOpacity>

          {/* Side Pose Image */}
          {displaySideUri && (
            <TouchableOpacity
              style={styles.imageSection}
              onPress={() => setFullScreenImage(displaySideUri)}
              activeOpacity={0.8}
            >
              <Text style={styles.imageLabel}>Side Pose</Text>
              <View style={styles.imageContainer}>
                {sideLoading && (
                  <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color="#007AFF" />
                  </View>
                )}

                {sideImageError ? (
                  <View style={styles.errorContainer}>
                    <Text style={styles.errorText}>Failed to load</Text>
                  </View>
                ) : (
                  <Image
                    source={{ uri: displaySideUri }}
                    style={styles.image}
                    resizeMode="cover"
                    onLoad={() => {
                      setSideLoading(false);
                      console.log('Side image loaded successfully');
                    }}
                    onError={error => {
                      setSideLoading(false);
                      setSideImageError(true);
                      console.error(
                        'Side image load error:',
                        error.nativeEvent.error,
                      );
                    }}
                  />
                )}
              </View>
              <Text style={styles.tapHint}>Tap to view fullscreen</Text>
            </TouchableOpacity>
          )}
        </View>

        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.saveButton, isSaving && styles.buttonDisabled]}
            onPress={savePose}
            disabled={isSaving}
          >
            {isSaving ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.saveButtonText}>💾 Save Pose</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.historyButton}
            onPress={handleViewHistory}
          >
            <Text style={styles.historyButtonText}>📁 View History</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.retakeButton} onPress={handleRetake}>
            <Text style={styles.retakeButtonText}>📷 Retake Photos</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.doneButton} onPress={handleDone}>
            <Text style={styles.doneButtonText}>✓ Done</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* Full Screen Image Modal */}
      <Modal
        visible={fullScreenImage !== null}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setFullScreenImage(null)}
      >
        <View style={styles.fullScreenContainer}>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => setFullScreenImage(null)}
          >
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>

          {fullScreenImage && (
            <Image
              source={{ uri: fullScreenImage }}
              style={styles.fullScreenImage}
              resizeMode="contain"
            />
          )}
        </View>
      </Modal>
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
  },
  scrollContent: {
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginTop: 20,
    marginBottom: 20,
  },
  imagesRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  imageSection: {
    flex: 1,
  },
  imageLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 8,
    textAlign: 'center',
  },
  imageContainer: {
    height: 400,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  tapHint: {
    fontSize: 12,
    color: '#888',
    textAlign: 'center',
    marginTop: 6,
  },
  buttonContainer: {
    gap: 12,
    marginTop: 12,
  },
  saveButton: {
    backgroundColor: '#4CAF50',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  historyButton: {
    backgroundColor: '#FF9800',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
  },
  historyButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  retakeButton: {
    backgroundColor: '#333',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
  },
  retakeButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  doneButton: {
    backgroundColor: '#007AFF',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
  },
  doneButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    color: '#ff4444',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  fullScreenContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  backButton: {
    position: 'absolute',
    top: 50,
    left: 20,
    zIndex: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 25,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  fullScreenImage: {
    width: '100%',
    height: '100%',
  },
  errorScreen: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ff4444',
    marginBottom: 16,
  },
  errorMessage: {
    fontSize: 18,
    color: '#fff',
    textAlign: 'center',
    marginBottom: 32,
  },
  retryButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default ResultScreen;
