import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  SafeAreaView,
  Image,
  ActivityIndicator,
  Alert,
  Modal,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';
import AsyncStorage from '@react-native-async-storage/async-storage';
import RNFS from 'react-native-fs';

type HistoryScreenNavigationProp = NativeStackNavigationProp<
  RootStackParamList,
  'History'
>;

interface Props {
  navigation: HistoryScreenNavigationProp;
}

interface SavedPose {
  id: string;
  frontUri: string;
  sideUri: string;
  timestamp: number;
  date: string;
}

const HistoryScreen: React.FC<Props> = ({ navigation }) => {
  const [history, setHistory] = useState<SavedPose[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPose, setSelectedPose] = useState<SavedPose | null>(null);
  const [fullScreenImage, setFullScreenImage] = useState<string | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const historyJson = await AsyncStorage.getItem('poseHistory');
      const savedHistory: SavedPose[] = historyJson ? JSON.parse(historyJson) : [];
      setHistory(savedHistory);
    } catch (error) {
      console.error('Error loading history:', error);
      Alert.alert('Error', 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const deletePose = async (pose: SavedPose) => {
    Alert.alert(
      'Delete Pose',
      'Are you sure you want to delete this pose?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              // Delete files
              const frontPath = pose.frontUri.replace('file://', '');
              const sidePath = pose.sideUri.replace('file://', '');
              
              if (await RNFS.exists(frontPath)) {
                await RNFS.unlink(frontPath);
              }
              if (await RNFS.exists(sidePath)) {
                await RNFS.unlink(sidePath);
              }

              // Update history
              const updatedHistory = history.filter(p => p.id !== pose.id);
              setHistory(updatedHistory);
              await AsyncStorage.setItem('poseHistory', JSON.stringify(updatedHistory));

              Alert.alert('Success', 'Pose deleted successfully');
            } catch (error) {
              console.error('Error deleting pose:', error);
              Alert.alert('Error', 'Failed to delete pose');
            }
          },
        },
      ]
    );
  };

  const renderPoseItem = ({ item }: { item: SavedPose }) => (
    <TouchableOpacity
      style={styles.poseItem}
      onPress={() => setSelectedPose(item)}
      activeOpacity={0.7}
    >
      <View style={styles.poseImages}>
        <View style={styles.thumbnailContainer}>
          <Text style={styles.thumbnailLabel}>Front</Text>
          <Image
            source={{ uri: item.frontUri }}
            style={styles.thumbnail}
            resizeMode="cover"
          />
        </View>
        <View style={styles.thumbnailContainer}>
          <Text style={styles.thumbnailLabel}>Side</Text>
          <Image
            source={{ uri: item.sideUri }}
            style={styles.thumbnail}
            resizeMode="cover"
          />
        </View>
      </View>
      <View style={styles.poseInfo}>
        <Text style={styles.poseDate}>{item.date}</Text>
        <Text style={styles.poseTime}>
          {new Date(item.timestamp).toLocaleTimeString()}
        </Text>
      </View>
      <TouchableOpacity
        style={styles.deleteButton}
        onPress={() => deletePose(item)}
      >
        <Text style={styles.deleteButtonText}>🗑️</Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Pose History</Text>
        <View style={styles.placeholder} />
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Loading history...</Text>
        </View>
      ) : history.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No saved poses yet</Text>
          <Text style={styles.emptySubtext}>
            Capture and save poses to see them here
          </Text>
        </View>
      ) : (
        <FlatList
          data={history}
          renderItem={renderPoseItem}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.listContent}
        />
      )}

      {/* Pose Detail Modal */}
      <Modal
        visible={selectedPose !== null}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setSelectedPose(null)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>
                {selectedPose?.date} - {new Date(selectedPose?.timestamp || 0).toLocaleTimeString()}
              </Text>
              <TouchableOpacity
                style={styles.closeButton}
                onPress={() => setSelectedPose(null)}
              >
                <Text style={styles.closeButtonText}>✕</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.modalImages}>
              <TouchableOpacity
                style={styles.modalImageContainer}
                onPress={() => setFullScreenImage(selectedPose?.frontUri || null)}
              >
                <Text style={styles.modalImageLabel}>Front Pose</Text>
                <Image
                  source={{ uri: selectedPose?.frontUri }}
                  style={styles.modalImage}
                  resizeMode="cover"
                />
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.modalImageContainer}
                onPress={() => setFullScreenImage(selectedPose?.sideUri || null)}
              >
                <Text style={styles.modalImageLabel}>Side Pose</Text>
                <Image
                  source={{ uri: selectedPose?.sideUri }}
                  style={styles.modalImage}
                  resizeMode="cover"
                />
              </TouchableOpacity>
            </View>

            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={() => setSelectedPose(null)}
            >
              <Text style={styles.modalCloseButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Full Screen Image Modal */}
      <Modal
        visible={fullScreenImage !== null}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setFullScreenImage(null)}
      >
        <View style={styles.fullScreenContainer}>
          <TouchableOpacity
            style={styles.fullScreenBackButton}
            onPress={() => setFullScreenImage(null)}
          >
            <Text style={styles.fullScreenBackText}>← Back</Text>
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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  placeholder: {
    width: 60,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#fff',
    fontSize: 16,
    marginTop: 12,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  emptySubtext: {
    color: '#888',
    fontSize: 14,
    textAlign: 'center',
  },
  listContent: {
    padding: 16,
  },
  poseItem: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    flexDirection: 'row',
    alignItems: 'center',
  },
  poseImages: {
    flexDirection: 'row',
    gap: 8,
    flex: 1,
  },
  thumbnailContainer: {
    flex: 1,
  },
  thumbnailLabel: {
    color: '#888',
    fontSize: 12,
    marginBottom: 4,
    textAlign: 'center',
  },
  thumbnail: {
    width: '100%',
    height: 120,
    borderRadius: 8,
  },
  poseInfo: {
    marginLeft: 12,
    marginRight: 12,
  },
  poseDate: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  poseTime: {
    color: '#888',
    fontSize: 12,
    marginTop: 4,
  },
  deleteButton: {
    padding: 8,
  },
  deleteButtonText: {
    fontSize: 24,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    justifyContent: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  modalTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  closeButton: {
    padding: 8,
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  modalImages: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  modalImageContainer: {
    flex: 1,
  },
  modalImageLabel: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
    textAlign: 'center',
  },
  modalImage: {
    width: '100%',
    height: 300,
    borderRadius: 8,
  },
  modalCloseButton: {
    backgroundColor: '#007AFF',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  modalCloseButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  fullScreenContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullScreenBackButton: {
    position: 'absolute',
    top: 50,
    left: 20,
    zIndex: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 25,
  },
  fullScreenBackText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  fullScreenImage: {
    width: '100%',
    height: '100%',
  },
});

export default HistoryScreen;
