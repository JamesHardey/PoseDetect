import { NativeModules } from 'react-native';

interface CameraXModuleInterface {
  checkCameraPermission(): Promise<boolean>;
  requestCameraPermission(): Promise<boolean>;
  isCameraAvailable(): Promise<boolean>;
}

const { CameraXModule } = NativeModules;

export default CameraXModule as CameraXModuleInterface;
