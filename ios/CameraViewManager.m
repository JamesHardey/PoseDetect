#import <React/RCTViewManager.h>
#import <React/RCTEventEmitter.h>

@interface RCT_EXTERN_MODULE(CameraViewManager, RCTViewManager)

RCT_EXPORT_VIEW_PROPERTY(onCaptureStatus, RCTDirectEventBlock)
RCT_EXPORT_VIEW_PROPERTY(onBothCaptured, RCTDirectEventBlock)
RCT_EXPORT_VIEW_PROPERTY(cameraType, NSString)

RCT_EXTERN_METHOD(setCameraType:(nonnull NSNumber *)node cameraType:(NSString *)cameraType)

@end
