# iOS Setup Helper Script
# This script will guide you through setting up the iOS project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  iOS Pose Detection Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Files created successfully:" -ForegroundColor Green
Write-Host "  ✓ PoseValidator.swift" -ForegroundColor Green
Write-Host "  ✓ SidePoseValidator.swift" -ForegroundColor Green
Write-Host "  ✓ BodyPositionChecker.swift" -ForegroundColor Green
Write-Host "  ✓ VoiceFeedbackProvider.swift" -ForegroundColor Green
Write-Host "  ✓ CameraViewManager.swift" -ForegroundColor Green
Write-Host "  ✓ CameraViewManager.m" -ForegroundColor Green
Write-Host "  ✓ PoseDetection-Bridging-Header.h" -ForegroundColor Green
Write-Host "  ✓ Info.plist updated with permissions" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  MANUAL STEPS REQUIRED (ON MAC)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Since you're on Windows, you'll need a Mac to complete iOS setup." -ForegroundColor Yellow
Write-Host ""

Write-Host "On your Mac, follow these steps:" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Transfer the project to your Mac" -ForegroundColor White
Write-Host ""

Write-Host "2. Install CocoaPods dependencies:" -ForegroundColor White
Write-Host "   cd ios" -ForegroundColor Gray
Write-Host "   pod install" -ForegroundColor Gray
Write-Host ""

Write-Host "3. Open Xcode:" -ForegroundColor White
Write-Host "   open PoseDetection.xcworkspace" -ForegroundColor Gray
Write-Host ""

Write-Host "4. In Xcode, add the Swift files to the project:" -ForegroundColor White
Write-Host "   a. Right-click 'PoseDetection' folder in Project Navigator" -ForegroundColor Gray
Write-Host "   b. Select 'Add Files to PoseDetection...'" -ForegroundColor Gray
Write-Host "   c. Select all these files:" -ForegroundColor Gray
Write-Host "      - PoseValidator.swift" -ForegroundColor DarkGray
Write-Host "      - SidePoseValidator.swift" -ForegroundColor DarkGray
Write-Host "      - BodyPositionChecker.swift" -ForegroundColor DarkGray
Write-Host "      - VoiceFeedbackProvider.swift" -ForegroundColor DarkGray
Write-Host "      - CameraViewManager.swift" -ForegroundColor DarkGray
Write-Host "      - CameraViewManager.m" -ForegroundColor DarkGray
Write-Host "   d. Check 'Copy items if needed'" -ForegroundColor Gray
Write-Host "   e. Make sure target 'PoseDetection' is selected" -ForegroundColor Gray
Write-Host ""

Write-Host "5. Configure Swift Bridging Header:" -ForegroundColor White
Write-Host "   a. Select project in Project Navigator" -ForegroundColor Gray
Write-Host "   b. Select 'PoseDetection' target" -ForegroundColor Gray
Write-Host "   c. Go to 'Build Settings' tab" -ForegroundColor Gray
Write-Host "   d. Search for 'Objective-C Bridging Header'" -ForegroundColor Gray
Write-Host "   e. Set to: PoseDetection/PoseDetection-Bridging-Header.h" -ForegroundColor Gray
Write-Host ""

Write-Host "6. Set Swift Version:" -ForegroundColor White
Write-Host "   a. In Build Settings, search for 'Swift Language Version'" -ForegroundColor Gray
Write-Host "   b. Set to 'Swift 5'" -ForegroundColor Gray
Write-Host ""

Write-Host "7. Build and Run:" -ForegroundColor White
Write-Host "   a. Select iPhone simulator or device" -ForegroundColor Gray
Write-Host "   b. Press Cmd+R to build and run" -ForegroundColor Gray
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "  iOS Architecture" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "The iOS implementation uses:" -ForegroundColor White
Write-Host "  • Vision framework - Built-in pose detection" -ForegroundColor Gray
Write-Host "  • AVFoundation - Camera management" -ForegroundColor Gray
Write-Host "  • AVSpeechSynthesizer - Voice feedback" -ForegroundColor Gray
Write-Host "  • Same two-stage capture flow as Android" -ForegroundColor Gray
Write-Host ""

Write-Host "See ios/README_IOS_SETUP.md for detailed documentation" -ForegroundColor Cyan
Write-Host ""
