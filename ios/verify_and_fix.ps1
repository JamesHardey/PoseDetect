# iOS Setup Verification and Instructions
# Run this to see what needs to be done in Xcode

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  iOS Native Module Setup Verification" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if files exist
Write-Host "📁 Checking if Swift files exist..." -ForegroundColor Yellow
Write-Host ""

$files = @(
    "PoseDetection\PoseValidator.swift",
    "PoseDetection\SidePoseValidator.swift",
    "PoseDetection\BodyPositionChecker.swift",
    "PoseDetection\VoiceFeedbackProvider.swift",
    "PoseDetection\CameraViewManager.swift",
    "PoseDetection\CameraViewManager.m",
    "PoseDetection\PoseDetection-Bridging-Header.h"
)

$allExist = $true
foreach ($file in $files) {
    $fullPath = Join-Path $PSScriptRoot $file
    if (Test-Path $fullPath) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (MISSING)" -ForegroundColor Red
        $allExist = $false
    }
}

Write-Host ""
if ($allExist) {
    Write-Host "✅ All files exist!" -ForegroundColor Green
} else {
    Write-Host "❌ Some files are missing!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Red
Write-Host "  ⚠️  FILES NOT ADDED TO XCODE PROJECT YET" -ForegroundColor Red
Write-Host "================================================" -ForegroundColor Red
Write-Host ""
Write-Host "The error 'View config not found for component CameraView'" -ForegroundColor Yellow
Write-Host "means the Swift files exist but haven't been added to Xcode." -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  FOLLOW THESE STEPS IN XCODE (ON MAC)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Open Xcode Workspace" -ForegroundColor White
Write-Host "───────────────────────────────" -ForegroundColor Gray
Write-Host "  cd ios" -ForegroundColor Gray
Write-Host "  open PoseDetection.xcworkspace" -ForegroundColor Gray
Write-Host ""

Write-Host "Step 2: Add Files to Xcode Project" -ForegroundColor White
Write-Host "────────────────────────────────────" -ForegroundColor Gray
Write-Host "  a. In Xcode Project Navigator (left panel)," -ForegroundColor Gray
Write-Host "     find the 'PoseDetection' folder (yellow folder icon)" -ForegroundColor Gray
Write-Host ""
Write-Host "  b. RIGHT-CLICK on 'PoseDetection' folder" -ForegroundColor Gray
Write-Host "     → Select 'Add Files to \"PoseDetection\"...'" -ForegroundColor Gray
Write-Host ""
Write-Host "  c. In the file picker, navigate to:" -ForegroundColor Gray
Write-Host "     ios/PoseDetection/" -ForegroundColor Gray
Write-Host ""
Write-Host "  d. SELECT THESE 7 FILES (Cmd+Click for multiple):" -ForegroundColor Gray
foreach ($file in $files) {
    $filename = Split-Path $file -Leaf
    Write-Host "     • $filename" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "  e. ✅ CHECK 'Copy items if needed'" -ForegroundColor Green
Write-Host "  f. ✅ CHECK 'PoseDetection' under 'Add to targets'" -ForegroundColor Green
Write-Host "  g. Click 'Add' button" -ForegroundColor Gray
Write-Host ""

Write-Host "Step 3: Configure Bridging Header" -ForegroundColor White
Write-Host "───────────────────────────────────" -ForegroundColor Gray
Write-Host "  a. Click 'PoseDetection' PROJECT (blue icon at top of navigator)" -ForegroundColor Gray
Write-Host "  b. Select 'PoseDetection' under TARGETS (not PROJECT)" -ForegroundColor Gray
Write-Host "  c. Click 'Build Settings' tab" -ForegroundColor Gray
Write-Host "  d. In search box, type: bridging" -ForegroundColor Gray
Write-Host "  e. Find 'Objective-C Bridging Header'" -ForegroundColor Gray
Write-Host "  f. Double-click the value field and enter:" -ForegroundColor Gray
Write-Host "     PoseDetection/PoseDetection-Bridging-Header.h" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 4: Set Swift Version" -ForegroundColor White
Write-Host "──────────────────────────" -ForegroundColor Gray
Write-Host "  a. Still in Build Settings, search for: swift" -ForegroundColor Gray
Write-Host "  b. Find 'Swift Language Version'" -ForegroundColor Gray
Write-Host "  c. Set to: Swift 5" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 5: Clean and Rebuild" -ForegroundColor White
Write-Host "──────────────────────────" -ForegroundColor Gray
Write-Host "  a. Menu: Product → Clean Build Folder (Shift+Cmd+K)" -ForegroundColor Gray
Write-Host "  b. Menu: Product → Build (Cmd+B)" -ForegroundColor Gray
Write-Host "  c. Wait for build to complete (should succeed)" -ForegroundColor Gray
Write-Host ""

Write-Host "Step 6: Run the App" -ForegroundColor White
Write-Host "────────────────────" -ForegroundColor Gray
Write-Host "  a. Select your iPhone/Simulator from device dropdown" -ForegroundColor Gray
Write-Host "  b. Click Run button (▶️) or press Cmd+R" -ForegroundColor Gray
Write-Host "  c. App should launch without 'View config not found' error" -ForegroundColor Gray
Write-Host ""

Write-Host "================================================" -ForegroundColor Green
Write-Host "  ✅ After completing steps, error will be fixed!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

Write-Host "💡 TIP: If you get 'Cannot find type' errors during build:" -ForegroundColor Yellow
Write-Host "   → Make sure ALL 7 files have checkmark under Target Membership" -ForegroundColor Yellow
Write-Host "   → Select each file → File Inspector → Target: PoseDetection" -ForegroundColor Yellow
Write-Host ""

Write-Host "For detailed troubleshooting, see: ios/TROUBLESHOOTING.md" -ForegroundColor Cyan
Write-Host ""
