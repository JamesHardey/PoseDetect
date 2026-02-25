#!/bin/bash

echo "================================================"
echo "  iOS Native Module Setup Verification"
echo "================================================"
echo ""

# Navigate to iOS directory
cd "$(dirname "$0")"

echo "📁 Checking if Swift files exist..."
files=(
    "PoseDetection/PoseValidator.swift"
    "PoseDetection/SidePoseValidator.swift"
    "PoseDetection/BodyPositionChecker.swift"
    "PoseDetection/VoiceFeedbackProvider.swift"
    "PoseDetection/CameraViewManager.swift"
    "PoseDetection/CameraViewManager.m"
    "PoseDetection/PoseDetection-Bridging-Header.h"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
        all_exist=false
    fi
done

echo ""
if [ "$all_exist" = true ]; then
    echo "✅ All files exist!"
else
    echo "❌ Some files are missing!"
    exit 1
fi

echo ""
echo "================================================"
echo "  NEXT STEPS - MUST BE DONE IN XCODE"
echo "================================================"
echo ""
echo "The files exist but aren't added to Xcode project."
echo "You MUST complete these steps in Xcode:"
echo ""
echo "1️⃣  Open Xcode workspace:"
echo "    open PoseDetection.xcworkspace"
echo ""
echo "2️⃣  Add files to project:"
echo "    a. In Xcode, right-click 'PoseDetection' folder (yellow folder)"
echo "    b. Select 'Add Files to \"PoseDetection\"...'"
echo "    c. Navigate to ios/PoseDetection/"
echo "    d. SELECT THESE FILES (Cmd+Click to select multiple):"
for file in "${files[@]}"; do
    filename=$(basename "$file")
    echo "       • $filename"
done
echo "    e. ✅ CHECK 'Copy items if needed'"
echo "    f. ✅ CHECK 'PoseDetection' under 'Add to targets'"
echo "    g. Click 'Add'"
echo ""
echo "3️⃣  Set Bridging Header:"
echo "    a. Click 'PoseDetection' project (blue icon at top)"
echo "    b. Select 'PoseDetection' under TARGETS"
echo "    c. Click 'Build Settings' tab"
echo "    d. Search for: Objective-C Bridging Header"
echo "    e. Double-click and set to:"
echo "       PoseDetection/PoseDetection-Bridging-Header.h"
echo ""
echo "4️⃣  Set Swift Version:"
echo "    a. In Build Settings, search for: Swift Language"
echo "    b. Set 'Swift Language Version' to: Swift 5"
echo ""
echo "5️⃣  Clean and rebuild:"
echo "    a. Product → Clean Build Folder (Shift+Cmd+K)"
echo "    b. Product → Build (Cmd+B)"
echo ""
echo "6️⃣  Run the app:"
echo "    a. Select your device/simulator"
echo "    b. Click Run (Cmd+R)"
echo ""
echo "================================================"
echo ""
echo "After completing these steps, the error will be fixed!"
echo ""
