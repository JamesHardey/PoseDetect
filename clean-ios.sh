#!/bin/bash

echo "🧹 Cleaning project..."

# Stop Metro if running
killall node 2>/dev/null

# Remove JS dependencies
rm -rf node_modules
rm -f package-lock.json

# Clear Metro cache
rm -rf $TMPDIR/metro-*
rm -rf $TMPDIR/haste-map-*

# Clean iOS
cd ios || exit
rm -rf Pods
rm -f Podfile.lock
cd ..

# Clean npm cache
npm cache clean --force

echo "📦 Reinstalling dependencies..."
npm install

echo "📱 Installing pods..."
cd ios || exit
pod install
cd ..

echo "🚀 Done. Now run:"
echo "npx react-native start --reset-cache"
echo "Then in another terminal:"
echo "npx react-native run-ios"