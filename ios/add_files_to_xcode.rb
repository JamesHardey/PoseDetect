#!/usr/bin/env ruby

require 'xcodeproj'

project_path = 'PoseDetection.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Get the main target
target = project.targets.first

# Get the PoseDetection group
pose_detection_group = project.main_group.find_subpath('PoseDetection', true)

# Files to add
swift_files = [
  'PoseDetection/PoseValidator.swift',
  'PoseDetection/SidePoseValidator.swift',
  'PoseDetection/BodyPositionChecker.swift',
  'PoseDetection/VoiceFeedbackProvider.swift',
  'PoseDetection/CameraViewManager.swift'
]

objc_files = [
  'PoseDetection/CameraViewManager.m'
]

header_files = [
  'PoseDetection/PoseDetection-Bridging-Header.h'
]

# Add Swift files
swift_files.each do |file_path|
  file_ref = pose_detection_group.new_reference(file_path)
  target.add_file_references([file_ref])
  puts "Added: #{file_path}"
end

# Add Objective-C files
objc_files.each do |file_path|
  file_ref = pose_detection_group.new_reference(file_path)
  target.add_file_references([file_ref])
  puts "Added: #{file_path}"
end

# Add header files (no need to add to target)
header_files.each do |file_path|
  pose_detection_group.new_reference(file_path)
  puts "Added: #{file_path}"
end

# Set bridging header
target.build_configurations.each do |config|
  config.build_settings['SWIFT_OBJC_BRIDGING_HEADER'] = 'PoseDetection/PoseDetection-Bridging-Header.h'
  config.build_settings['SWIFT_VERSION'] = '5.0'
end

project.save

puts "\n✅ Successfully added files to Xcode project!"
puts "Next steps:"
puts "1. cd ios"
puts "2. pod install"
puts "3. Open PoseDetection.xcworkspace in Xcode"
puts "4. Build and run"
