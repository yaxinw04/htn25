#!/usr/bin/env python3
"""
Threshold-based MindRove → Minecraft Controller
===============================================

Uses calibrated thresholds for real-time gesture detection:
- Z acceleration for up/down (mining/placing)  
- Gyroscope (X/Y) for left/right turning
- Fast and responsive threshold-based detection

Usage: python threshold_key_mapper.py --thresholds ../mindrove/improved_thresholds.json
"""

import argparse
import json
import time
import numpy as np
from collections import deque
from threading import Lock
import signal
import sys

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets, IpProtocolTypes
from mindrove.data_filter import DataFilter, FilterTypes
from mindrove.exit_codes import MindRoveError

# Keyboard control
from pynput import keyboard

# Configuration
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET = MindRovePresets.DEFAULT_PRESET.value
SAMPLE_RATE = 500

# Gesture detection parameters
WINDOW_SIZE = 25        # ~50ms windows at 500Hz
MIN_CONFIDENCE = 0.8    # Higher minimum confidence for gesture detection
COOLDOWN_TIME = 0.5     # Longer cooldown between repeated actions
DETECTION_THRESHOLD = 1.2  # Additional multiplier for thresholds

class ThresholdGestureDetector:
    """Real-time threshold-based gesture detection"""
    
    def __init__(self, thresholds_file: str):
        self.thresholds = self._load_thresholds(thresholds_file)
        self.sample_rate = SAMPLE_RATE
        
        # Data buffers
        self.accel_buffer = deque(maxlen=WINDOW_SIZE)
        self.gyro_buffer = deque(maxlen=WINDOW_SIZE)
        self.buffer_lock = Lock()
        
        print(f"🎯 Loaded thresholds from {thresholds_file}")
        self._print_thresholds()
        
    def _load_thresholds(self, filename: str) -> dict:
        """Load calibrated thresholds"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return data['thresholds']
        except Exception as e:
            raise RuntimeError(f"Failed to load thresholds: {e}")
    
    def _print_thresholds(self):
        """Print loaded thresholds for verification"""
        print("📊 Gesture thresholds:")
        if 'swing_az_valley_threshold' in self.thresholds:
            print(f"   ⛏️  Mining (swing_down): az_valley < {self.thresholds['swing_az_valley_threshold']:.2f}")
        if 'arm_az_peak_threshold' in self.thresholds:
            print(f"   🧱 Placing (arm_up): az_peak > {self.thresholds['arm_az_peak_threshold']:.2f}")
        if 'turn_left_gyro_threshold' in self.thresholds:
            axis = self.thresholds.get('turn_left_axis', 'gy')
            print(f"   👈 Turn left: {axis} > {self.thresholds['turn_left_gyro_threshold']:.1f}")
        if 'turn_right_gyro_threshold' in self.thresholds:
            axis = self.thresholds.get('turn_right_axis', 'gy')  
            print(f"   👉 Turn right: {axis} < {self.thresholds['turn_right_gyro_threshold']:.1f}")
    
    def add_sample(self, accel_data: np.ndarray, gyro_data: np.ndarray):
        """Add new sample to sliding window buffers"""
        with self.buffer_lock:
            self.accel_buffer.append(accel_data.copy())
            self.gyro_buffer.append(gyro_data.copy())
    
    def detect_gesture(self) -> tuple:
        """Detect gesture from current buffer window
        
        Returns:
            tuple: (gesture_name, confidence) or (None, 0.0)
        """
        with self.buffer_lock:
            if len(self.accel_buffer) < WINDOW_SIZE:
                return None, 0.0
                
            # Get current window data
            accel_window = np.array(list(self.accel_buffer))  # Shape: (window_size, 3)
            gyro_window = np.array(list(self.gyro_buffer))    # Shape: (window_size, 3)
        
        # Extract axes
        ax, ay, az = accel_window[:, 0], accel_window[:, 1], accel_window[:, 2]
        gx, gy, gz = gyro_window[:, 0], gyro_window[:, 1], gyro_window[:, 2]
        
        # Calculate key metrics
        az_peak = np.max(az)
        az_valley = np.min(az)
        
        # Check for swing down (mining) - downward Z acceleration spike
        if 'swing_az_valley_threshold' in self.thresholds:
            if az_valley < self.thresholds['swing_az_valley_threshold'] * DETECTION_THRESHOLD:
                # Additional validation - check for motion intensity
                motion_intensity = np.std(az)
                swing_thresh = self.thresholds.get('swing_intensity_threshold', 0.1) * DETECTION_THRESHOLD
                
                if motion_intensity > swing_thresh:
                    confidence = min(1.0, abs(az_valley) / abs(self.thresholds['swing_az_valley_threshold']))
                    return 'swing_down', confidence
        
        # Check for arm up (placing) - upward Z acceleration
        if 'arm_az_peak_threshold' in self.thresholds:
            if az_peak > self.thresholds['arm_az_peak_threshold'] * DETECTION_THRESHOLD:
                motion_intensity = np.std(az)
                arm_thresh = self.thresholds.get('arm_swing_threshold', 0.1) * DETECTION_THRESHOLD
                
                if motion_intensity > arm_thresh:
                    confidence = min(1.0, az_peak / self.thresholds['arm_az_peak_threshold'])
                    return 'arm_up', confidence
        
        # Check for turn left
        if 'turn_left_gyro_threshold' in self.thresholds:
            axis = self.thresholds.get('turn_left_axis', 'gy')
            gyro_data = gy if axis == 'gy' else gz
            gyro_peak = np.max(gyro_data)
            
            if gyro_peak > self.thresholds['turn_left_gyro_threshold'] * DETECTION_THRESHOLD:
                confidence = min(1.0, gyro_peak / self.thresholds['turn_left_gyro_threshold'])
                return 'turn_left', confidence
        
        # Check for turn right  
        if 'turn_right_gyro_threshold' in self.thresholds:
            axis = self.thresholds.get('turn_right_axis', 'gy')
            gyro_data = gy if axis == 'gy' else gz
            gyro_valley = np.min(gyro_data)
            
            if gyro_valley < self.thresholds['turn_right_gyro_threshold'] * DETECTION_THRESHOLD:
                confidence = min(1.0, abs(gyro_valley) / abs(self.thresholds['turn_right_gyro_threshold']))
                return 'turn_right', confidence
        
        return None, 0.0

class MinecraftController:
    """Minecraft keyboard controller"""
    
    def __init__(self):
        self.keyboard_controller = keyboard.Controller()
        self.last_action_time = 0.0
        self.action_counts = {"mine": 0, "place": 0, "look_left": 0, "look_right": 0, "idle": 0}
    
    def execute_gesture(self, gesture_name: str, confidence: float):
        """Execute Minecraft action based on detected gesture"""
        current_time = time.time()
        
        # Enforce cooldown period
        if current_time - self.last_action_time < COOLDOWN_TIME:
            self.action_counts["idle"] += 1
            return
        
        if gesture_name == "swing_down":
            # Left click for mining
            self.keyboard_controller.click(keyboard.Button.left)
            print(f"⛏️  MINING: swing_down (confidence: {confidence:.2f})")
            self.action_counts["mine"] += 1
            self.last_action_time = current_time
            
        elif gesture_name == "arm_up":
            # Right click for placing blocks
            self.keyboard_controller.click(keyboard.Button.right)
            print(f"🧱 PLACING: arm_up (confidence: {confidence:.2f})")
            self.action_counts["place"] += 1
            self.last_action_time = current_time
            
        elif gesture_name == "turn_left":
            # Left arrow key for camera
            self.keyboard_controller.press(keyboard.Key.left)
            self.keyboard_controller.release(keyboard.Key.left)
            print(f"👈 LOOK LEFT: turn_left (confidence: {confidence:.2f})")
            self.action_counts["look_left"] += 1
            self.last_action_time = current_time
            
        elif gesture_name == "turn_right":
            # Right arrow key for camera
            self.keyboard_controller.press(keyboard.Key.right)
            self.keyboard_controller.release(keyboard.Key.right)
            print(f"👉 LOOK RIGHT: turn_right (confidence: {confidence:.2f})")
            self.action_counts["look_right"] += 1
            self.last_action_time = current_time
            
        else:
            self.action_counts["idle"] += 1
    
    def print_stats(self):
        """Print action statistics"""
        total = sum(self.action_counts.values())
        if total > 0:
            mine_pct = self.action_counts["mine"] / total * 100
            place_pct = self.action_counts["place"] / total * 100
            left_pct = self.action_counts["look_left"] / total * 100
            right_pct = self.action_counts["look_right"] / total * 100
            idle_pct = self.action_counts["idle"] / total * 100
            print(f"📊 Actions: Mine {mine_pct:.0f}% | Place {place_pct:.0f}% | Left {left_pct:.0f}% | Right {right_pct:.0f}% | Idle {idle_pct:.0f}%")

def main():
    parser = argparse.ArgumentParser(description="Threshold-based MindRove → Minecraft controller")
    parser.add_argument("--thresholds", required=True, help="Calibrated thresholds JSON file")
    parser.add_argument("--ip", help="MindRove IP address")
    parser.add_argument("--port", type=int, help="MindRove port")
    parser.add_argument("--debug", action="store_true", help="Show detection details")
    args = parser.parse_args()
    
    # Setup signal handler for clean exit
    def signal_handler(signum, frame):
        print("\n🛑 Stopping threshold controller...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🎮 Starting Threshold-based MindRove → Minecraft Controller")
    print("=" * 50)
    
    try:
        # Initialize gesture detector
        detector = ThresholdGestureDetector(args.thresholds)
        controller = MinecraftController()
        
        # Setup MindRove connection
        params = MindRoveInputParams()
        if args.ip:
            params.ip_address = args.ip
        else:
            params.ip_address = "192.168.4.1"
        
        if args.port:
            params.ip_port = args.port
        else:
            params.ip_port = 4210
            
        params.ip_protocol = IpProtocolTypes.TCP.value
        
        board = BoardShim(BOARD_ID, params)
        board.prepare_session()
        board.start_stream(45000, "")  # 45 second buffer
        
        print(f"🔗 Connected to MindRove at {params.ip_address}:{params.ip_port}")
        print("🎯 Threshold-based gesture detection active!")
        print("   ⛏️  Swing down → Mining (left click)")
        print("   🧱 Arm up → Place block (right click)")
        print("   👈 Turn left → Left arrow")  
        print("   👉 Turn right → Right arrow")
        print("Press Ctrl+C to stop\n")
        
        # Get channel indices
        accel_idx = board.get_accel_channels(BOARD_ID)[:3]  # X, Y, Z
        gyro_idx = board.get_gyro_channels(BOARD_ID)[:3]    # X, Y, Z
        
        samples_processed = 0
        detections_made = 0
        last_stats_time = time.time()
        
        # Main detection loop
        while True:
            data_count = board.get_board_data_count(PRESET)
            
            if data_count > 0:
                data = board.get_current_board_data(data_count, PRESET)
                
                # Process each sample
                for i in range(data.shape[1]):
                    accel_sample = data[accel_idx, i]
                    gyro_sample = data[gyro_idx, i]
                    
                    # Add to detector buffers
                    detector.add_sample(accel_sample, gyro_sample)
                    samples_processed += 1
                    
                    # Detect gesture
                    gesture, confidence = detector.detect_gesture()
                    
                    if gesture and confidence >= MIN_CONFIDENCE:
                        controller.execute_gesture(gesture, confidence)
                        detections_made += 1
                        
                        if args.debug:
                            print(f"  📊 Sample {samples_processed}: {gesture} ({confidence:.2f})")
            
            # Print periodic stats
            current_time = time.time()
            if current_time - last_stats_time > 10.0:  # Every 10 seconds
                controller.print_stats()
                print(f"  📊 Processed: {samples_processed} samples, {detections_made} detections")
                last_stats_time = current_time
            
            time.sleep(0.001)  # Small delay to prevent CPU spinning
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping threshold controller...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except:
            pass
        
        # Final statistics
        total_actions = sum(controller.action_counts.values())
        print(f"\n📊 Final Statistics:")
        print(f"   Samples processed: {samples_processed}")
        print(f"   Detections made: {detections_made}")  
        print(f"   Total actions: {total_actions}")
        if total_actions > 0:
            print(f"   Mining: {controller.action_counts['mine']}")
            print(f"   Placing: {controller.action_counts['place']}")
            print(f"   Look Left: {controller.action_counts['look_left']}")
            print(f"   Look Right: {controller.action_counts['look_right']}")
            print(f"   Idle: {controller.action_counts['idle']}")

if __name__ == '__main__':
    main()