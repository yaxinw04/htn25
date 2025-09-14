#!/usr/bin/env python3
"""
ML Training Data Collection for MindRove Gestures
=================================================

Collects training data with the improved methodology:
1. 2-second intervals for each gesture
2. 3 samples per gesture class
3. Rest periods between samples to return to neutral
4. Combined IMU + EMG features
5. Saves data for ML training (SVM, XGBoost, etc.)

Usage: python collect_ml_training_data.py --output training_data.csv
"""

import argparse
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
from mindrove.data_filter import DataFilter, FilterTypes

# Configuration
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET = MindRovePresets.DEFAULT_PRESET.value

# Data collection parameters
SAMPLE_DURATION = 1.0      # 1 second per sample
REST_DURATION = 2.0        # 3 seconds rest between individual samples  
SAMPLES_PER_GESTURE = 20   # 10 samples per gesture class (total 50 samples)
BATCH_REST_DURATION = 2.0 # 10 seconds rest after every 10 actions
SAMPLE_RATE = 500          # Hz

# Gesture classes to collect (updated for actual Minecraft use case)
GESTURE_CLASSES = [
    ("idle", "Rest position - no movement"),
    ("swing_down", "Swing down motion (mining)"),
    ("arm_up", "Lift arm up (placing blocks)"),
    ("wrist_flex_left", "Wrist flexion for looking left"),
    ("wrist_supinate_right", "Wrist supination for looking right")
]

# Signal processing
ACCEL_LP_CUTOFF = 5.0
GYRO_LP_CUTOFF = 8.0
EMG_BP_LOW = 20.0
EMG_BP_HIGH = 450.0
EMG_ENV_CUTOFF = 8.0

AXIS_MAP = dict(ax=(0, +1), ay=(1, +1), az=(2, +1),
                gx=(0, +1), gy=(1, +1), gz=(2, +1))

@dataclass
class MLFeatures:
    """Comprehensive feature set for ML training"""
    # Statistical features for IMU (6 channels: ax, ay, az, gx, gy, gz)
    imu_mean: List[float]           # Mean values (6 features)
    imu_std: List[float]            # Standard deviation (6 features)
    imu_min: List[float]            # Minimum values (6 features)
    imu_max: List[float]            # Maximum values (6 features)
    imu_peak_to_peak: List[float]   # Max - Min (6 features)
    imu_rms: List[float]            # RMS values (6 features)
    
    # EMG statistical features (8 channels)
    emg_mean: List[float]           # Mean EMG envelope (8 features)
    emg_std: List[float]            # EMG variation (8 features)
    emg_max: List[float]            # Peak EMG (8 features)
    emg_integrated: List[float]     # Integrated EMG (8 features)
    
    # Cross-channel features
    accel_magnitude_stats: List[float]  # [mean, std, max, min] of |acceleration|
    gyro_magnitude_stats: List[float]   # [mean, std, max, min] of |angular_velocity|
    emg_total_activation: float         # Sum of all EMG channels
    
    # Temporal features
    motion_onset_time: float            # When significant motion started
    peak_motion_time: float             # When peak motion occurred
    gesture_duration: float             # Duration of active motion
    
    # Spectral features (simplified)
    dominant_freq_accel: List[float]    # Dominant frequency per accel axis (3 features)
    dominant_freq_gyro: List[float]     # Dominant frequency per gyro axis (3 features)
    
    # Label
    gesture_label: str                  # Ground truth label
    gesture_id: int                     # Numeric label (0=idle, 1=swing_down, etc.)

class MLDataCollector:
    """Collects RAW time-series training data with proper rest periods"""
    
    def __init__(self):
        self.raw_samples = []  # Store RAW time-series data instead of features
        self.sampling_rate = SAMPLE_RATE
        
    def collect_sample(self, board: BoardShim, accel_idx: List[int], 
                      gyro_idx: List[int], emg_idx: List[int],
                      gesture_name: str, gesture_id: int, sample_num: int) -> bool:
        """Collect RAW time-series data for exactly 1 second during gesture"""
        
        print(f"\n🎯 Collecting {gesture_name} (sample {sample_num + 1}/10)")
        print("Get ready...")
        
        # 3-second countdown - CLEAR BUFFER during countdown
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
            # Clear buffer during countdown to avoid old data
            data_count = board.get_board_data_count(PRESET)
            if data_count > 0:
                board.get_current_board_data(data_count, PRESET)
            
        print("🔴 RECORDING - Perform the gesture now!")
        
        # Collect RAW data for exactly SAMPLE_DURATION seconds with proper pacing
        raw_data_points = []
        start_time = time.time()
        end_time = start_time + SAMPLE_DURATION
        max_points = int(SAMPLE_DURATION * self.sampling_rate)  # Target ~500 points
        
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Only collect if we haven't hit our target rate
            target_points_so_far = int((elapsed / SAMPLE_DURATION) * max_points)
            
            if len(raw_data_points) < target_points_so_far:
                data_count = board.get_board_data_count(PRESET)
                if data_count > 0:
                    data = board.get_current_board_data(data_count, PRESET)
                    
                    # Take only the most recent data point to avoid buffer dumps
                    if data_count > 0:
                        i = data_count - 1  # Most recent sample
                        
                        # Store RAW sensor readings with timestamp
                        timestamp = elapsed
                        
                        # Extract raw sensor values
                        accel_x, accel_y, accel_z = data[accel_idx, i] if len(accel_idx) >= 3 else [0, 0, 0]
                        gyro_x, gyro_y, gyro_z = data[gyro_idx, i] if len(gyro_idx) >= 3 else [0, 0, 0]
                        
                        # Extract EMG channels
                        emg_values = data[emg_idx, i] if len(emg_idx) > 0 else []
                        emg_ch = [emg_values[j] if j < len(emg_values) else 0.0 for j in range(8)]
                        
                        # Create raw data point
                        data_point = {
                            'timestamp': timestamp,
                            'gesture_id': gesture_id,
                            'gesture_name': gesture_name,
                            'sample_id': sample_num,  # Simplified sample_id
                            'accel_x': float(accel_x),
                            'accel_y': float(accel_y), 
                            'accel_z': float(accel_z),
                            'gyro_x': float(gyro_x),
                            'gyro_y': float(gyro_y),
                            'gyro_z': float(gyro_z),
                            'emg_0': float(emg_ch[0]),
                            'emg_1': float(emg_ch[1]),
                            'emg_2': float(emg_ch[2]),
                            'emg_3': float(emg_ch[3]),
                            'emg_4': float(emg_ch[4]),
                            'emg_5': float(emg_ch[5]),
                            'emg_6': float(emg_ch[6]),
                            'emg_7': float(emg_ch[7])
                        }
                        
                        raw_data_points.append(data_point)
            
            # Small delay to pace the data collection properly
            time.sleep(0.01)  # 10ms delay to spread collection over 1 second
        
        elapsed_time = time.time() - start_time
        print(f"✅ Sample collected ({len(raw_data_points)} time points in {elapsed_time:.2f}s)")
        
        # Store this sample's raw data points
        self.raw_samples.extend(raw_data_points)
        
        return True
        
    def save_training_data(self, output_file: str):
        """Save RAW time-series data for ML training and augmentation"""
        if not self.raw_samples:
            print("❌ No data collected!")
            return
            
        print(f"\n💾 Saving RAW time-series training data to {output_file}")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(self.raw_samples)
        
        # Ensure CSV extension
        csv_file = output_file.replace('.npz', '.csv') if output_file.endswith('.npz') else output_file
        if not csv_file.endswith('.csv'):
            csv_file += '.csv'
            
        # Save as CSV
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Saved {len(df)} time points from {df.groupby(['gesture_name', 'sample_id']).ngroups} samples")
        print(f"📊 Data shape: {df.shape}")
        
        # Print sample distribution
        sample_counts = df.groupby('gesture_name')['sample_id'].nunique()
        print("� Sample distribution:")
        for gesture_name, count in sample_counts.items():
            avg_points = len(df[df['gesture_name'] == gesture_name]) / count
            print(f"   {gesture_name}: {count} samples (~{avg_points:.0f} points each)")
            
        print(f"\n💡 Ready for windowed ML training with sliding windows!")
        
    def _get_gesture_name(self, gesture_id: int) -> str:
        """Get gesture name from ID"""
        gesture_map = {0: "idle", 1: "swing_down", 2: "arm_up", 3: "wrist_flex_left", 4: "wrist_supinate_right"}
        return gesture_map.get(gesture_id, f"unknown_{gesture_id}")

def main():
    parser = argparse.ArgumentParser(description="Collect ML training data for MindRove gestures")
    parser.add_argument("--output", default="training_data.csv", help="Output file for training data (CSV format)")
    parser.add_argument("--ip", help="MindRove IP address")
    parser.add_argument("--port", type=int, help="MindRove port")
    parser.add_argument("--gestures", help="Comma-separated list of gestures to collect (default: all)")
    args = parser.parse_args()

    # Setup gesture list
    gestures_to_collect = GESTURE_CLASSES
    if args.gestures:
        requested = args.gestures.split(',')
        gestures_to_collect = [(name, desc) for name, desc in GESTURE_CLASSES if name in requested]
    
    print("🤖 ML Training Data Collection for MindRove")
    print("=" * 50)
    print(f"📊 Collecting {SAMPLES_PER_GESTURE} samples per gesture")
    print(f"⏱️  {SAMPLE_DURATION}s per sample, {REST_DURATION}s rest between samples")
    print(f"🎯 Gestures: {', '.join([name for name, _ in gestures_to_collect])}")
    print(f"💾 Output: {args.output}")
    print()

    # Initialize data collector
    collector = MLDataCollector()

    # Setup MindRove
    BoardShim.enable_board_logger()
    params = MindRoveInputParams()
    if args.ip: 
        params.ip_address = args.ip
    if args.port: 
        params.ip_port = args.port

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    
    # Get channel info
    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID, PRESET)
    accel_idx = BoardShim.get_accel_channels(BOARD_ID, PRESET)
    gyro_idx = BoardShim.get_gyro_channels(BOARD_ID, PRESET)
    emg_idx = BoardShim.get_emg_channels(BOARD_ID, PRESET)
    
    collector.sampling_rate = sampling_rate
    
    print(f"📡 Streaming at {sampling_rate} Hz")
    print(f"📊 Channels: {len(accel_idx)} accel, {len(gyro_idx)} gyro, {len(emg_idx)} EMG")
    
    board.start_stream(450000)
    
    try:
        # Create randomized collection schedule
        collection_schedule = []
        for gesture_id, (gesture_name, description) in enumerate(gestures_to_collect):
            for sample_num in range(SAMPLES_PER_GESTURE):
                collection_schedule.append({
                    'gesture_id': gesture_id,
                    'gesture_name': gesture_name,
                    'description': description,
                    'sample_num': sample_num
                })
        
        # Randomize the collection order
        import random
        random.shuffle(collection_schedule)
        
        total_samples = len(collection_schedule)
        print(f"\n🎲 Randomized collection order created!")
        print(f"📊 Total samples to collect: {total_samples}")
        print(f"⏱️  {SAMPLE_DURATION}s per sample, {REST_DURATION}s rest between samples")
        print(f"💤 {BATCH_REST_DURATION}s rest after every 10 samples")
        
        input("\nPress Enter when ready to start randomized collection...")
        
        # Collect samples in randomized order
        for idx, item in enumerate(collection_schedule):
            current_sample = idx + 1
            gesture_name = item['gesture_name']
            gesture_id = item['gesture_id']
            description = item['description']
            sample_num = item['sample_num']
            
            print(f"\n{'='*50}")
            print(f"🎯 Sample {current_sample}/{total_samples}: {gesture_name.upper()}")
            print(f"📝 {description}")
            print(f"🔄 This is sample #{sample_num + 1}/10 for this gesture")
            
            success = collector.collect_sample(board, accel_idx, gyro_idx, emg_idx,
                                             gesture_name, gesture_id, sample_num)
            if not success:
                print("❌ Sample collection failed")
                continue
            
            # Rest period logic
            if current_sample < total_samples:  # Not the last sample
                if current_sample % 10 == 0:  # Every 10 samples
                    print(f"� BATCH REST: {BATCH_REST_DURATION} seconds after 10 samples...")
                    time.sleep(BATCH_REST_DURATION)
                else:  # Regular rest
                    print(f"😴 Rest for {REST_DURATION} seconds...")
                    time.sleep(REST_DURATION)
        
        # Save collected data
        collector.save_training_data(args.output)
        
        # Print collection summary
        print(f"\n🎉 Data collection complete!")
        gesture_counts = {}
        for item in collection_schedule:
            name = item['gesture_name']
            gesture_counts[name] = gesture_counts.get(name, 0) + 1
        
        print(f"📊 Collection Summary:")
        for gesture_name, count in gesture_counts.items():
            print(f"   {gesture_name}: {count} samples")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Train ML model: python train_ml_classifier.py --data {args.output}")
        print(f"   2. Test classifier: python test_ml_classifier.py --model trained_model.pkl")
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted by user")
    finally:
        try:
            board.stop_stream()
        except:
            pass
        board.release_session()

if __name__ == '__main__':
    main()