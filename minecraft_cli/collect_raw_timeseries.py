#!/usr/bin/env python3
"""
RAW Time-Series Data Collection for MindRove Gestures
====================================================

Collects RAW time-series data for proper ML training with windowing and augmentation.
Saves actual sensor readings over time, not statistical features.

Usage: python collect_raw_timeseries.py --output raw_training_data.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
from typing import List, Dict
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets

# Configuration
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET = MindRovePresets.DEFAULT_PRESET.value

# Data collection parameters
SAMPLE_DURATION = 1.0      # 1 second per sample
REST_DURATION = 3.0        # 3 seconds rest between samples  
SAMPLES_PER_GESTURE = 10   # 10 samples per gesture (50 total)
BATCH_REST_DURATION = 10.0 # 10 seconds rest after every 10 actions
SAMPLE_RATE = 500          # Hz

# Gesture classes
GESTURE_CLASSES = [
    ("idle", "Rest position - no movement"),
    ("swing_down", "Swing down motion (mining)"),
    ("arm_up", "Lift arm up (placing blocks)"),
    ("wrist_flex_left", "Wrist flexion for looking left"),
    ("wrist_supinate_right", "Wrist supination for looking right")
]

class RawDataCollector:
    """Collects RAW time-series data for ML training"""
    
    def __init__(self):
        self.raw_samples = []  # Store all raw time-series samples
        self.sampling_rate = SAMPLE_RATE
        
    def collect_sample(self, board: BoardShim, accel_idx: List[int], 
                      gyro_idx: List[int], emg_idx: List[int],
                      gesture_name: str, gesture_id: int, sample_num: int) -> bool:
        """Collect RAW time-series data for 1 second"""
        
        print(f"\n🎯 Collecting {gesture_name} (sample {sample_num + 1}/10)")
        print("Get ready...")
        
        # 3-second countdown
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
            
        print("🔴 RECORDING - Perform the gesture now!")
        
        # Collect RAW data for exactly SAMPLE_DURATION seconds
        raw_data = []
        start_time = time.time()
        end_time = start_time + SAMPLE_DURATION
        
        while time.time() < end_time:
            data_count = board.get_board_data_count(PRESET)
            if data_count > 0:
                data = board.get_current_board_data(data_count, PRESET)
                
                for i in range(data_count):
                    timestamp = time.time() - start_time
                    
                    # Extract raw sensor values
                    accel_x, accel_y, accel_z = data[accel_idx, i] if len(accel_idx) >= 3 else [0, 0, 0]
                    gyro_x, gyro_y, gyro_z = data[gyro_idx, i] if len(gyro_idx) >= 3 else [0, 0, 0]
                    
                    # Extract EMG channels (up to 8)
                    emg_values = data[emg_idx, i] if len(emg_idx) > 0 else []
                    emg_ch = [emg_values[j] if j < len(emg_values) else 0.0 for j in range(8)]
                    
                    # Create time-series row
                    sample_row = {
                        'timestamp': timestamp,
                        'gesture_id': gesture_id,
                        'gesture_name': gesture_name,
                        'sample_id': len(self.raw_samples),  # Unique sample identifier
                        'sample_num': sample_num,
                        'accel_x': accel_x,
                        'accel_y': accel_y,
                        'accel_z': accel_z,
                        'gyro_x': gyro_x,
                        'gyro_y': gyro_y,
                        'gyro_z': gyro_z,
                        'emg_0': emg_ch[0],
                        'emg_1': emg_ch[1],
                        'emg_2': emg_ch[2],
                        'emg_3': emg_ch[3],
                        'emg_4': emg_ch[4],
                        'emg_5': emg_ch[5],
                        'emg_6': emg_ch[6],
                        'emg_7': emg_ch[7]
                    }
                    
                    raw_data.append(sample_row)
                    
            time.sleep(0.002)  # Small delay
            
        elapsed_time = time.time() - start_time
        print(f"✅ Sample collected ({len(raw_data)} time points in {elapsed_time:.2f}s)")
        
        # Store this sample's raw data
        self.raw_samples.extend(raw_data)
        
        return True
        
    def save_raw_data(self, output_file: str):
        """Save RAW time-series data as CSV"""
        if not self.raw_samples:
            print("❌ No data collected!")
            return
            
        print(f"\n💾 Saving RAW time-series data to {output_file}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.raw_samples)
        
        # Ensure CSV extension
        csv_file = output_file.replace('.npz', '.csv') if output_file.endswith('.npz') else output_file
        if not csv_file.endswith('.csv'):
            csv_file += '.csv'
            
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Saved {len(df)} time points from {len(df['sample_id'].unique())} samples")
        print(f"📊 Data shape: {df.shape}")
        print(f"⏱️  Average points per sample: {len(df) / len(df['sample_id'].unique()):.1f}")
        
        # Print class distribution
        gesture_counts = df.groupby(['gesture_name', 'sample_id']).size().groupby('gesture_name').count()
        print("📈 Sample distribution:")
        for gesture_name, count in gesture_counts.items():
            print(f"   {gesture_name}: {count} samples")
            
        print(f"\n💡 Next steps for windowed ML training:")
        print(f"   1. Use sliding windows (e.g., 50-100 time points)")
        print(f"   2. Data augmentation (time-warping, noise, etc.)")
        print(f"   3. Train with windowed features or RNN/CNN")

def main():
    parser = argparse.ArgumentParser(description="Collect RAW time-series data for MindRove gestures")
    parser.add_argument("--output", default="raw_training_data.csv", help="Output CSV file")
    parser.add_argument("--ip", help="MindRove IP address")
    parser.add_argument("--port", type=int, help="MindRove port")
    args = parser.parse_args()

    print("🤖 RAW Time-Series Data Collection for MindRove")
    print("=" * 50)
    print(f"📊 Collecting {SAMPLES_PER_GESTURE} samples per gesture")
    print(f"⏱️  {SAMPLE_DURATION}s per sample, {REST_DURATION}s rest")
    print(f"🎯 Gestures: {', '.join([name for name, _ in GESTURE_CLASSES])}")
    print(f"💾 Output: {args.output}")
    print(f"🔬 This saves RAW sensor data for windowing & augmentation!")
    print()

    # Initialize collector
    collector = RawDataCollector()

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
        for gesture_id, (gesture_name, description) in enumerate(GESTURE_CLASSES):
            for sample_num in range(SAMPLES_PER_GESTURE):
                collection_schedule.append({
                    'gesture_id': gesture_id,
                    'gesture_name': gesture_name,
                    'description': description,
                    'sample_num': sample_num
                })
        
        # Randomize order
        import random
        random.shuffle(collection_schedule)
        
        total_samples = len(collection_schedule)
        print(f"\n🎲 Randomized collection order created!")
        print(f"📊 Total samples: {total_samples} (will create ~{total_samples * 500} time points)")
        
        input("\nPress Enter when ready to start collection...")
        
        # Collect samples
        for idx, item in enumerate(collection_schedule):
            current_sample = idx + 1
            
            print(f"\n{'='*50}")
            print(f"🎯 Sample {current_sample}/{total_samples}: {item['gesture_name'].upper()}")
            print(f"📝 {item['description']}")
            print(f"🔄 This is sample #{item['sample_num'] + 1}/10 for this gesture")
            
            success = collector.collect_sample(board, accel_idx, gyro_idx, emg_idx,
                                             item['gesture_name'], item['gesture_id'], item['sample_num'])
            
            if not success:
                print("❌ Sample collection failed")
                continue
            
            # Rest logic
            if current_sample < total_samples:
                if current_sample % 10 == 0:
                    print(f"💤 BATCH REST: {BATCH_REST_DURATION} seconds...")
                    time.sleep(BATCH_REST_DURATION)
                else:
                    print(f"😴 Rest for {REST_DURATION} seconds...")
                    time.sleep(REST_DURATION)
        
        # Save data
        collector.save_raw_data(args.output)
        
        print(f"\n🎉 RAW data collection complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted")
    finally:
        try:
            board.stop_stream()
        except:
            pass
        board.release_session()

if __name__ == '__main__':
    main()