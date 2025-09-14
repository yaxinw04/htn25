#!/usr/bin/env python3
"""
Improved Calibration with 2-Second Bursts and Rest Periods
==========================================================

Collects calibration data using the improved methodology:
1. 2-second intervals for each gesture
2. 3 samples per gesture class  
3. 3-second rest periods between samples to return to neutral
4. Peak-based threshold calculation instead of averaging

Usage: python improved_calibrate.py --outfile improved_thresholds.json --ip 192.168.4.1
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets, IpProtocolTypes
from mindrove.data_filter import DataFilter, FilterTypes
from mindrove.exit_codes import MindRoveError

# Configuration
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET = MindRovePresets.DEFAULT_PRESET.value

# Calibration parameters
SAMPLE_DURATION = 1.0       # 1 second per sample
REST_DURATION = 2.0         # 2 seconds rest between samples
SAMPLES_PER_GESTURE = 3     # 3 samples per gesture
SAMPLE_RATE = 500           # Hz

# Signal processing
LOWPASS_ACCEL_CUTOFF = 5.0
LOWPASS_GYRO_CUTOFF = 8.0

# Gestures to calibrate
GESTURES = [
    ("neutral", "Stay relaxed and still"),
    ("swing_down", "Clear downward swing motion (like mining)"),
    ("arm_up", "Raise arm up (like placing blocks)"),
    ("turn_left", "Rotate wrist/hand left for camera turn"),
    ("turn_right", "Rotate wrist/hand right for camera turn")
]

AXIS_MAP = dict(ax=(0, +1), ay=(1, +1), az=(2, +1),
                gx=(0, +1), gy=(1, +1), gz=(2, +1))

class ImprovedCalibrator:
    """Calibrates thresholds using 2-second bursts and peak detection"""
    
    def __init__(self):
        self.calibration_data = defaultdict(list)
        self.sample_rate = SAMPLE_RATE
        
    def collect_gesture_samples(self, board: BoardShim, accel_idx: list, gyro_idx: list,
                              gesture_name: str, gesture_desc: str) -> bool:
        """Collect 3 samples for a gesture with rest periods"""
        
        print(f"\n{'='*60}")
        print(f"🎯 Calibrating: {gesture_name.upper()}")
        print(f"📝 Description: {gesture_desc}")
        print(f"🔄 Will collect {SAMPLES_PER_GESTURE} samples with rest periods")
        
        input("Press Enter when ready to start...")
        
        for sample_num in range(SAMPLES_PER_GESTURE):
            print(f"\n🎯 Sample {sample_num + 1}/{SAMPLES_PER_GESTURE}")
            
            # Countdown
            print("Get ready...")
            for i in range(3, 0, -1):
                print(f"  {i}...")
                time.sleep(1)
            
            print(f"🔴 RECORDING - {gesture_desc}")
            
            # Collect 2-second sample
            success = self._collect_single_sample(board, accel_idx, gyro_idx, 
                                                gesture_name, sample_num)
            if not success:
                print("❌ Sample collection failed")
                return False
                
            print(f"✅ Sample {sample_num + 1} collected")
            
            # Rest period (except after last sample)
            if sample_num < SAMPLES_PER_GESTURE - 1:
                print(f"😴 Rest for {REST_DURATION} seconds (return to neutral)...")
                time.sleep(REST_DURATION)
        
        print(f"✅ Completed {gesture_name} calibration")
        return True
        
    def _collect_single_sample(self, board: BoardShim, accel_idx: list, gyro_idx: list,
                             gesture_name: str, sample_num: int) -> bool:
        """Collect a single 2-second sample"""
        
        collected_data = []
        start_time = time.time()
        
        # Collect data for EXACTLY the sample duration (2 seconds)
        while time.time() - start_time < SAMPLE_DURATION:
            data_count = board.get_board_data_count(PRESET)
            if data_count > 0:
                data = board.get_current_board_data(data_count, PRESET)
                
                for i in range(data.shape[1]):  # Use shape[1] for number of samples
                    # Extract IMU data
                    accel_sample = data[accel_idx, i]
                    gyro_sample = data[gyro_idx, i]
                    imu_sample = np.concatenate([accel_sample, gyro_sample])
                    collected_data.append(imu_sample)
            
            time.sleep(0.01)  # Small delay to prevent CPU spinning
            
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Process collected data
        imu_array = np.array(collected_data)  # Shape: (samples, 6)
        features = self._extract_peak_features(imu_array)
        
        # Store features
        self.calibration_data[gesture_name].append(features)
        
        print(f"   📊 Collected {len(collected_data)} samples in {actual_duration:.2f}s (target: {SAMPLE_DURATION:.1f}s)")
        return True
        
    def _extract_peak_features(self, imu_data: np.ndarray) -> dict:
        """Extract peak-based features from 2-second sample"""
        
        # Process data (filtering and axis remapping)
        processed_data = self._process_imu_data(imu_data)
        
        ax, ay, az = processed_data[:, 0], processed_data[:, 1], processed_data[:, 2]
        gx, gy, gz = processed_data[:, 3], processed_data[:, 4], processed_data[:, 5]
        
        # Extract PEAK features (not averages!)
        features = {
            # Peak values
            'ax_peak': float(np.max(ax)),
            'ay_peak': float(np.max(ay)),
            'az_peak': float(np.max(az)),
            
            # Valley values (most negative)
            'ax_valley': float(np.min(ax)),
            'ay_valley': float(np.min(ay)), 
            'az_valley': float(np.min(az)),
            
            # Gyro peak and valley values (signed)
            'gx_peak': float(np.max(gx)),
            'gy_peak': float(np.max(gy)),
            'gz_peak': float(np.max(gz)),
            'gx_valley': float(np.min(gx)),
            'gy_valley': float(np.min(gy)),
            'gz_valley': float(np.min(gz)),
            
            # Peak absolute gyro
            'gx_peak_abs': float(np.max(np.abs(gx))),
            'gy_peak_abs': float(np.max(np.abs(gy))),
            'gz_peak_abs': float(np.max(np.abs(gz))),
            
            # Motion intensities
            'swing_intensity': float(max(
                np.max(ax) - np.min(ax),  # X swing
                np.max(az) - np.min(az)   # Z swing (up/down)
            )),
            
            'rotation_intensity': float(max(
                np.max(np.abs(gx)),
                np.max(np.abs(gy)),
                np.max(np.abs(gz))
            )),
            
            # Total motion
            'total_motion': float(np.max(np.sqrt(ax**2 + ay**2 + az**2))),
            
            # Motion duration
            'motion_duration': self._calculate_motion_duration(ax, ay, az)
        }
        
        return features
        
    def _calculate_motion_duration(self, ax, ay, az) -> float:
        """Calculate how long significant motion lasted"""
        motion_mag = np.sqrt(ax**2 + ay**2 + az**2)
        threshold = np.mean(motion_mag) + np.std(motion_mag)
        above_threshold = motion_mag > threshold
        return float(np.sum(above_threshold) / self.sample_rate)
        
    def _process_imu_data(self, imu_data: np.ndarray) -> np.ndarray:
        """Process IMU data with filtering and axis remapping"""
        # Separate accel and gyro
        accel_data = imu_data[:, :3].T  # Shape: (3, samples)
        gyro_data = imu_data[:, 3:6].T  # Shape: (3, samples)
        
        # Remap axes
        accel_remapped = self._remap_axes(accel_data, "accel")
        gyro_remapped = self._remap_axes(gyro_data, "gyro")
        
        # Apply filtering
        self._filter_signal(accel_remapped, LOWPASS_ACCEL_CUTOFF)
        self._filter_signal(gyro_remapped, LOWPASS_GYRO_CUTOFF)
        
        # Recombine and transpose
        processed = np.vstack([accel_remapped, gyro_remapped]).T  # Shape: (samples, 6)
        return processed
    
    def _remap_axes(self, arr3xN: np.ndarray, which: str) -> np.ndarray:
        """Apply axis remapping"""
        assert arr3xN.shape[0] == 3
        if which == "accel":
            x_idx, x_sign = AXIS_MAP['ax']; y_idx, y_sign = AXIS_MAP['ay']; z_idx, z_sign = AXIS_MAP['az']
        else:
            x_idx, x_sign = AXIS_MAP['gx']; y_idx, y_sign = AXIS_MAP['gy']; z_idx, z_sign = AXIS_MAP['gz']
        return np.vstack([
            x_sign * arr3xN[x_idx],
            y_sign * arr3xN[y_idx],
            z_sign * arr3xN[z_idx],
        ])
    
    def _filter_signal(self, signal: np.ndarray, cutoff_freq: float):
        """Apply low-pass filter in place"""
        for i in range(signal.shape[0]):
            DataFilter.perform_lowpass(signal[i], self.sample_rate, cutoff_freq,
                                     3, FilterTypes.BUTTERWORTH.value, 0.0)
    
    def calculate_improved_thresholds(self) -> dict:
        """Calculate thresholds from collected peak-based features"""
        
        print(f"\n📊 Calculating improved thresholds from peak features...")
        
        thresholds = {}
        
        # Get neutral baseline
        neutral_features = self.calibration_data.get('neutral', [])
        if not neutral_features:
            print("⚠️  No neutral data collected, using default values")
            neutral_baseline = {'total_motion': 0.2, 'swing_intensity': 0.1}
        else:
            neutral_baseline = {
                'total_motion': max([f['total_motion'] for f in neutral_features]),
                'swing_intensity': max([f['swing_intensity'] for f in neutral_features])
            }
        
        print(f"📊 Neutral baseline: motion={neutral_baseline['total_motion']:.2f}, "
              f"swing={neutral_baseline['swing_intensity']:.2f}")
        
        # Swing down thresholds (mining)
        swing_features = self.calibration_data.get('swing_down', [])
        if swing_features:
            # Use peak values, not averages!
            az_valleys = [f['az_valley'] for f in swing_features]
            swing_intensities = [f['swing_intensity'] for f in swing_features]
            rotation_intensities = [f['rotation_intensity'] for f in swing_features]
            total_motions = [f['total_motion'] for f in swing_features]
            
            # Set thresholds based on minimum values from samples (ensure we capture all)
            thresholds['swing_az_valley_threshold'] = min(az_valleys) * 0.8  # 80% of minimum
            thresholds['swing_intensity_threshold'] = min(swing_intensities) * 0.7
            thresholds['swing_rotation_threshold'] = min(rotation_intensities) * 0.7
            thresholds['swing_motion_threshold'] = min(total_motions) * 0.7
            
            print(f"⛏️  Swing down thresholds:")
            print(f"   az_valley < {thresholds['swing_az_valley_threshold']:.2f}")
            print(f"   swing_intensity > {thresholds['swing_intensity_threshold']:.2f}")
            print(f"   rotation > {thresholds['swing_rotation_threshold']:.1f}")
            print(f"   motion > {thresholds['swing_motion_threshold']:.2f}")
        
        # Arm up thresholds (placing)
        arm_features = self.calibration_data.get('arm_up', [])
        if arm_features:
            az_peaks = [f['az_peak'] for f in arm_features]
            swing_intensities = [f['swing_intensity'] for f in arm_features]
            total_motions = [f['total_motion'] for f in arm_features]
            
            thresholds['arm_az_peak_threshold'] = min(az_peaks) * 0.7
            thresholds['arm_swing_threshold'] = min(swing_intensities) * 0.6
            thresholds['arm_motion_threshold'] = min(total_motions) * 0.6
            
            print(f"🧱 Arm up thresholds:")
            print(f"   az_peak > {thresholds['arm_az_peak_threshold']:.2f}")
            print(f"   swing_intensity > {thresholds['arm_swing_threshold']:.2f}")
            print(f"   motion > {thresholds['arm_motion_threshold']:.2f}")
        
        # Turn left thresholds (using gyro_y or gyro_z for rotation)
        left_features = self.calibration_data.get('turn_left', [])
        if left_features:
            gy_peaks = [f['gy_peak'] for f in left_features]  # Positive Y gyro for left
            gy_valleys = [f['gy_valley'] for f in left_features]  # Negative Y gyro 
            gz_peaks = [f['gz_peak'] for f in left_features]  # Z gyro alternative
            gz_valleys = [f['gz_valley'] for f in left_features]
            
            # Use whichever axis shows stronger signal
            gy_range = max(gy_peaks) - min(gy_valleys) 
            gz_range = max(gz_peaks) - min(gz_valleys)
            
            if gy_range > gz_range:
                thresholds['turn_left_gyro_threshold'] = min(gy_peaks) * 0.7
                thresholds['turn_left_axis'] = 'gy'
                print(f"👈 Turn left thresholds (using gy):")
                print(f"   gy_peak > {thresholds['turn_left_gyro_threshold']:.1f}")
            else:
                thresholds['turn_left_gyro_threshold'] = min(gz_peaks) * 0.7
                thresholds['turn_left_axis'] = 'gz'
                print(f"👈 Turn left thresholds (using gz):")
                print(f"   gz_peak > {thresholds['turn_left_gyro_threshold']:.1f}")
        
        # Turn right thresholds
        right_features = self.calibration_data.get('turn_right', [])
        if right_features:
            gy_peaks = [f['gy_peak'] for f in right_features]
            gy_valleys = [f['gy_valley'] for f in right_features]
            gz_peaks = [f['gz_peak'] for f in right_features]  
            gz_valleys = [f['gz_valley'] for f in right_features]
            
            # Use whichever axis shows stronger signal 
            gy_range = max(gy_peaks) - min(gy_valleys)
            gz_range = max(gz_peaks) - min(gz_valleys)
            
            if gy_range > gz_range:
                thresholds['turn_right_gyro_threshold'] = max(gy_valleys) * 0.7  # Negative for right
                thresholds['turn_right_axis'] = 'gy'
                print(f"👉 Turn right thresholds (using gy):")
                print(f"   gy_valley < {thresholds['turn_right_gyro_threshold']:.1f}")
            else:
                thresholds['turn_right_gyro_threshold'] = max(gz_valleys) * 0.7
                thresholds['turn_right_axis'] = 'gz'
                print(f"👉 Turn right thresholds (using gz):")
                print(f"   gz_valley < {thresholds['turn_right_gyro_threshold']:.1f}")
        
        # General motion threshold (based on neutral + margin)
        thresholds['min_motion_threshold'] = neutral_baseline['total_motion'] + 0.2
        
        print(f"💤 General motion threshold: {thresholds['min_motion_threshold']:.2f}")
        
        return thresholds
    
    def save_thresholds(self, output_file: str):
        """Save calculated thresholds to file"""
        
        # Calculate thresholds
        thresholds = self.calculate_improved_thresholds()
        
        # Create output structure
        output_data = {
            'thresholds': thresholds,
            'calibration_info': {
                'method': 'improved_peak_based',
                'sample_duration': SAMPLE_DURATION,
                'samples_per_gesture': SAMPLES_PER_GESTURE,
                'rest_duration': REST_DURATION,
                'total_samples': sum(len(samples) for samples in self.calibration_data.values())
            },
            'raw_features': dict(self.calibration_data)  # Include raw data for analysis
        }
        
        # Save to file
        Path(output_file).write_text(json.dumps(output_data, indent=2))
        
        print(f"\n💾 Saved improved thresholds to {output_file}")
        print(f"📊 Total samples collected: {output_data['calibration_info']['total_samples']}")
        
        # Summary
        print(f"\n📋 Calibration Summary:")
        for gesture, samples in self.calibration_data.items():
            print(f"   {gesture}: {len(samples)} samples")

def main():
    parser = argparse.ArgumentParser(description="Improved calibration with 2-second bursts")
    parser.add_argument("--outfile", default="thresholds.json", 
                       help="Output file for thresholds (default: thresholds.json)")
    parser.add_argument("--ip", help="MindRove IP address")
    parser.add_argument("--port", type=int, help="MindRove port")
    parser.add_argument("--protocol", choices=["udp","tcp"], default="tcp", help="Socket protocol (default tcp)")
    parser.add_argument("--gestures", help="Comma-separated list of gestures (default: all)")
    parser.add_argument("--retries", type=int, default=5, help="Reconnect retries on socket errors")
    args = parser.parse_args()

    # Setup gesture list
    gestures_to_collect = GESTURES
    if args.gestures:
        requested = args.gestures.split(',')
        gestures_to_collect = [(name, desc) for name, desc in GESTURES if name in requested]

    print("🎯 Improved MindRove Calibration")
    print("=" * 50)
    print("🔍 Key improvement: Peak detection instead of averaging")
    print(f"⏱️  {SAMPLE_DURATION}s samples, {REST_DURATION}s rest periods")
    print(f"🔄 {SAMPLES_PER_GESTURE} samples per gesture")
    print(f"🎮 Gestures: {', '.join([name for name, _ in gestures_to_collect])}")
    print(f"💾 Output: {args.outfile}")

    # Initialize calibrator
    calibrator = ImprovedCalibrator()

    # Setup MindRove connection (same as original calibrate.py)
    BoardShim.enable_board_logger()
    params = MindRoveInputParams()
    if args.ip: 
        params.ip_address = args.ip
    if args.port: 
        params.ip_port = args.port
    if args.protocol == "udp": 
        params.ip_protocol = IpProtocolTypes.UDP.value
    else: 
        params.ip_protocol = IpProtocolTypes.TCP.value
    
    # Set timeout like original calibrate.py
    params.timeout = 8000

    board = BoardShim(BOARD_ID, params)
    
    # Try connection with retries (like original calibrate.py)
    for attempt in range(args.retries):
        try:
            board.prepare_session()
            break
        except Exception as e:
            print(f"⚠️  Connection attempt {attempt+1} failed: {e}")
            if attempt == args.retries - 1:
                raise
            time.sleep(2)
    
    # Get channel info
    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID, PRESET)
    accel_idx = BoardShim.get_accel_channels(BOARD_ID, PRESET)
    gyro_idx = BoardShim.get_gyro_channels(BOARD_ID, PRESET)
    
    calibrator.sample_rate = sampling_rate
    
    print(f"\n📡 Streaming at {sampling_rate} Hz")
    print(f"📊 Channels: {len(accel_idx)} accel, {len(gyro_idx)} gyro")
    
    if sampling_rate != SAMPLE_RATE:
        print(f"⚠️  Warning: Expected {SAMPLE_RATE} Hz, got {sampling_rate} Hz")
    
    board.start_stream(450000)
    
    try:
        # Collect calibration data for each gesture
        for gesture_name, gesture_desc in gestures_to_collect:
            success = calibrator.collect_gesture_samples(
                board, accel_idx, gyro_idx, gesture_name, gesture_desc
            )
            
            if not success:
                print(f"❌ Failed to calibrate {gesture_name}")
                continue
        
        # Calculate and save thresholds
        calibrator.save_thresholds(args.outfile)
        
        print(f"\n🎉 Improved calibration complete!")
        print(f"💡 Next step: python improved_thresholds.py --thresholds {args.outfile} --ip {args.ip}")
        
    except KeyboardInterrupt:
        print("\n🛑 Calibration interrupted")
    finally:
        try:
            board.stop_stream()
        except:
            pass
        board.release_session()

if __name__ == '__main__':
    main()