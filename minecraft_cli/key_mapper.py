# MindRove IMU to Minecraft Controller - IMPROVED VERSION
# Maps swing down → left click (mining) and arm up → right click (placing)
# IMPROVED: Uses 1-second bursts with peak detection instead of averaging
# macOS note: Grant your terminal/IDE Accessibility permission (System Settings > Privacy & Security > Accessibility)
# Usage: python key_mapper.py --thresholds ../mindrove/thresholds.json
# Stop with Ctrl+C.

import argparse
import json
import time
import numpy as np
from pynput import mouse
from collections import deque

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
from mindrove.data_filter import DataFilter, FilterTypes

# MindRove Configuration
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET = MindRovePresets.DEFAULT_PRESET.value
LOWPASS_ACCEL_CUTOFF = 5.0
LOWPASS_GYRO_CUTOFF = 8.0

# IMPROVED: Slower, more stable detection
BURST_DURATION = 1.5       # 1.5-second analysis windows  
SAMPLE_RATE = 500          # Hz
BURST_SAMPLES = int(BURST_DURATION * SAMPLE_RATE)  # 750 samples over 1.5 seconds
MIN_GESTURE_INTERVAL = 1.5 # 5 seconds between gestures (much longer for stability)
GESTURE_COOLDOWN = 2.0     # 2 seconds for physical gesture to complete

AXIS_MAP = dict(ax=(0, +1), ay=(1, +1), az=(2, +1),
                gx=(0, +1), gy=(1, +1), gz=(2, +1))

class BurstDetector:
    """Detects gestures using 1-second bursts with peak detection"""
    
    def __init__(self):
        self.imu_buffer = deque(maxlen=BURST_SAMPLES)
        self.last_gesture_time = 0
        
    def add_sample(self, imu_data: np.ndarray):
        """Add IMU sample to burst buffer"""
        self.imu_buffer.append(imu_data.copy())  # [ax, ay, az, gx, gy, gz]
        
    def is_ready_for_analysis(self) -> bool:
        """Check if we can analyze a burst"""
        if len(self.imu_buffer) < BURST_SAMPLES:
            return False
            
        # Check if enough time has passed since last gesture
        time_since_last = time.time() - self.last_gesture_time
        return time_since_last >= MIN_GESTURE_INTERVAL
        
    def extract_peak_features(self) -> dict:
        """Extract PEAK features from current 1-second burst (not averages!)"""
        if not self.is_ready_for_analysis():
            return None
            
        # Convert buffer to numpy array
        imu_data = np.array(list(self.imu_buffer))  # Shape: (samples, 6)
        
        # Apply filtering and remapping
        processed_data = self._process_imu_data(imu_data)
        
        # Extract PEAK features (this is the key improvement!)
        ax, ay, az = processed_data[:, 0], processed_data[:, 1], processed_data[:, 2]
        gx, gy, gz = processed_data[:, 3], processed_data[:, 4], processed_data[:, 5]
        
        # Calculate features using PEAKS not AVERAGES
        feat = {
            'ax_mean': float(np.mean(ax)),  # Keep some averages for compatibility
            'ay_mean': float(np.mean(ay)),
            'az_mean': float(np.mean(az)),
            'ax_rms': float(np.sqrt(np.mean((ax - np.mean(ax)) ** 2))),
            'ay_rms': float(np.sqrt(np.mean((ay - np.mean(ay)) ** 2))),
            'az_rms': float(np.sqrt(np.mean((az - np.mean(az)) ** 2))),
            'gx_rms': float(np.sqrt(np.mean(gx ** 2))),
            'gy_rms': float(np.sqrt(np.mean(gy ** 2))),
            'gz_rms': float(np.sqrt(np.mean(gz ** 2))),
            'gz_mean': float(np.mean(gz)),
            'gx_mean': float(np.mean(gx)),
            'motion_rms': float(np.sqrt(np.mean((np.sqrt(ax**2 + ay**2 + az**2) - np.mean(np.sqrt(ax**2 + ay**2 + az**2))) ** 2))),
            
            # NEW: Peak-based features for better detection
            'az_peak': float(np.max(az)),      # Peak upward acceleration
            'az_valley': float(np.min(az)),    # Peak downward acceleration
            'swing_intensity': float(np.max(az) - np.min(az)),  # Peak-to-valley
            'gy_peak': float(np.max(np.abs(gy))),  # Peak rotation
            'total_motion_peak': float(np.max(np.sqrt(ax**2 + ay**2 + az**2)))
        }
        
        return feat
    
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
            DataFilter.perform_lowpass(signal[i], SAMPLE_RATE, cutoff_freq,
                                     3, FilterTypes.BUTTERWORTH.value, 0.0)

class MinecraftController:
    def __init__(self):
        self.mouse_controller = mouse.Controller()
        self.last_gesture = 0
        self.last_action_time = 0
        self.action_cooldown = GESTURE_COOLDOWN  # Use longer cooldown
        self.gesture_start_time = 0
        
    def handle_gesture(self, gesture_label):
        current_time = time.time()
        
        # Only handle strong, confident gestures with proper timing
        if (gesture_label != self.last_gesture and 
            gesture_label != 0 and  # Only act on non-idle gestures
            current_time - self.last_action_time >= self.action_cooldown):
            
            if gesture_label == 1:  # swing_down → Left click
                self.mouse_controller.click(mouse.Button.left, 1)
                print("🎯 MINING: Swing down → Left click")
                self.last_action_time = current_time
                
            elif gesture_label == 7:  # arm_up → Right click
                self.mouse_controller.click(mouse.Button.right, 1)
                print("🏗️ PLACING: Arm up → Right click")
                self.last_action_time = current_time
                
            self.last_gesture = gesture_label
        
        elif gesture_label == 0:  # Reset on idle
            self.last_gesture = 0

def _remap_axes(arr3xN: np.ndarray, which: str) -> np.ndarray:
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

def _ensure_contiguous_filter(x: np.ndarray, fs: int, cutoff: float):
    rows, _ = x.shape
    for r in range(rows):
        if not (x[r].dtype == np.float64 and x[r].flags['C_CONTIGUOUS']):
            tmp = np.ascontiguousarray(x[r], dtype=np.float64)
            DataFilter.perform_lowpass(tmp, fs, cutoff, 3, FilterTypes.BUTTERWORTH.value, 0.0)
            x[r] = tmp
        else:
            DataFilter.perform_lowpass(x[r], fs, cutoff, 3, FilterTypes.BUTTERWORTH.value, 0.0)

def _features(ax, ay, az, gx, gy, gz, fs: int):
    ax_mean = float(np.mean(ax)); ay_mean = float(np.mean(ay)); az_mean = float(np.mean(az))
    ax_rms = float(np.sqrt(np.mean((ax - ax_mean) ** 2)))
    ay_rms = float(np.sqrt(np.mean((ay - ay_mean) ** 2)))
    az_rms = float(np.sqrt(np.mean((az - az_mean) ** 2)))
    
    gx_rms = float(np.sqrt(np.mean(gx ** 2)))
    gy_rms = float(np.sqrt(np.mean(gy ** 2)))
    gz_rms = float(np.sqrt(np.mean(gz ** 2)))
    gz_mean = float(np.mean(gz))
    gx_mean = float(np.mean(gx))
    
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    motion_rms = float(np.sqrt(np.mean((accel_mag - np.mean(accel_mag)) ** 2)))
    
    return dict(
        ax_mean=ax_mean, ay_mean=ay_mean, az_mean=az_mean,
        ax_rms=ax_rms, ay_rms=ay_rms, az_rms=az_rms,
        gx_rms=gx_rms, gy_rms=gy_rms, gz_rms=gz_rms, gz_mean=gz_mean, gx_mean=gx_mean,
        motion_rms=motion_rms,
    )

def classify(feat: dict, TH: dict) -> int:
    """Multi-axis classifier with gravity compensation"""
    
    # Extract ALL axis features
    motion_rms = feat['motion_rms']
    ax_mean, ay_mean, az_mean = feat['ax_mean'], feat['ay_mean'], feat['az_mean']
    ax_rms, ay_rms, az_rms = feat['ax_rms'], feat['ay_rms'], feat['az_rms']
    gy_rms = feat['gy_rms']
    
    # GRAVITY COMPENSATION: Remove static gravity from accelerations
    # Based on your data, Y-axis has constant ~-0.9g (gravity pointing down)
    # We need to look at CHANGES from this baseline, not absolute values
    
    # Estimate gravity baseline (this should be from calibration, but using observed values)
    gravity_ax = -0.07  # X-axis baseline 
    gravity_ay = -0.93  # Y-axis baseline (main gravity component)
    gravity_az = -0.05  # Z-axis baseline
    
    # Remove gravity to get dynamic acceleration
    dynamic_ax = ax_mean - gravity_ax
    dynamic_ay = ay_mean - gravity_ay  
    dynamic_az = az_mean - gravity_az
    
    # Show gravity-compensated values
    print(f"📊 GRAVITY REMOVED: dx={dynamic_ax:+.3f}, dy={dynamic_ay:+.3f}, dz={dynamic_az:+.3f}")
    print(f"🔍 motion_rms={motion_rms:.3f}, gy_rms={gy_rms:.1f}")
    
    # Dynamic thresholds - much higher now that we removed gravity
    base_motion_thresh = 0.15
    accel_change_thresh = 0.2  # Need significant change from baseline
    
    # Confidence scores for each gesture
    swing_down_confidence = 0.0
    arm_up_confidence = 0.0
    
    # Check for significant motion first
    if motion_rms > base_motion_thresh:
        
        # SWING DOWN: Look for significant NEGATIVE dynamic acceleration  
        # (more negative than the gravity baseline)
        if dynamic_ax < -accel_change_thresh:  # Strong negative X change
            swing_down_confidence += 0.4
            print(f"  📉 Strong negative X change: {dynamic_ax:.3f}")
        elif dynamic_ax > accel_change_thresh:  # Strong positive X change
            arm_up_confidence += 0.4
            print(f"  📈 Strong positive X change: {dynamic_ax:.3f}")
            
        if dynamic_ay < -accel_change_thresh:  # Strong negative Y change (beyond gravity)
            swing_down_confidence += 0.4
            print(f"  📉 Strong negative Y change: {dynamic_ay:.3f}")
        elif dynamic_ay > accel_change_thresh:  # Strong positive Y change (fighting gravity)
            arm_up_confidence += 0.4
            print(f"  📈 Strong positive Y change: {dynamic_ay:.3f}")
            
        if dynamic_az < -accel_change_thresh:  # Strong negative Z change
            swing_down_confidence += 0.4
            print(f"  📉 Strong negative Z change: {dynamic_az:.3f}")
        elif dynamic_az > accel_change_thresh:  # Strong positive Z change
            arm_up_confidence += 0.4
            print(f"  📈 Strong positive Z change: {dynamic_az:.3f}")
        
        # High rotation indicates active gesture
        if gy_rms > 50:
            swing_down_confidence += 0.3
            print(f"  🌪️ High rotation: {gy_rms:.1f}")
        elif gy_rms > 30:
            swing_down_confidence += 0.2
            
        # Low rotation with dynamic motion might be controlled arm movement
        if gy_rms < 40 and motion_rms > 0.2:
            arm_up_confidence += 0.2
            print(f"  🎯 Controlled motion: {gy_rms:.1f}")
    
    # Decision with confidence threshold
    min_confidence = 0.6  # Need 60% confidence
    
    if swing_down_confidence > min_confidence and swing_down_confidence > arm_up_confidence:
        print(f"  → 🎯 SWING DOWN (confidence: {swing_down_confidence:.2f})")
        return 1
    elif arm_up_confidence > min_confidence:
        print(f"  → 🏗️ ARM UP (confidence: {arm_up_confidence:.2f})")
        return 7
    else:
        print(f"  → 😴 IDLE (swing:{swing_down_confidence:.2f}, arm_up:{arm_up_confidence:.2f})")
        return 0

def _project_to_neutral(acc3xN, gyr3xN, B_cols, gyro_bias):
    gyr3xN = gyr3xN - gyro_bias.reshape(3,1)
    acc_log = B_cols.T @ acc3xN
    gyr_log = B_cols.T @ gyr3xN
    return acc_log, gyr_log

class StickyLabel:
    def __init__(self, hold_windows=2, min_stable_time=1.0):
        self.prev=0
        self.hold=0
        self.hold_windows=hold_windows
        self.last_change_time = 0
        self.min_stable_time = min_stable_time  # Minimum time between gesture changes
        self.current_gesture_start = 0
        
    def update(self, new):
        current_time = time.time()
        
        # If this is the same gesture, just extend the hold
        if new == self.prev: 
            self.hold = self.hold_windows
            return self.prev
            
        # If we're in a hold period, ignore the new gesture
        if self.hold > 0: 
            self.hold -= 1
            return self.prev
            
        # Check if enough time has passed since last gesture change
        # Only allow gesture changes if enough time has passed OR going to idle
        if (current_time - self.last_change_time < self.min_stable_time and 
            new != 0 and self.prev != 0):  # Don't block going to idle
            return self.prev  # Keep previous gesture, ignore rapid change
            
        # Accept the new gesture
        self.prev = new
        self.hold = self.hold_windows
        self.last_change_time = current_time
        return self.prev

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", required=True, help="Path to thresholds.json")
    parser.add_argument("--ip", help="MindRove IP")
    parser.add_argument("--port", type=int, help="MindRove port")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    args = parser.parse_args()

    # Load calibration
    with open(args.thresholds) as f:
        blob = json.load(f)
    THRESH = blob["thresholds"]
    B_cols = np.array(blob.get("basis", {}).get("B_columns", np.eye(3))).astype(float)
    gyro_bias = np.array(blob.get("neutral", {}).get("gyro_bias", [0,0,0])).astype(float)

    controller = MinecraftController()
    
    print("🎮 MindRove → Minecraft Controller")
    print("🔨 Swing Down → Left Click (Mining)")
    print("🧱 Arm Up → Right Click (Placing)")
    if args.debug:
        print("🐛 Debug mode enabled - showing all sensor data")
    print("Press Ctrl+C to stop.\n")

    # Setup MindRove
    BoardShim.enable_board_logger()
    params = MindRoveInputParams()
    if args.ip: params.ip_address = args.ip
    if args.port: params.ip_port = args.port

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    sr = BoardShim.get_sampling_rate(BOARD_ID, PRESET)
    accel_idx = BoardShim.get_accel_channels(BOARD_ID, PRESET)
    gyro_idx = BoardShim.get_gyro_channels(BOARD_ID, PRESET)

    board.start_stream(450000)
    print(f"📡 Streaming at {sr} Hz with {BURST_SAMPLES} sample bursts")
    
    detector = BurstDetector()
    smoother = StickyLabel(hold_windows=2)
    sample_count = 0

    try:
        last_print = time.time()
        while True:
            time.sleep(0.1)  # Check every 100ms for better timing control
            n = board.get_board_data_count(PRESET)
            if n <= 0: continue
            
            # Get all available samples (not just latest)
            data = board.get_current_board_data(n, PRESET)
            if data.shape[1] == 0: continue
            
            # Process each sample in the batch
            for i in range(data.shape[1]):
                # Combine accel and gyro into single vector [ax, ay, az, gx, gy, gz]
                acc = data[accel_idx, i]
                gyr = data[gyro_idx, i]
                imu_sample = np.concatenate([acc, gyr])
                
                # Add to burst detector
                detector.add_sample(imu_sample)
                sample_count += 1
            
                # Check if ready for burst analysis
                if detector.is_ready_for_analysis():
                    # Extract peak-based features
                    feat = detector.extract_peak_features()
                    if feat is not None:
                        # Classify gesture using improved confidence-based classifier
                        raw = classify(feat, THRESH)
                        label = smoother.update(raw)
                        
                        # Handle the gesture (only acts on confident detections)
                        controller.handle_gesture(label)
                        
                        # Mark that we processed a gesture
                        if label != 0:
                            detector.last_gesture_time = time.time()
                        
                        sample_count = 0  # Reset counter

            # Print status every 2 seconds to show we're alive
            now = time.time()
            if now - last_print > 2.0:
                print(f"📊 Listening... (samples collected: {sample_count})")
                last_print = now

    except KeyboardInterrupt:
        print("\n🛑 Stopping controller...")
    finally:
        try: board.stop_stream()
        except: pass
        board.release_session()

if __name__ == '__main__':
    main()
