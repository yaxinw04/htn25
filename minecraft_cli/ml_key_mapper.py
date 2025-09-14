#!/usr/bin/env python3
"""
ML-Powered MindRove → Minecraft Controller
==========================================

Uses trained ML model (SVM/XGBoost) for gesture classification with:
- 2-second burst detection
- Combined IMU + EMG features
- High-confidence predictions only
- Proper cooldown periods

Usage: python ml_key_mapper.py --model best_model.pkl --ip 192.168.4.1
"""

import argparse
import pickle
import time
import numpy as np
from pynput import mouse, keyboard
from collections import deque
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
from mindrove.data_filter import DataFilter, FilterTypes

# Configuration
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET = MindRovePresets.DEFAULT_PRESET.value

# ML Detection parameters - Updated to match training preprocessing
WINDOW_DURATION = 0.5       # 0.5-second analysis windows (matches training)
SAMPLE_RATE = 500          # Hz  
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLE_RATE)
CONFIDENCE_THRESHOLD = 0.7  # Minimum prediction confidence
ACTION_COOLDOWN = 0.5      # Seconds between actions
BUFFER_SIZE = WINDOW_SAMPLES + 50  # Extra buffer for continuous processing

# Signal processing (must match training)
ACCEL_LP_CUTOFF = 5.0
GYRO_LP_CUTOFF = 8.0
EMG_BP_LOW = 20.0
EMG_BP_HIGH = 450.0
EMG_ENV_CUTOFF = 8.0

AXIS_MAP = dict(ax=(0, +1), ay=(1, +1), az=(2, +1),
                gx=(0, +1), gy=(1, +1), gz=(2, +1))

class MLGestureDetector:
    """ML-based gesture detection using trained model"""
    
    def __init__(self, model_file: str):
        self.model_info = self._load_model(model_file)
        self.model = self.model_info['model']
        self.feature_names = self.model_info['feature_names']
        self.gesture_names = self.model_info['gesture_names']
        self.model_name = self.model_info['model_name']
        
        # Data buffers for window analysis
        self.imu_buffer = deque(maxlen=BUFFER_SIZE)
        self.emg_buffer = deque(maxlen=BUFFER_SIZE)
        self.time_buffer = deque(maxlen=BUFFER_SIZE)
        
        # Prediction tracking
        self.last_prediction_time = 0
        self.prediction_counts = {name: 0 for name in self.gesture_names.values()}
        
        print(f"🤖 Loaded {self.model_name} model")
        print(f"📊 Features: {len(self.feature_names)}")
        print(f"🎯 Classes: {list(self.gesture_names.values())}")
        print(f"⚡ Accuracy: {self.model_info['accuracy']:.3f}")
        
    def _load_model(self, model_file: str) -> dict:
        """Load trained model and metadata"""
        print(f"📥 Loading ML model from {model_file}")
        
        with open(model_file, 'rb') as f:
            model_info = pickle.load(f)
            
        required_keys = ['model', 'feature_names', 'gesture_names', 'model_name', 'accuracy']
        for key in required_keys:
            if key not in model_info:
                raise ValueError(f"Model file missing required key: {key}")
                
        return model_info
    
    def add_sample(self, imu_data: np.ndarray, emg_data: np.ndarray):
        """Add sensor sample to burst buffer"""
        current_time = time.time()
        self.imu_buffer.append(imu_data.copy())
        self.emg_buffer.append(emg_data.copy())
        self.time_buffer.append(current_time)
    
    def is_ready_for_prediction(self) -> bool:
        """Check if ready for ML prediction"""
        if len(self.imu_buffer) < WINDOW_SAMPLES:
            return False
            
        # Cooldown check
        time_since_last = time.time() - self.last_prediction_time
        return time_since_last >= ACTION_COOLDOWN
    
    def predict_gesture(self) -> Tuple[int, float, str]:
        """
        Predict gesture using ML model
        Returns: (gesture_id, confidence, gesture_name)
        """
        if not self.is_ready_for_prediction():
            return 0, 0.0, "idle"
            
        # Extract features from current burst
        features = self._extract_features_for_prediction()
        if features is None:
            return 0, 0.0, "idle"
            
        # Make prediction
        try:
            prediction = self.model.predict([features])[0]
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([features])[0]
                confidence = np.max(probabilities)
                
                # Debug: Show all class probabilities occasionally
                if np.random.random() < 0.01:  # 1% of the time
                    prob_str = ", ".join([f"{self.gesture_names.get(i, f'class_{i}')}: {p:.3f}" 
                                        for i, p in enumerate(probabilities)])
                    print(f"🔍 Class probabilities: {prob_str}")
            else:
                confidence = 1.0  # For models without probability estimates
                
            gesture_name = self.gesture_names.get(prediction, f"unknown_{prediction}")
            
            # Debug: Print prediction info occasionally
            if np.random.random() < 0.05:  # 5% of the time
                print(f"🎯 Prediction: {gesture_name} (confidence: {confidence:.3f}, threshold: {CONFIDENCE_THRESHOLD:.3f})")
            
            # Update prediction tracking
            self.prediction_counts[gesture_name] += 1
            
            return int(prediction), float(confidence), gesture_name
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return 0, 0.0, "idle"
    
    def _extract_features_for_prediction(self) -> Optional[np.ndarray]:
        """Extract windowed statistical features matching training format"""
        try:
            # Get the most recent window of data
            if len(self.imu_buffer) < WINDOW_SAMPLES:
                return None
                
            # Extract window data (most recent WINDOW_SAMPLES)
            window_imu = np.array(list(self.imu_buffer))[-WINDOW_SAMPLES:]  # (50, 6)
            window_emg = np.array(list(self.emg_buffer))[-WINDOW_SAMPLES:]  # (50, n_emg)
            
            # Sensor columns (match training preprocessing)
            sensor_data = np.column_stack([window_imu, window_emg])  # (50, 14)
            
            # Extract statistical features exactly like training
            features = self._extract_window_features(sensor_data)
            
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def _extract_window_features(self, window_data: np.ndarray) -> np.ndarray:
        """Extract statistical features from time window (SAME AS TRAINING)"""
        features = []
        
        # For each sensor channel (14 channels: 6 IMU + 8 EMG)
        for ch in range(window_data.shape[1]):
            channel_data = window_data[:, ch]
            
            # Statistical features (same as training)
            features.append(np.mean(channel_data))      # Mean
            features.append(np.std(channel_data))       # Standard deviation  
            features.append(np.min(channel_data))       # Minimum
            features.append(np.max(channel_data))       # Maximum
            features.append(np.max(channel_data) - np.min(channel_data))  # Range
            features.append(np.sqrt(np.mean(channel_data**2)))  # RMS
        
        # Cross-channel features (same as training)
        accel_data = window_data[:, :3]  # Accelerometer
        gyro_data = window_data[:, 3:6]  # Gyroscope
        emg_data = window_data[:, 6:]    # EMG
        
        # Magnitude features
        accel_mag = np.sqrt(np.sum(accel_data**2, axis=1))
        gyro_mag = np.sqrt(np.sum(gyro_data**2, axis=1))
        
        features.extend([
            np.mean(accel_mag), np.std(accel_mag), np.max(accel_mag),
            np.mean(gyro_mag), np.std(gyro_mag), np.max(gyro_mag),
            np.mean(np.sum(emg_data, axis=1))  # Total EMG activation
        ])
        
        return np.array(features)
    
    def _process_imu(self, imu_data: np.ndarray) -> np.ndarray:
        """Process IMU data (same as training)"""
        accel_data = imu_data[:, :3].T  # (3, samples)
        gyro_data = imu_data[:, 3:6].T  # (3, samples)
        
        # Remap axes
        accel_remapped = self._remap_axes(accel_data, "accel")
        gyro_remapped = self._remap_axes(gyro_data, "gyro")
        
        # Filter
        self._filter_signal(accel_remapped, ACCEL_LP_CUTOFF)
        self._filter_signal(gyro_remapped, GYRO_LP_CUTOFF)
        
        # Combine
        combined = np.vstack([accel_remapped, gyro_remapped]).T  # (samples, 6)
        return combined
    
    def _process_emg(self, emg_data: np.ndarray) -> np.ndarray:
        """Process EMG data (same as training)"""
        if emg_data.shape[1] == 0:
            return np.zeros((emg_data.shape[0], 8))
            
        emg_processed = emg_data.T.astype(np.float64, copy=True)  # (channels, samples)
        
        # Bandpass filter
        for i in range(emg_processed.shape[0]):
            DataFilter.perform_bandpass(emg_processed[i], SAMPLE_RATE,
                                      EMG_BP_LOW, EMG_BP_HIGH, 3, 
                                      FilterTypes.BUTTERWORTH.value, 0.0)
        
        # Rectify and envelope
        emg_processed = np.abs(emg_processed)
        for i in range(emg_processed.shape[0]):
            DataFilter.perform_lowpass(emg_processed[i], SAMPLE_RATE,
                                     EMG_ENV_CUTOFF, 3, 
                                     FilterTypes.BUTTERWORTH.value, 0.0)
        
        return emg_processed.T  # (samples, channels)
    
    def _compute_ml_features(self, imu_data: np.ndarray, emg_data: np.ndarray) -> np.ndarray:
        """Compute same features as used in training"""
        features = []
        
        # IMU statistical features (6 channels × 6 stats = 36 features)
        imu_mean = np.mean(imu_data, axis=0)
        imu_std = np.std(imu_data, axis=0) 
        imu_min = np.min(imu_data, axis=0)
        imu_max = np.max(imu_data, axis=0)
        imu_peak_to_peak = imu_max - imu_min
        imu_rms = np.sqrt(np.mean(imu_data**2, axis=0))
        
        features.extend(imu_mean)
        features.extend(imu_std)
        features.extend(imu_min)
        features.extend(imu_max)
        features.extend(imu_peak_to_peak)
        features.extend(imu_rms)
        
        # EMG features (pad to 8 channels, 4 stats each = 32 features)
        n_emg = min(emg_data.shape[1], 8)
        emg_mean = np.mean(emg_data[:, :n_emg], axis=0) if n_emg > 0 else np.zeros(8)
        emg_std = np.std(emg_data[:, :n_emg], axis=0) if n_emg > 0 else np.zeros(8)
        emg_max = np.max(emg_data[:, :n_emg], axis=0) if n_emg > 0 else np.zeros(8)
        emg_integrated = np.trapz(emg_data[:, :n_emg], axis=0) if n_emg > 0 else np.zeros(8)
        
        # Pad to 8 channels
        if len(emg_mean) < 8:
            pad_size = 8 - len(emg_mean)
            emg_mean = np.pad(emg_mean, (0, pad_size))
            emg_std = np.pad(emg_std, (0, pad_size))
            emg_max = np.pad(emg_max, (0, pad_size))
            emg_integrated = np.pad(emg_integrated, (0, pad_size))
        
        features.extend(emg_mean)
        features.extend(emg_std)  
        features.extend(emg_max)
        features.extend(emg_integrated)
        
        # Cross-channel features (8 features)
        accel_mag = np.sqrt(np.sum(imu_data[:, :3]**2, axis=1))
        gyro_mag = np.sqrt(np.sum(imu_data[:, 3:6]**2, axis=1))
        
        accel_magnitude_stats = [np.mean(accel_mag), np.std(accel_mag), 
                               np.max(accel_mag), np.min(accel_mag)]
        gyro_magnitude_stats = [np.mean(gyro_mag), np.std(gyro_mag),
                              np.max(gyro_mag), np.min(gyro_mag)]
        
        features.extend(accel_magnitude_stats)
        features.extend(gyro_magnitude_stats)
        
        # EMG total activation (1 feature)
        emg_total_activation = np.sum(emg_mean[:n_emg]) if n_emg > 0 else 0.0
        features.append(emg_total_activation)
        
        # Temporal features (3 features)
        motion_threshold = np.mean(accel_mag) + np.std(accel_mag)
        motion_onset_idx = np.argmax(accel_mag > motion_threshold) if np.any(accel_mag > motion_threshold) else 0
        peak_motion_idx = np.argmax(accel_mag)
        
        motion_onset_time = motion_onset_idx / SAMPLE_RATE
        peak_motion_time = peak_motion_idx / SAMPLE_RATE
        
        above_threshold = accel_mag > motion_threshold
        gesture_duration = np.sum(above_threshold) / SAMPLE_RATE
        
        features.extend([motion_onset_time, peak_motion_time, gesture_duration])
        
        # Spectral features (6 features)
        dominant_freq_accel = []
        dominant_freq_gyro = []
        
        for i in range(3):  # Accelerometer
            fft_vals = np.abs(np.fft.fft(imu_data[:, i]))
            freqs = np.fft.fftfreq(len(fft_vals), 1/SAMPLE_RATE)
            dominant_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
            dominant_freq_accel.append(abs(freqs[dominant_idx]))
            
        for i in range(3, 6):  # Gyroscope
            fft_vals = np.abs(np.fft.fft(imu_data[:, i]))
            freqs = np.fft.fftfreq(len(fft_vals), 1/SAMPLE_RATE)
            dominant_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
            dominant_freq_gyro.append(abs(freqs[dominant_idx]))
        
        features.extend(dominant_freq_accel)
        features.extend(dominant_freq_gyro)
        
        # Handle NaN/inf values
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return features
    
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
        """Apply filtering in place"""
        for i in range(signal.shape[0]):
            DataFilter.perform_lowpass(signal[i], SAMPLE_RATE, cutoff_freq,
                                     3, FilterTypes.BUTTERWORTH.value, 0.0)

class MLMinecraftController:
    """Minecraft controller using ML predictions with keyboard camera control"""
    
    def __init__(self):
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        self.last_action_time = 0
        self.action_counts = {"mine": 0, "place": 0, "look_left": 0, "look_right": 0, "idle": 0}
        
    def handle_prediction(self, gesture_id: int, confidence: float, gesture_name: str):
        """Handle ML prediction with confidence threshold"""
        current_time = time.time()
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            self.action_counts["idle"] += 1
            return
            
        # Check cooldown (shorter for camera movement)
        camera_gestures = ["wrist_flex_left", "wrist_supinate_right"]
        cooldown = 0.1 if gesture_name in camera_gestures else ACTION_COOLDOWN
        
        if current_time - self.last_action_time < cooldown:
            return
            
        # Execute actions based on gesture
        if gesture_name == "swing_down":
            self.mouse_controller.click(mouse.Button.left, 1)
            print(f"⛏️  MINING: {gesture_name} (confidence: {confidence:.2f})")
            self.action_counts["mine"] += 1
            self.last_action_time = current_time
            
        elif gesture_name == "arm_up":
            self.mouse_controller.click(mouse.Button.right, 1)
            print(f"🧱 PLACING: {gesture_name} (confidence: {confidence:.2f})")
            self.action_counts["place"] += 1
            self.last_action_time = current_time
            
        elif gesture_name == "rotate_left":
            # Press left arrow key for camera control
            self.keyboard_controller.press(keyboard.Key.left)
            self.keyboard_controller.release(keyboard.Key.left)
            print(f"👈 LOOK LEFT: rotate_left → Left Arrow (confidence: {confidence:.2f})")
            self.action_counts["look_left"] += 1
            self.last_action_time = current_time
            
        elif gesture_name == "rotate_right":
            # Press right arrow key for camera control
            self.keyboard_controller.press(keyboard.Key.right)
            self.keyboard_controller.release(keyboard.Key.right)
            print(f"👉 LOOK RIGHT: rotate_right → Right Arrow (confidence: {confidence:.2f})")
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
    parser = argparse.ArgumentParser(description="ML-powered MindRove → Minecraft controller")
    parser.add_argument("--model", required=True, help="Trained ML model file (.pkl)")
    parser.add_argument("--ip", help="MindRove IP address")
    parser.add_argument("--port", type=int, help="MindRove port")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="Confidence threshold (default: 0.7)")
    parser.add_argument("--debug", action="store_true", help="Show prediction details")
    args = parser.parse_args()

    # Update confidence threshold
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence

    print("🎮 ML-Powered MindRove → Minecraft Controller")
    print("=" * 50)
    print(f"🔍 Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"⏱️  Action cooldown: {ACTION_COOLDOWN}s")
    print(f"📊 Window analysis: {WINDOW_DURATION}s windows")
    
    # Initialize ML detector and controller
    ml_detector = MLGestureDetector(args.model)
    controller = MLMinecraftController()
    
    print("\n🎯 Gesture → Action Mapping:")
    print("   swing_down → Left Click (Mining)")
    print("   arm_up → Right Click (Placing)")
    print("   wrist_flex_left → Left Arrow Key (Look Left)")
    print("   wrist_supinate_right → Right Arrow Key (Look Right)")
    print("   🎮 Camera mod compatible - uses arrow keys instead of mouse")
    print("   Other gestures → Ignored")
    print("\nPress Ctrl+C to stop.\n")

    # Setup MindRove connection
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
    
    print(f"📡 Streaming at {sampling_rate} Hz")
    print(f"📊 Channels: {len(accel_idx)} accel, {len(gyro_idx)} gyro, {len(emg_idx)} EMG")
    
    if sampling_rate != SAMPLE_RATE:
        print(f"⚠️  Warning: Expected {SAMPLE_RATE} Hz, got {sampling_rate} Hz")
    
    board.start_stream(450000)
    
    try:
        last_status_time = time.time()
        samples_processed = 0
        predictions_made = 0
        
        while True:
            time.sleep(0.02)  # Slower sampling - 20ms delay
            
            # Get data
            data_count = board.get_board_data_count(PRESET)
            if data_count <= 0:
                continue
                
            data = board.get_current_board_data(data_count, PRESET)
            samples_processed += data_count
            
            # Process every 5th sample to reduce prediction frequency
            process_interval = max(1, data_count // 5)
            
            # Process fewer samples
            for i in range(0, data_count, process_interval):
                # Extract sensor data
                accel_sample = data[accel_idx, i]
                gyro_sample = data[gyro_idx, i]
                imu_sample = np.concatenate([accel_sample, gyro_sample])
                
                emg_sample = data[emg_idx, i] if len(emg_idx) > 0 else np.zeros(8)
                
                # Add to ML detector
                ml_detector.add_sample(imu_sample, emg_sample)
                
                # Make prediction if ready (but less frequently)
                if ml_detector.is_ready_for_prediction() and i % 10 == 0:
                    gesture_id, confidence, gesture_name = ml_detector.predict_gesture()
                    predictions_made += 1
                    
                    if args.debug:
                        print(f"🔍 Prediction: {gesture_name} (id={gesture_id}, conf={confidence:.2f})")
                    
                    # Handle prediction
                    controller.handle_prediction(gesture_id, confidence, gesture_name)
                    ml_detector.last_prediction_time = time.time()
            
            # Status updates
            current_time = time.time()
            if current_time - last_status_time >= 3.0:
                controller.print_stats()
                print(f"📈 Processed: {samples_processed} samples, {predictions_made} predictions")
                
                if predictions_made > 0:
                    print("🎯 Prediction distribution:")
                    for gesture, count in ml_detector.prediction_counts.items():
                        if count > 0:
                            pct = count / predictions_made * 100
                            print(f"   {gesture}: {count} ({pct:.1f}%)")
                
                last_status_time = current_time
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping ML controller...")
    finally:
        try:
            board.stop_stream()
        except:
            pass
        board.release_session()
        
        # Final stats
        total_actions = sum(controller.action_counts.values())
        print(f"\n📊 Final Statistics:")
        print(f"   Samples processed: {samples_processed}")
        print(f"   Predictions made: {predictions_made}")
        print(f"   Total actions: {total_actions}")
        if total_actions > 0:
            print(f"   Mining: {controller.action_counts['mine']}")
            print(f"   Placing: {controller.action_counts['place']}")
            print(f"   Look Left: {controller.action_counts['look_left']}")
            print(f"   Look Right: {controller.action_counts['look_right']}")
            print(f"   Idle: {controller.action_counts['idle']}")

if __name__ == '__main__':
    main()