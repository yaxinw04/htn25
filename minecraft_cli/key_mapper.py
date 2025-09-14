# MindRove IMU to Minecraft Controller
# Maps swing down → left click (mining) and arm up → right click (placing)
# macOS note: Grant your terminal/IDE Accessibility permission (System Settings > Privacy & Security > Accessibility)
# Usage: python key_mapper.py --thresholds ../mindrove/thresholds.json
# Stop with Ctrl+C.

import argparse
import json
import time
import numpy as np
from pynput import mouse

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
from mindrove.data_filter import DataFilter, FilterTypes

# MindRove Configuration
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET = MindRovePresets.DEFAULT_PRESET.value
LOWPASS_ACCEL_CUTOFF = 5.0
LOWPASS_GYRO_CUTOFF = 8.0
WINDOW_SEC = 0.25
HOP_SEC = 0.10

AXIS_MAP = dict(ax=(0, +1), ay=(1, +1), az=(2, +1),
                gx=(0, +1), gy=(1, +1), gz=(2, +1))

class MinecraftController:
    def __init__(self):
        self.mouse_controller = mouse.Controller()
        self.last_gesture = 0
        self.last_action_time = 0
        self.action_cooldown = 0.3
        
    def handle_gesture(self, gesture_label):
        current_time = time.time()
        
        if (gesture_label != self.last_gesture and 
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
    # Use more permissive thresholds for testing
    min_motion = TH.get('min_motion_mag', 0.12)
    swing_az_thresh = -0.3  # More permissive than -0.55
    swing_gy_thresh = 50.0  # More permissive than 60.0
    arm_up_thresh = 0.3     # More permissive than 0.40
    
    # Debug output
    print(f"motion_rms={feat['motion_rms']:.3f}, az_mean={feat['az_mean']:+.3f}, gy_rms={feat['gy_rms']:.1f}")
    
    # Motion gate - but more permissive
    if feat['motion_rms'] < min_motion:
        if feat['az_mean'] > arm_up_thresh:
            print(f"  → ARM UP detected (az_mean={feat['az_mean']:+.3f} > {arm_up_thresh})")
            return 7  # arm_up
        print(f"  → IDLE (low motion: {feat['motion_rms']:.3f} < {min_motion})")
        return 0
    
    # Swing down detection (prioritized)
    if (feat['az_mean'] < swing_az_thresh and feat['gy_rms'] > swing_gy_thresh):
        print(f"  → SWING DOWN detected (az_mean={feat['az_mean']:+.3f} < {swing_az_thresh}, gy_rms={feat['gy_rms']:.1f} > {swing_gy_thresh})")
        return 1  # swing_down
    
    # Arm up detection
    if feat['az_mean'] > arm_up_thresh:
        print(f"  → ARM UP detected (az_mean={feat['az_mean']:+.3f} > {arm_up_thresh})")
        return 7  # arm_up
    
    print(f"  → IDLE (no gesture detected)")
    return 0  # idle

def _project_to_neutral(acc3xN, gyr3xN, B_cols, gyro_bias):
    gyr3xN = gyr3xN - gyro_bias.reshape(3,1)
    acc_log = B_cols.T @ acc3xN
    gyr_log = B_cols.T @ gyr3xN
    return acc_log, gyr_log

class StickyLabel:
    def __init__(self, hold_windows=2):
        self.prev=0; self.hold=0; self.hold_windows=hold_windows
    def update(self, new):
        if new==self.prev: self.hold=self.hold_windows; return self.prev
        if self.hold>0: self.hold-=1; return self.prev
        self.prev=new; self.hold=self.hold_windows; return self.prev

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
    print(f"📡 Streaming at {sr} Hz")
    
    win_len = int(WINDOW_SEC * sr)
    hop_len = int(HOP_SEC * sr)
    buf_acc = np.empty((3, 0))
    buf_gyro = np.empty((3, 0))
    smoother = StickyLabel(hold_windows=2)

    try:
        last_print = time.time()
        while True:
            time.sleep(HOP_SEC * 0.9)
            n = board.get_board_data_count(PRESET)
            if n <= 0: continue
            data = board.get_current_board_data(min(n, hop_len), PRESET)
            acc = data[accel_idx, :]; gyr = data[gyro_idx, :]

            buf_acc = np.concatenate([buf_acc, acc], axis=1)
            buf_gyro = np.concatenate([buf_gyro, gyr], axis=1)
            if buf_acc.shape[1] > win_len:
                buf_acc = buf_acc[:, -win_len:]; buf_gyro = buf_gyro[:, -win_len:]
            if buf_acc.shape[1] < win_len: continue

            # Process data
            acc_win = _remap_axes(buf_acc.copy(), "accel")
            gyr_win = _remap_axes(buf_gyro.copy(), "gyro")
            _ensure_contiguous_filter(acc_win, sr, LOWPASS_ACCEL_CUTOFF)
            _ensure_contiguous_filter(gyr_win, sr, LOWPASS_GYRO_CUTOFF)
            acc_win, gyr_win = _project_to_neutral(acc_win, gyr_win, B_cols, gyro_bias)

            # Classify and handle gesture
            feat = _features(acc_win[0], acc_win[1], acc_win[2],
                           gyr_win[0], gyr_win[1], gyr_win[2], sr)
            raw = classify(feat, THRESH)
            label = smoother.update(raw)
            controller.handle_gesture(label)

            # Print status every 0.5 seconds
            now = time.time()
            if now - last_print > 0.5:
                gesture_name = {0: "idle", 1: "swing_down", 7: "arm_up"}.get(label, f"unknown({label})")
                print(f"📊 Current: {gesture_name}")
                last_print = now

    except KeyboardInterrupt:
        print("\n🛑 Stopping controller...")
    finally:
        try: board.stop_stream()
        except: pass
        board.release_session()

if __name__ == '__main__':
    main()
