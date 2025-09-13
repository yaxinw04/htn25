"""
MindRove Live Movement Classifier (Python)

Detects 3 movement classes in real time from IMU streams:
  1) Swinging down
  2) Reaching forward
  3) Arm rotation (right / left)

Prereqs:
  pip install mindrove
Hardware:
  MindRove WiFi board (has accel + gyro). Adjust BOARD_ID if different.

How it works (simple rule-based v0):
  - Stream accel (AX, AY, AZ) and gyro (GX, GY, GZ) at device sampling rate
  - Low-pass accel (cutoff ~5 Hz) and gyro (cutoff ~8 Hz)
  - Use a sliding window (0.25 s) of features
  - Classify with thresholds:
      * Swinging down   → strong downward acceleration spike (−Z dir) + pitch angular vel
      * Reaching forward→ forward-axis accel + sustained magnitude + low yaw
      * Rotation R/L    → dominant yaw gyro ± sign
  - Optional calibration to remap axes if your device orientation differs

This is intentionally simple; you can later replace `classify_window()` with an ML model.
"""
from __future__ import annotations
import time
import math
import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
from mindrove.data_filter import DataFilter, FilterTypes

# ------------- CONFIG -------------
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value  # change if you use another board
PRESET   = MindRovePresets.DEFAULT_PRESET.value
LOWPASS_ACCEL_CUTOFF = 5.0   # Hz
LOWPASS_GYRO_CUTOFF  = 8.0   # Hz
WINDOW_SEC  = 0.25
HOP_SEC     = 0.10

# Axis mapping (device frame → logical frame)
# Logical frame: X=forward, Y=right, Z=up
# If your sensor is worn differently, remap here using +1/-1 indices
# e.g., if device X points up, set AXIS_MAP = dict(ax=(2, +1), ay=(1, +1), az=(0, +1))
AXIS_MAP = dict(ax=(0, +1), ay=(1, +1), az=(2, +1), gx=(0, +1), gy=(1, +1), gz=(2, +1))

# Thresholds (tune on your data)
THRESH = dict(
    reach_ax_mean=0.7,      # g units approx if accel in g; otherwise scale accordingly
    reach_ax_rms=0.5,
    swing_neg_az_mean=-0.5,
    swing_pitch_rms=40.0,   # deg/s
    yaw_rot_rms=50.0,       # deg/s
    min_motion_mag=0.15,    # general motion gate
)

LABELS = {
    0: "idle/other",
    1: "swing_down",
    2: "reach_forward",
    3: "rotate_right",
    4: "rotate_left",
}

# ------------- UTILITIES -------------
def _remap_axes(arr3xN: np.ndarray, which: str) -> np.ndarray:
    """Remap raw 3xN to logical frame using AXIS_MAP for accel or gyro."""
    assert arr3xN.shape[0] == 3
    if which == "accel":
        x_idx, x_sign = AXIS_MAP['ax']
        y_idx, y_sign = AXIS_MAP['ay']
        z_idx, z_sign = AXIS_MAP['az']
    else:
        x_idx, x_sign = AXIS_MAP['gx']
        y_idx, y_sign = AXIS_MAP['gy']
        z_idx, z_sign = AXIS_MAP['gz']
    out = np.vstack([
        x_sign * arr3xN[x_idx],
        y_sign * arr3xN[y_idx],
        z_sign * arr3xN[z_idx],
    ])
    return out


def _butter_lowpass_inplace(x: np.ndarray, fs: int, cutoff: float):
    """Low-pass filter each row in-place using DataFilter (Butterworth)."""
    rows, cols = x.shape
    for r in range(rows):
        DataFilter.perform_lowpass(x[r], fs, cutoff, 3, FilterTypes.BUTTERWORTH.value, 0.0)


def features_from_window(ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
                         gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, fs: int) -> dict:
    """Compute simple features on a window."""
    # Means
    ax_mean = float(np.mean(ax))
    ay_mean = float(np.mean(ay))
    az_mean = float(np.mean(az))

    # RMS (magnitude of fluctuations)
    ax_rms = float(np.sqrt(np.mean(np.square(ax - ax_mean))))
    ay_rms = float(np.sqrt(np.mean(np.square(ay - ay_mean))))
    az_rms = float(np.sqrt(np.mean(np.square(az - az_mean))))

    # Gyro RMS (deg/s if gyro already in deg/s)
    gx_rms = float(np.sqrt(np.mean(np.square(gx))))
    gy_rms = float(np.sqrt(np.mean(np.square(gy))))
    gz_rms = float(np.sqrt(np.mean(np.square(gz))))

    # Sign of yaw (gz) to tell right/left
    gz_mean = float(np.mean(gz))

    # Overall motion gate
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    motion_rms = float(np.sqrt(np.mean(np.square(accel_mag - np.mean(accel_mag)))))

    return dict(
        ax_mean=ax_mean, ay_mean=ay_mean, az_mean=az_mean,
        ax_rms=ax_rms, ay_rms=ay_rms, az_rms=az_rms,
        gx_rms=gx_rms, gy_rms=gy_rms, gz_rms=gz_rms, gz_mean=gz_mean,
        motion_rms=motion_rms,
    )


def classify_window(feat: dict) -> int:
    """Very simple rule-based classifier. Returns label id."""
    # Gate: ignore tiny motions
    if feat['motion_rms'] < THRESH['min_motion_mag']:
        return 0

    # Rotation: dominant yaw
    if feat['gz_rms'] > THRESH['yaw_rot_rms'] and feat['gz_rms'] > feat['gx_rms'] and feat['gz_rms'] > feat['gy_rms']:
        return 3 if feat['gz_mean'] > 0 else 4

    # Swing down: downward accel (−Z) and some pitch activity (gy)
    if feat['az_mean'] < THRESH['swing_neg_az_mean'] and feat['gy_rms'] > THRESH['swing_pitch_rms']:
        return 1

    # Reach forward: forward accel (X) with decent RMS, not much yaw
    if feat['ax_mean'] > THRESH['reach_ax_mean'] and feat['ax_rms'] > THRESH['reach_ax_rms'] and feat['gz_rms'] < THRESH['yaw_rot_rms']:
        return 2

    return 0


# ------------- MAIN LOOP -------------
def main():
    BoardShim.enable_board_logger()
    params = MindRoveInputParams()
    # Fill params as needed, e.g., params.ip_address = "..."; params.ip_port = 1234

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    sr = BoardShim.get_sampling_rate(BOARD_ID, PRESET)
    print(f"Sampling rate: {sr} Hz")

    accel_idx = BoardShim.get_accel_channels(BOARD_ID, PRESET)
    gyro_idx  = BoardShim.get_gyro_channels(BOARD_ID, PRESET)
    ts_idx    = BoardShim.get_timestamp_channel(BOARD_ID, PRESET)

    if len(accel_idx) < 3 or len(gyro_idx) < 3:
        raise RuntimeError("This board does not expose 3-axis accel/gyro on the selected preset.")

    board.start_stream(450000)
    print("Streaming… Press Ctrl+C to stop.")

    win_len = int(WINDOW_SEC * sr)
    hop_len = int(HOP_SEC * sr)
    buf_acc = np.empty((3, 0))
    buf_gyro = np.empty((3, 0))

    try:
        last_time = time.time()
        while True:
            time.sleep(HOP_SEC * 0.9)
            # Pull latest chunk (won't remove older than requested count)
            n = board.get_board_data_count(PRESET)
            if n <= 0:
                continue
            data = board.get_current_board_data(min(n, hop_len), PRESET)
            # data shape: rows x cols; rows include many channels
            acc = data[accel_idx, :]
            gyr = data[gyro_idx, :]

            # Accumulate into buffers
            buf_acc = np.concatenate([buf_acc, acc], axis=1)
            buf_gyro = np.concatenate([buf_gyro, gyr], axis=1)

            # Keep only last win_len samples
            if buf_acc.shape[1] > win_len:
                buf_acc = buf_acc[:, -win_len:]
                buf_gyro = buf_gyro[:, -win_len:]

            if buf_acc.shape[1] < win_len:
                continue  # not enough samples yet

            # Copy for filtering (in-place ops)
            acc_win = _remap_axes(buf_acc.copy(), which="accel")
            gyr_win = _remap_axes(buf_gyro.copy(), which="gyro")
            _butter_lowpass_inplace(acc_win, sr, LOWPASS_ACCEL_CUTOFF)
            _butter_lowpass_inplace(gyr_win, sr, LOWPASS_GYRO_CUTOFF)

            feat = features_from_window(
                acc_win[0], acc_win[1], acc_win[2],
                gyr_win[0], gyr_win[1], gyr_win[2],
                sr,
            )
            label = classify_window(feat)
            now = time.time()
            if now - last_time > 0.25:
                print(f"{LABELS[label]:>14s} | ax_mean={feat['ax_mean']:+.2f} ax_rms={feat['ax_rms']:.2f}  az_mean={feat['az_mean']:+.2f}  gz_rms={feat['gz_rms']:.1f}")
                last_time = now

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        board.release_session()


if __name__ == "__main__":
    main()
