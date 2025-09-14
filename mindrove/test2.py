"""
detect_movements.py
Live movement detection using thresholds + neutral frame from calibration JSON.

Loads:
  - thresholds: per-motion thresholds
  - neutral.gravity_vec, neutral.gyro_bias
  - basis.B_columns: 3x3 matrix with columns [X_forward, Y_right, Z_up]

Run:
  python detect_movements.py --thresholds thresholds.json
  # optional:
  #   --ip 192.168.4.1 --port 4210
  #   --sensitivity 0.7
  #   --arm-up-scale 0.75
  #   --swing-scale 1.10
"""

from __future__ import annotations
import argparse
import json
import time
import numpy as np

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
from mindrove.data_filter import DataFilter, FilterTypes

# ----------- Defaults you can edit -----------
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET   = MindRovePresets.DEFAULT_PRESET.value

LOWPASS_ACCEL_CUTOFF = 5.0   # Hz
LOWPASS_GYRO_CUTOFF  = 8.0   # Hz
WINDOW_SEC  = 0.25
HOP_SEC     = 0.10

# Initial device→logical remap (before applying your neutral basis)
# Logical frame here is: X=forward, Y=right, Z=up
AXIS_MAP = dict(ax=(0, +1), ay=(1, +1), az=(2, +1),
                gx=(0, +1), gy=(1, +1), gz=(2, +1))

LABELS = {
    0: "idle/other",
    1: "swing_down",
    7: "arm_up",
}

# ---------- axis & filtering helpers ----------
def _remap_axes(arr3xN: np.ndarray, which: str) -> np.ndarray:
    """Map device axes to a consistent logical frame (X=forward, Y=right, Z=up)."""
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
    """Butterworth low-pass each row in-place; robust to non-contiguous views."""
    rows, _ = x.shape
    for r in range(rows):
        if not (x[r].dtype == np.float64 and x[r].flags['C_CONTIGUOUS']):
            tmp = np.ascontiguousarray(x[r], dtype=np.float64)
            DataFilter.perform_lowpass(tmp, fs, cutoff, 3, FilterTypes.BUTTERWORTH.value, 0.0)
            x[r] = tmp
        else:
            DataFilter.perform_lowpass(x[r], fs, cutoff, 3, FilterTypes.BUTTERWORTH.value, 0.0)

# ---------- feature extraction ----------
def _features(ax, ay, az, gx, gy, gz, fs: int):
    """Extract simple stats per window."""
    # gentle gravity removal helper (for high-pass RMS used in rotation veto)
    hp_len = max(3, min(int(0.35 * fs), len(ax) - 1))
    ker = np.ones(hp_len) / hp_len
    def _hp(sig): return sig - np.convolve(sig, ker, mode="same")
    ax_hp, az_hp = _hp(ax), _hp(az)

    ax_mean = float(np.mean(ax)); ay_mean = float(np.mean(ay)); az_mean = float(np.mean(az))
    ax_rms  = float(np.sqrt(np.mean((ax - ax_mean) ** 2)))
    ay_rms  = float(np.sqrt(np.mean((ay - ay_mean) ** 2)))
    az_rms  = float(np.sqrt(np.mean((az - az_mean) ** 2)))

    gx_rms  = float(np.sqrt(np.mean(gx ** 2)))
    gy_rms  = float(np.sqrt(np.mean(gy ** 2)))
    gz_rms  = float(np.sqrt(np.mean(gz ** 2)))
    gz_mean = float(np.mean(gz))
    gx_mean = float(np.mean(gx))

    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    motion_rms = float(np.sqrt(np.mean((accel_mag - np.mean(accel_mag)) ** 2)))

    ax_hp_rms = float(np.sqrt(np.mean(ax_hp ** 2)))
    az_hp_rms = float(np.sqrt(np.mean(az_hp ** 2)))

    return dict(
        ax_mean=ax_mean, ay_mean=ay_mean, az_mean=az_mean,
        ax_rms=ax_rms, ay_rms=ay_rms, az_rms=az_rms,
        gx_rms=gx_rms, gy_rms=gy_rms, gz_rms=gz_rms, gz_mean=gz_mean, gx_mean=gx_mean,
        motion_rms=motion_rms,
        ax_hp_rms=ax_hp_rms, az_hp_rms=az_hp_rms,
    )

# ---------- classification ----------
def classify(feat: dict, TH: dict) -> int:
    """
    Simplified classifier for only swing_down and arm_up gestures.
    """
    # --- motion gate ---
    if feat['motion_rms'] < TH.get('min_motion_mag', 0.12):
        # Still allow arm up if strongly above threshold while mostly static
        if feat['az_mean'] > TH.get('arm_up_az_mean', 0.40):
            return 7
        return 0

    # --- 1) SWING DOWN (prioritized) ---
    # Downward accel (−Z) + strong pitch (gy)
    if (feat['az_mean'] < TH.get('swing_neg_az_mean', -0.55) and
        feat['gy_rms']  > TH.get('swing_pitch_rms', 60.0)):
        return 1

    # --- 2) ARM UP ---
    if feat['az_mean'] > TH.get('arm_up_az_mean', 0.40):
        return 7

    # --- 3) idle/other ---
    return 0

class StickyLabel:
    def __init__(self, hold_windows=2):
        self.prev = 0
        self.hold = 0
        self.hold_windows = hold_windows
    def update(self, new):
        if new == self.prev:
            self.hold = self.hold_windows
            return self.prev
        if self.hold > 0:
            self.hold -= 1
            return self.prev
        self.prev = new
        self.hold = self.hold_windows
        return self.prev

# ---------- sensitivity control ----------
def relax_thresholds(thresh: dict, scale: float = 0.7) -> dict:
    """
    Make detection *more sensitive* by lowering thresholds.
    For negative thresholds (e.g., swing_neg_az_mean), scaling toward 0 makes them easier to meet.
    """
    t = thresh.copy()
    for k in ["swing_pitch_rms", "min_motion_mag"]:
        if k in t: t[k] *= scale
    if "swing_neg_az_mean" in t:
        t["swing_neg_az_mean"] *= scale  # less negative = easier swing
    if 'arm_up_az_mean' in t:
        t['arm_up_az_mean'] *= scale     # smaller az mean required = easier arm up
    return t

# ---------- neutral projection ----------
def _project_to_neutral(acc3xN, gyr3xN, B_cols, gyro_bias):
    """
    Project signals to the calibrated neutral frame:
      acc_log = B^T * acc_map
      gyr_log = B^T * (gyr_map - gyro_bias)
    """
    gyr3xN = gyr3xN - gyro_bias.reshape(3, 1)
    acc_log = B_cols.T @ acc3xN
    gyr_log = B_cols.T @ gyr3xN
    return acc_log, gyr_log

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", required=True, help="Path to thresholds JSON from calibration")
    ap.add_argument("--ip", help="MindRove WiFi board IP (optional)")
    ap.add_argument("--port", type=int, help="MindRove WiFi board port (optional)")
    ap.add_argument("--sensitivity", type=float, default=0.7,
                    help="Global scale (<1.0 = more sensitive). 1.0 uses calibrated values.")
    ap.add_argument("--arm-up-scale", type=float, default=0.75,
                    help="Additional scale for ARM UP threshold (<1 = easier).")
    ap.add_argument("--swing-scale", type=float, default=1.10,
                    help="Scale for SWING DOWN thresholds (>1 = stricter; <1 = looser).")
    args = ap.parse_args()

    blob = json.load(open(args.thresholds))
    THRESH = blob["thresholds"]
    B_cols = np.array(blob.get("basis", {}).get("B_columns", np.eye(3))).astype(float)  # 3x3
    gyro_bias = np.array(blob.get("neutral", {}).get("gyro_bias", [0, 0, 0])).astype(float)

    # 1) Global relax/tighten (default 0.7 = more sensitive)
    if args.sensitivity != 1.0:
        THRESH = relax_thresholds(THRESH, scale=args.sensitivity)
        base_scale_note = f"relaxed thresholds (global scale={args.sensitivity:.2f})"
    else:
        base_scale_note = "calibrated thresholds (global scale=1.00)"

    # 2) Per-gesture tweaks
    # ARM UP: lower az_mean threshold → easier to trigger
    if "arm_up_az_mean" in THRESH:
        THRESH["arm_up_az_mean"] *= args.arm_up_scale
    # SWING DOWN: scale pitch RMS; and swing_neg_az_mean (note: neg value → >1 makes it more negative = stricter)
    if "swing_pitch_rms" in THRESH:
        THRESH["swing_pitch_rms"] *= args.swing_scale
    if "swing_neg_az_mean" in THRESH:
        THRESH["swing_neg_az_mean"] *= args.swing_scale

    print(f"Using {base_scale_note}")
    print(f"[per-gesture] arm_up_scale={args.arm_up_scale:.2f}  swing_scale={args.swing_scale:.2f}")
    print("\n=== Active thresholds ===")
    for k in sorted(THRESH.keys()):
        v = THRESH[k]
        try:
            print(f"  {k:>32s}: {float(v):.3f}")
        except Exception:
            print(f"  {k:>32s}: {v}")

    BoardShim.enable_board_logger()
    params = MindRoveInputParams()
    if args.ip:   params.ip_address = args.ip
    if args.port: params.ip_port    = args.port

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    sr = BoardShim.get_sampling_rate(BOARD_ID, PRESET)
    accel_idx = BoardShim.get_accel_channels(BOARD_ID, PRESET)
    gyro_idx  = BoardShim.get_gyro_channels(BOARD_ID, PRESET)
    if len(accel_idx) < 3 or len(gyro_idx) < 3:
        raise RuntimeError("Board lacks 3-axis accel/gyro on this preset.")

    board.start_stream(450000)
    print(f"\nStreaming at {sr} Hz. Press Ctrl+C to stop.\n")

    win_len = int(WINDOW_SEC * sr); hop_len = int(HOP_SEC * sr)
    buf_acc = np.empty((3, 0)); buf_gyro = np.empty((3, 0))
    smoother = StickyLabel(hold_windows=2)

    try:
        last = time.time()
        while True:
            time.sleep(HOP_SEC * 0.9)
            n = board.get_board_data_count(PRESET)
            if n <= 0:
                continue
            data = board.get_current_board_data(min(n, hop_len), PRESET)
            acc = data[accel_idx, :]
            gyr = data[gyro_idx, :]

            buf_acc = np.concatenate([buf_acc, acc], axis=1)
            buf_gyro = np.concatenate([buf_gyro, gyr], axis=1)
            if buf_acc.shape[1] > win_len:
                buf_acc = buf_acc[:, -win_len:]
                buf_gyro = buf_gyro[:, -win_len:]
            if buf_acc.shape[1] < win_len:
                continue

            # 1) device → logical remap
            acc_win = _remap_axes(buf_acc.copy(), "accel")
            gyr_win = _remap_axes(buf_gyro.copy(), "gyro")

            # 2) low-pass filtering
            _ensure_contiguous_filter(acc_win, sr, LOWPASS_ACCEL_CUTOFF)
            _ensure_contiguous_filter(gyr_win, sr, LOWPASS_GYRO_CUTOFF)

            # 3) project to your neutral frame (forward/right/up)
            acc_win, gyr_win = _project_to_neutral(acc_win, gyr_win, B_cols, gyro_bias)

            # 4) features + classify
            feat = _features(acc_win[0], acc_win[1], acc_win[2],
                             gyr_win[0], gyr_win[1], gyr_win[2], sr)
            raw = classify(feat, THRESH)
            label = smoother.update(raw)

            now = time.time()
            if now - last > 0.25:
                print(f"{LABELS[label]:>14s} | ax_mean={feat['ax_mean']:+.2f} "
                      f"ax_rms={feat['ax_rms']:.2f}  az_mean={feat['az_mean']:+.2f}  "
                      f"gz_rms={feat['gz_rms']:.1f}")
                last = now

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
