"""
calibrate_movements.py
Collect IMU windows for 4 motions and save per-user thresholds + neutral basis to JSON.

Motions:
  - swing_down
  - reach_forward
  - rotate_left
  - rotate_right

Also captures a NEUTRAL (still) pose first to estimate:
  - gravity_vec (defines "up")
  - gyro_bias   (stationary drift)
  - body-frame basis (columns = [X_forward, Y_right, Z_up])

Run:
  python calibrate_movements.py --outfile thresholds.json
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

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

# Logical frame: X=forward, Y=right, Z=up (initial remap from device axes)
AXIS_MAP = dict(ax=(0, +1), ay=(1, +1), az=(2, +1),
                gx=(0, +1), gy=(1, +1), gz=(2, +1))

CALIB_SECONDS_PER_CLASS = 6.0
REST_SECONDS_BETWEEN    = 3.0

MOTIONS = [
    ("swing_down",    "Swing DOWN clearly (arm/elbow swing)"),
    ("reach_forward", "Reach FORWARD / push"),
    ("rotate_left",   "Rotate LEFT in place"),
    ("rotate_right",  "Rotate RIGHT in place"),
]

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
    # light gravity removal for extra signals (hp)
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

    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    motion_rms = float(np.sqrt(np.mean((accel_mag - np.mean(accel_mag)) ** 2)))

    ax_hp_rms = float(np.sqrt(np.mean(ax_hp ** 2)))
    az_hp_rms = float(np.sqrt(np.mean(az_hp ** 2)))

    return dict(
        ax_mean=ax_mean, ay_mean=ay_mean, az_mean=az_mean,
        ax_rms=ax_rms, ay_rms=ay_rms, az_rms=az_rms,
        gx_rms=gx_rms, gy_rms=gy_rms, gz_rms=gz_rms, gz_mean=gz_mean,
        motion_rms=motion_rms,
        ax_hp_rms=ax_hp_rms, az_hp_rms=az_hp_rms,
    )

def _pct(vals, p):
    if not vals: return 0.0
    return float(np.percentile(np.asarray(vals, dtype=float), p))

# ---------- neutral capture & basis ----------
def _collect_neutral(board, sr, accel_idx, gyro_idx, seconds: float = 3.0):
    print("\n=== NEUTRAL CAPTURE ===")
    print("Stand STILL in your neutral pose, facing your *forward* direction.")
    for i in range(3, 0, -1):
        print(f"  starting in {i}…", end="\r"); time.sleep(1)
    print("  capturing…        ")

    hop_len = int(HOP_SEC * sr)
    acc_samples = []
    gyr_samples = []
    t_end = time.time() + seconds

    while time.time() < t_end:
        time.sleep(HOP_SEC * 0.9)
        n = board.get_board_data_count(PRESET)
        if n <= 0: continue
        data = board.get_current_board_data(min(n, hop_len), PRESET)
        acc = data[accel_idx, :]
        gyr = data[gyro_idx, :]
        acc_samples.append(acc)
        gyr_samples.append(gyr)

    if not acc_samples:
        raise RuntimeError("No samples captured for neutral.")
    acc_all = np.concatenate(acc_samples, axis=1)
    gyr_all = np.concatenate(gyr_samples, axis=1)

    # Map to logical frame first
    acc_map = _remap_axes(acc_all, "accel")
    gyr_map = _remap_axes(gyr_all, "gyro")

    gravity_vec = np.mean(acc_map, axis=1)   # approx gravity direction
    gyro_bias   = np.mean(gyr_map, axis=1)   # stationary drift
    print("Neutral captured.")
    return gravity_vec, gyro_bias

def _orthonormal_basis_from(gravity_vec, reach_forward_acc_mean=None):
    """
    Build a body-to-logical basis:
      Z_up: opposite of gravity
      X_fwd: from reach_forward mean accel projected to horizontal plane
      Y_right: Z_up x X_fwd
    Returns 3x3 matrix B with columns [X_fwd, Y_right, Z_up] in the mapped coords.
    """
    g = gravity_vec.astype(float)
    Z_up = -g / (np.linalg.norm(g) + 1e-9)

    if reach_forward_acc_mean is None or np.linalg.norm(reach_forward_acc_mean) < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0])  # fallback
        X_fwd = tmp - np.dot(tmp, Z_up) * Z_up
    else:
        r = reach_forward_acc_mean.astype(float)
        r_proj = r - np.dot(r, Z_up) * Z_up
        if np.linalg.norm(r_proj) < 1e-6:
            r_proj = np.array([1.0, 0.0, 0.0])
        X_fwd = r_proj

    X_fwd = X_fwd / (np.linalg.norm(X_fwd) + 1e-9)
    Y_right = np.cross(Z_up, X_fwd)
    Y_right = Y_right / (np.linalg.norm(Y_right) + 1e-9)
    # re-orthogonalize X_fwd
    X_fwd = np.cross(Y_right, Z_up)
    X_fwd = X_fwd / (np.linalg.norm(X_fwd) + 1e-9)
    return np.stack([X_fwd, Y_right, Z_up], axis=1)  # columns

# ---------- motion collection ----------
def _collect(board: BoardShim, sr: int, accel_idx, gyro_idx, seconds: float, prompt: str):
    print("\n=== CALIBRATION ===")
    print(f"Get ready to: {prompt}")
    for i in range(int(REST_SECONDS_BETWEEN), 0, -1):
        print(f"  starting in {i}…", end="\r"); time.sleep(1)
    print("  go!            ")

    win_len = int(WINDOW_SEC * sr)
    hop_len = int(HOP_SEC * sr)
    buf_acc = np.empty((3, 0)); buf_gyro = np.empty((3, 0))
    feats = []
    t_end = time.time() + seconds
    while time.time() < t_end:
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

        acc_win = _remap_axes(buf_acc.copy(), "accel")
        gyr_win = _remap_axes(buf_gyro.copy(), "gyro")
        _ensure_contiguous_filter(acc_win, sr, LOWPASS_ACCEL_CUTOFF)
        _ensure_contiguous_filter(gyr_win, sr, LOWPASS_GYRO_CUTOFF)

        feat = _features(acc_win[0], acc_win[1], acc_win[2],
                         gyr_win[0], gyr_win[1], gyr_win[2], sr)
        feats.append(feat)
    print(f"  captured {len(feats)} windows.")
    return feats

# ---------- thresholds derivation ----------
def derive_thresholds(calib: dict) -> dict:
    """Return a THRESH dict from calibration feature windows."""
    TH = {}

    all_motion = [f['motion_rms'] for seq in calib.values() for f in seq]
    TH['min_motion_mag'] = max(0.05, _pct(all_motion, 15))

    # reach
    reach = calib.get('reach_forward', [])
    if reach:
        TH['reach_ax_mean'] = _pct([f['ax_mean'] for f in reach], 50)
        TH['reach_ax_rms']  = _pct([f['ax_rms']  for f in reach], 50)
    else:
        TH['reach_ax_mean'] = 0.55; TH['reach_ax_rms'] = 0.35

    # swing
    swing = calib.get('swing_down', [])
    if swing:
        TH['swing_neg_az_mean'] = _pct([f['az_mean'] for f in swing], 30)  # more negative
        TH['swing_pitch_rms']   = _pct([f['gy_rms']  for f in swing], 60)
    else:
        TH['swing_neg_az_mean'] = -0.55; TH['swing_pitch_rms'] = 60.0

    # rotation
    rots = calib.get('rotate_left', []) + calib.get('rotate_right', [])
    if rots:
        gz_rms_vals = [f['gz_rms'] for f in rots]
        dom_ratios  = [f['gz_rms'] / max(1e-6, max(f['gx_rms'], f['gy_rms'])) for f in rots]
        TH['yaw_rot_rms']   = _pct(gz_rms_vals, 55)
        TH['yaw_dom_ratio'] = max(1.2, _pct(dom_ratios, 40))
        TH['max_ax_mean_for_rotation']     = max(0.15, _pct([abs(f['ax_mean']) for f in rots], 70))
        TH['max_az_abs_mean_for_rotation'] = max(0.15, _pct([abs(f['az_mean']) for f in rots], 70))
        TH['max_ax_rms_for_rotation']      = max(0.15, _pct([f['ax_rms'] for f in rots], 70))
    else:
        TH.update(dict(
            yaw_rot_rms=80.0, yaw_dom_ratio=1.4,
            max_ax_mean_for_rotation=0.35, max_az_abs_mean_for_rotation=0.35,
            max_ax_rms_for_rotation=0.35,
        ))

    return TH

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", default="thresholds.json", help="Where to save thresholds JSON")
    ap.add_argument("--ip", help="MindRove WiFi board IP (optional)")
    ap.add_argument("--port", type=int, help="MindRove WiFi board port (optional)")
    ap.add_argument("--seconds", type=float, default=CALIB_SECONDS_PER_CLASS, help="Seconds per motion")
    args = ap.parse_args()

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
    print(f"Streaming at {sr} Hz. Starting calibration…")

    try:
        # 1) Neutral capture
        gravity_vec, gyro_bias = _collect_neutral(board, sr, accel_idx, gyro_idx, seconds=3.0)

        # 2) Motions
        calib = {}
        for key, prompt in MOTIONS:
            calib[key] = _collect(board, sr, accel_idx, gyro_idx, args.seconds, prompt)

        # 3) Basis using reach-forward
        reach = calib.get("reach_forward", [])
        reach_acc_mean = None
        if reach:
            rx = float(np.mean([f["ax_mean"] for f in reach]))
            ry = float(np.mean([f["ay_mean"] for f in reach]))
            rz = float(np.mean([f["az_mean"] for f in reach]))
            reach_acc_mean = np.array([rx, ry, rz], dtype=float)
        basis = _orthonormal_basis_from(gravity_vec, reach_acc_mean)

         # 4) Thresholds
        TH = derive_thresholds(calib)

        # 5) Pretty print
        print("\n=== Derived thresholds ===")
        for k, v in TH.items():
            print(f"  {k:>32s}: {v:.3f}")

        # --- NEW: Print summaries per motion category ---
        print("\n=== Per-motion calibration summaries ===")
        for key, feats in calib.items():
            if not feats:
                print(f"  {key:>14s}: no data")
                continue
            ax_means = [f['ax_mean'] for f in feats]
            az_means = [f['az_mean'] for f in feats]
            gx_rms   = [f['gx_rms'] for f in feats]
            gy_rms   = [f['gy_rms'] for f in feats]
            gz_rms   = [f['gz_rms'] for f in feats]

            print(f"  {key:>14s}: "
                  f"ax_mean≈{np.mean(ax_means):+.2f}, "
                  f"az_mean≈{np.mean(az_means):+.2f}, "
                  f"gx_rms≈{np.mean(gx_rms):.1f}, "
                  f"gy_rms≈{np.mean(gy_rms):.1f}, "
                  f"gz_rms≈{np.mean(gz_rms):.1f}")


        # 6) Save payload
        payload = {
            "thresholds": TH,
            "neutral": {
                "gravity_vec": gravity_vec.tolist(),
                "gyro_bias":   gyro_bias.tolist(),
            },
            "basis": {
                "B_columns": basis.tolist()  # 3x3 matrix
            }
        }
        Path(args.outfile).write_text(json.dumps(payload, indent=2))
        print(f"\nSaved thresholds + neutral → {args.outfile}")

    finally:
        try: board.stop_stream()
        except Exception: pass
        board.release_session()

if __name__ == "__main__":
    main()
