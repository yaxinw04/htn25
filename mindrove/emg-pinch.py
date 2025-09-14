# detect_pinch_emg.py  (more permissive + very verbose)
from __future__ import annotations
import argparse, time, sys, numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
from mindrove.data_filter import DataFilter, FilterTypes, NoiseTypes

BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET   = MindRovePresets.DEFAULT_PRESET.value

# Signal processing
BP_LO, BP_HI = 20.0, 450.0
ENV_LP       = 8.0
WINDOW_SEC   = 0.200
HOP_SEC      = 0.050

# Detection params (more permissive)
DEFAULT_K_STD  = 1.8    # lower than before
DEFAULT_PCTL   = 90.0   # lower backup percentile
MIN_HOLD_SEC   = 0.06   # allow faster pinches
REFRACT_SEC    = 0.45
COACT_RATIOMAX = 3.5    # more forgiving

def butter_inplace(x: np.ndarray, fs: int, kind: str):
    for r in range(x.shape[0]):
        if kind == "bp":
            DataFilter.perform_bandpass(x[r], fs, BP_LO, BP_HI, 3, FilterTypes.BUTTERWORTH.value, 0.0)
        elif kind == "lp":
            DataFilter.perform_lowpass(x[r], fs, ENV_LP, 3, FilterTypes.BUTTERWORTH.value, 0.0)

def notch_inplace(x: np.ndarray, fs: int):
    for r in range(x.shape[0]):
        DataFilter.remove_environmental_noise(x[r], fs, NoiseTypes.FIFTY.value)
        DataFilter.remove_environmental_noise(x[r], fs, NoiseTypes.SIXTY.value)

def envelope(emg: np.ndarray, fs: int) -> np.ndarray:
    z = emg.astype(np.float64, copy=True)
    butter_inplace(z, fs, "bp")
    notch_inplace(z, fs)
    z = np.abs(z)
    butter_inplace(z, fs, "lp")
    return z

def summarize_rest(env: np.ndarray, pctl_backup: float):
    mean = np.mean(env, axis=1)
    std  = np.std(env, axis=1, ddof=1) + 1e-9
    pctl = np.percentile(env, pctl_backup, axis=1)
    return mean, std, pctl

def auto_pick_channels(env_stream: list[np.ndarray], top_k: int = 2) -> list[int]:
    if not env_stream:
        return []
    env_all = np.concatenate(env_stream, axis=1)
    med = np.median(env_all, axis=1)
    p95 = np.percentile(env_all, 95, axis=1)
    score = p95 - med
    idx = np.argsort(score)[::-1]
    idx = [i for i in idx[:top_k] if score[i] > np.max(score) * 0.2]
    return idx if idx else [int(np.argmax(score))]

def live_line(vals, thrs, prefix=""):
    comps = [f"{v:.4f}/{t:.4f}" for v,t in zip(vals, thrs)]
    return prefix + "  ".join(comps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", help="MindRove WiFi board IP")
    ap.add_argument("--port", type=int, help="MindRove WiFi board port")
    ap.add_argument("--channels", help="Comma-separated EMG indices within EMG set (e.g., 0,2,4,6)")
    ap.add_argument("--no-demo", action="store_true", help="Skip pinch demo auto-channel selection")
    ap.add_argument("--kstd", type=float, default=DEFAULT_K_STD, help="Threshold factor over baseline std")
    ap.add_argument("--pctl", type=float, default=DEFAULT_PCTL, help="Backup rest percentile (90–97)")
    ap.add_argument("--rest-sec", type=float, default=8.0, help="Rest calibration seconds")
    ap.add_argument("--demo-sec", type=float, default=8.0, help="Pinch demo seconds (if not --no-demo)")
    ap.add_argument("--show", action="store_true", help="Print live per-channel value/threshold")
    ap.add_argument("--no-z", action="store_true", help="Disable z-score gate")
    ap.add_argument("--no-coact", action="store_true", help="Disable co-activation ratio check")
    ap.add_argument("--mode", choices=["any","majority"], default="any",
                    help="Trigger if ANY selected channel exceeds thr (default) or MAJORITY")
    args = ap.parse_args()

    BoardShim.enable_board_logger()
    params = MindRoveInputParams()
    if args.ip:   params.ip_address = args.ip
    if args.port: params.ip_port    = args.port

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()

    fs = BoardShim.get_sampling_rate(BOARD_ID, PRESET)
    emg_rows = BoardShim.get_emg_channels(BOARD_ID, PRESET)
    if not emg_rows:
        raise RuntimeError("This board/preset exposes no EMG channels.")

    # Select channels (within EMG set) — lock to 0,2,4,6 if you pass them
    if args.channels:
        use_idx = [int(s) for s in args.channels.split(",")]
        print(f"Using specified EMG indices (within EMG set): {use_idx}")
    else:
        use_idx = None  # decide after demo or use all

    board.start_stream(450000)
    print(f"Streaming EMG at {fs} Hz from {len(emg_rows)} channels.")
    print(f"Calibrating REST baseline for {args.rest_sec:.1f}s…")

    hop_len = max(1, int(HOP_SEC * fs))
    win_len = max(4, int(WINDOW_SEC * fs))

    def pull_env(samples: int) -> np.ndarray:
        n = board.get_board_data_count(PRESET)
        if n <= 0:
            time.sleep(HOP_SEC*0.9)
            return np.zeros((len(emg_rows), 0))
        data = board.get_current_board_data(min(n, samples), PRESET)
        return envelope(data[emg_rows, :], fs)

    try:
        # 1) Rest
        rest_env = []
        t_end = time.time() + float(args.rest_sec)
        while time.time() < t_end:
            e = pull_env(hop_len)
            if e.size: rest_env.append(e)
        rest_all = np.concatenate(rest_env, axis=1) if rest_env else np.zeros((len(emg_rows),1))
        base_mean, base_std, base_pctl = summarize_rest(rest_all, args.pctl)

        # 2) Demo for channel auto-pick (unless channels were provided)
        if use_idx is None:
            if args.no_demo:
                use_idx = list(range(len(emg_rows)))
                print(f"Using ALL EMG channels: {use_idx}")
            else:
                print(f"Show ~3 clear PINCHES in the next {args.demo_sec:.1f}s…")
                demo = []
                t_end = time.time() + float(args.demo_sec)
                while time.time() < t_end:
                    e = pull_env(hop_len); 
                    if e.size: demo.append(e)
                use_idx = auto_pick_channels(demo, top_k=2)
                print(f"Auto-selected EMG indices: {use_idx}")

        # Per-channel thresholds (selected channels use their own baselines)
        abs_thr_all = np.maximum(base_mean + args.kstd * base_std, base_pctl)

        print("\nBaseline (mean ± std | pctl) and thresholds (all EMG rows):")
        for i in range(len(emg_rows)):
            star = " *" if i in use_idx else "  "
            print(f"ch[{i}]{star} mean={base_mean[i]:.5f}  std={base_std[i]:.5f}  "
                  f"p{int(args.pctl)}={base_pctl[i]:.5f}  thr={abs_thr_all[i]:.5f}")

        print("\nDetecting… Press Ctrl+C to stop.\n")
        last_event_t = 0.0
        over_start_t = None
        buf_env = np.zeros((len(emg_rows), 0))

        while True:
            e = pull_env(hop_len)
            if e.size == 0: 
                continue
            buf_env = np.concatenate([buf_env, e], axis=1)
            if buf_env.shape[1] > win_len:
                buf_env = buf_env[:, -win_len:]
            if buf_env.shape[1] < win_len:
                continue

            sel_env = buf_env[use_idx, :]
            cur = sel_env[:, -1]
            thr = abs_thr_all[use_idx]
            z = (cur - base_mean[use_idx]) / (base_std[use_idx] + 1e-9)

            # Live print of selected channels: value/threshold
            if args.show:
                sys.stdout.write("\r" + live_line(cur, thr, prefix="vals/thr: ") + "   ")
                sys.stdout.flush()

            # Channel-wise over checks
            over = cur > thr
            if not args.no_z:
                over = over & (z >= 1.2)  # softer z-gate than before

            if args.mode == "any":
                is_over = np.any(over)
            else:  # majority
                is_over = (np.count_nonzero(over) >= max(1, len(use_idx)//2))

            if not args.no_coact and len(use_idx) >= 2:
                top2 = np.sort(cur)[-2:]
                ratio = (top2[-1] / (top2[-2] + 1e-9))
                is_over = is_over and (ratio <= COACT_RATIOMAX)

            now = time.time()
            if is_over and over_start_t is None:
                over_start_t = now
            if not is_over:
                over_start_t = None

            if over_start_t and (now - over_start_t >= MIN_HOLD_SEC) and (now - last_event_t >= REFRACT_SEC):
                last_event_t = now
                print(f"\n[{time.strftime('%H:%M:%S')}] PINCH detected  chans={use_idx}  "
                      f"vals={np.round(cur,4).tolist()}  thr={np.round(thr,4).tolist()}")

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        try: board.stop_stream()
        except Exception: pass
        board.release_session()

if __name__ == "__main__":
    main()
