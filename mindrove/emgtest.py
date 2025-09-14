# live_emg_monitor.py
# Stream MindRove EMG envelopes (per channel) in real time.
#
# Usage:
#   python live_emg_monitor.py --ip 192.168.4.1 --port 7000 --channels 0,2,4,6

from __future__ import annotations
import argparse, time, sys, numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds, MindRovePresets
from mindrove.data_filter import DataFilter, FilterTypes, NoiseTypes

BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD.value
PRESET   = MindRovePresets.DEFAULT_PRESET.value

BP_LO, BP_HI = 20.0, 450.0
ENV_LP       = 8.0
HOP_SEC      = 0.05

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", help="MindRove WiFi board IP")
    ap.add_argument("--port", type=int, help="MindRove WiFi board port")
    ap.add_argument("--channels", help="Comma-separated EMG indices within EMG set (e.g., 0,2,4,6)")
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

    if args.channels:
        sel = [int(s) for s in args.channels.split(",")]
    else:
        sel = list(range(len(emg_rows)))

    board.start_stream(450000)
    print(f"Streaming EMG envelopes at {fs} Hz. Monitoring channels {sel}… Press Ctrl+C to stop.")

    hop_len = max(1, int(HOP_SEC * fs))

    try:
        while True:
            n = board.get_board_data_count(PRESET)
            if n <= 0:
                time.sleep(HOP_SEC*0.9)
                continue

            data = board.get_current_board_data(min(n, hop_len), PRESET)
            emg = data[emg_rows, :]
            env = envelope(emg, fs)   # [ch x T]

            # Take last sample for display
            cur = env[:, -1]
            sys.stdout.write("\r" + " | ".join([f"ch{c}={cur[c]:.4f}" for c in sel]) + "   ")
            sys.stdout.flush()

            time.sleep(HOP_SEC)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        try: board.stop_stream()
        except Exception: pass
        board.release_session()

if __name__ == "__main__":
    main()
