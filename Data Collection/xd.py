import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt
import time

def create_bandpass_filter(lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sos

def apply_filter(data, sos):
    return sosfilt(sos, data)

fs = 250  # Hz
duration = 1  # second
samples_per_chunk = int(fs * duration)

# Simulate a 10 Hz sine wave EEG signal + noise
t = np.linspace(0, duration, samples_per_chunk, endpoint=False)

lowcut = 1.0
highcut = 50.0
sos = create_bandpass_filter(lowcut, highcut, fs)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(t, np.zeros_like(t))

while True:
    eeg_chunk = np.sin(2 * np.pi * 10 * t) + 0.1*np.sin(2 * np.pi * 49 * t)
    filtered_chunk = apply_filter(eeg_chunk, sos)
    
    line.set_ydata(filtered_chunk)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(duration)
