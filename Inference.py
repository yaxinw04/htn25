import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.signal import butter, lfilter, iirnotch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import matplotlib.pyplot as plt
import time
from pynput.keyboard import Key, Controller

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, hidden):

        seq_len = encoder_outputs.size(1)
        h_tiled = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, h_tiled), dim=2)))
        attn_weights = torch.softmax(self.v(energy), dim=1)
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)

        return context_vector

class DPARSEncoder(nn.Module):
    def __init__(self, input_size, encode_size, num_classes):
        super(DPARSEncoder, self).__init__()
        second_size = input_size*2
        
        self.bn1 = nn.BatchNorm1d(second_size)
        self.bn2 = nn.BatchNorm1d(encode_size)
        self.dropout = nn.Dropout(p=0.3)
        self.attention = Attention(2*encode_size)

        self.depthwise1 = nn.Conv1d(
            input_size, input_size, kernel_size=64, stride = 16, groups=input_size, bias=False)
        self.pointwise1 = nn.Conv1d(input_size, second_size, kernel_size=1)


        self.depthwise2 = nn.Conv1d(
            second_size, second_size, kernel_size=16, stride = 4, groups=second_size, bias=False)
        self.pointwise2 = nn.Conv1d(second_size, encode_size, kernel_size=1)

        self.fc1 = nn.Linear(encode_size, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):

        # Input -> (batch_size, 1000, 8)
        x = x.permute(0, 2, 1)
        x = self.depthwise1(x)  # output: (batch, 64, 48)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # print(x.shape)


        x = self.depthwise2(x)  # output: (batch, 64, 48)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x = self.attention(x, x[:, -1, :])
        # print(x.shape)

        return self.fc1(x)
    
def butter_filter(data, cutoff, fs, btype='low', order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = lfilter(b, a, data, axis=0)
    return y

def notch_filter(signal, fs, freq=50, Q=30):
    b, a = iirnotch(freq, Q, fs)
    return lfilter(b, a, signal, axis=0)


keyboard = Controller()

def send_key(label):
    if label == 0:
        print("Relax")
        keyboard.release(Key.space)
        keyboard.release("w")
        pass
    elif label == 1:
        keyboard.press("w")
        time.sleep(1)
        print("Pressed: w")
    elif label == 2:
        keyboard.press(Key.space)
        print("Pressed: space")


params = BrainFlowInputParams()
params.serial_number = 'UN-2023.02.30'
board_id = BoardIds.UNICORN_BOARD

board = BoardShim(board_id, params)
channels = board.get_eeg_channels(board_id) #EEG Channels
timestamp_channel = board.get_timestamp_channel(board_id) # Timestamp channel
marker_channel = board.get_marker_channel(board_id) # Marker channel for synchronization
sampling_rate = BoardShim.get_sampling_rate(board_id) # Hz
    
board.prepare_session()
board.start_stream()


window_size = int(4*sampling_rate)
buffer = 750

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DPARSEncoder(8, 16, 3).to(device)
model.load_state_dict(torch.load(r"Data Collection/model.pth"))
time.sleep(5)
model.eval()
with torch.no_grad():
    while True:
        temp_emg = board.get_current_board_data(window_size + buffer)[channels, :]
        temp_emg = temp_emg.T
        # print(temp_emg.shape)

        # Preprocess EMG
        temp_emg = butter_filter(temp_emg, 3, sampling_rate, btype='high', order=2)
        temp_emg = butter_filter(temp_emg, 80, sampling_rate, btype='low', order=2)
        temp_emg = notch_filter(temp_emg, sampling_rate, freq=60)

        temp_emg = temp_emg[buffer:, :]
        temp_emg -= np.mean(temp_emg, axis=1, keepdims=True)
        temp_mean = np.mean(temp_emg, axis=0, keepdims=True)
        temp_std = np.std(temp_emg, axis=0, keepdims=True)
        temp_emg = (temp_emg - temp_mean) / temp_std

        # Get the EEG data and corresponding label
        temp_emg = torch.tensor(temp_emg, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(temp_emg)
        
        predictions = outputs.argmax(dim=1, keepdim=True)
        pred = predictions.cpu().detach().numpy()[0][0]
        
        # if pred == 0:
        #     print("Relax")
        # if pred == 1:
        #     print("Forward")
        # if pred == 2:
        #     print("Jump")

        send_key(pred)

        time.sleep(0.1)

                   