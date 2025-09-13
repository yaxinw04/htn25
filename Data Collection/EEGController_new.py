import time
import atexit
import numpy as np
import scipy.io as sio
from scipy import signal
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, lfilter, iirnotch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

class EEGController:

    # Initializer
    def __init__(self, debug=False):

        # Setting up the board
        if debug:
            # Create a synthetic board for debugging purposes
            self.board_id = BoardIds.SYNTHETIC_BOARD
            self.params = BrainFlowInputParams()
        else:
            # Setting up the board
            self.params = BrainFlowInputParams()
            self.params.serial_number = 'UN-2023.02.30'
            self.board_id = BoardIds.UNICORN_BOARD
            
        self.board = BoardShim(self.board_id, self.params)

        # Getting specific board details
        self.channels = self.board.get_eeg_channels(self.board_id) #EEG Channels
        self.timestamp_channel = self.board.get_timestamp_channel(self.board_id) # Timestamp channel
        self.marker_channel = self.board.get_marker_channel(self.board_id) # Marker channel for synchronization
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id) # Hz

        # Start recording/collecting data
        self.board.prepare_session()
        self.board.start_stream ()
        print(f"Starting EEG Stream, debug={debug}")
        atexit.register(self.close)
        
        # # Populate the board, will crash if not
        # time.sleep(0.2)
        # start = self.board.get_current_board_data(200)

        # # Get timestamp data
        # self.initial_time = start[self.timestamp_channel,0] #Get the initial timestamp data
        
        # Create an active variable to ensure proper closing
        self.active = True
        
        # Filter properties
        self.cutoff_high = 0.5
        self.cutoff_low = 45
        
        # SSVEP Properties
        self.epoch_timelength = 4 # Seconds
        self.epoch_samples =  self.epoch_timelength * self.sampling_rate
        self.labels = [9, 12, 8]
        self.labels_phaseshift = [0, 0.7, 1.5]
        self.train = True
        self.reference_signals = []
        self.templates = []
        for i in range(len(self.labels)):
            self.reference_signals.append(self.CCAReferenceSignal(self.labels[i], self.labels_phaseshift[i], 2))


    '''
    Simple Preprocessing Filters
    '''
    def butter_filter(data, cutoff, fs, btype='low', order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        y = lfilter(b, a, data, axis=0)
        return y

    def notch_filter(signal, fs, freq=50, Q=30):
        b, a = iirnotch(freq, Q, fs)
        return lfilter(b, a, signal, axis=0)
    
        

    # Return timestamp data
    def setMarker(self, start=False):
        if start == True:
            self.board.insert_marker(420)
        else:
            self.board.insert_marker(469)
    

    def close(self):
        if self.active == True:
            # Get EEG data from board and stops EEG session
            data = self.board.get_board_data()
            self.board.stop_stream()
            self.board.release_session()
            
            np.savetxt("eeg_data.csv", np.transpose(data), delimiter=",")
            print("Saved File")
            self.active = False