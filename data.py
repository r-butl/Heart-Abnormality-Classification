import tensorflow as tf
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from scipy.signal import butter, sosfilt, sosfreqz, ShortTimeFFT, get_window
import wfdb
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import config

# Have to place this outside of the class inorder to use multiprocessing on it
def process_file(args):
    root_path, f, leads, sos, sft = args
    data = wfdb.rdsamp(os.path.join(root_path, os.path.splitext(f)[0]), channel_names=leads, return_res=32)[0].T
    filtered = sosfilt(sos, data, axis=-1)
    spectrogram = sft.spectrogram(filtered, axis=-1)
    return np.transpose(spectrogram, (1, 2, 0))

# Class for creatinng dataset
class PTBXLDataset(object):
    def __init__(self, cfg, meta_file, root_path):
        self.cfg = cfg
        self.meta_file = meta_file  # Full path to PTBXL database file
        self.root_path = root_path  # Full path to the dataset folder

        self.dataset = pd.read_json(os.path.join(root_path, meta_file))
        self.dataset = self.dataset[['filename_lr', 'filename_hr', 'MI', 'STTC', 'CD', 'HYP', 'AD']]

        self.leads = ['I', 'II', 'V2']
        self.classes = ['MI', 'STTC', 'CD', 'HYP', 'AD']
        self.sampling_rate = 100 # or 500

        # Define the filter
        lowcut = 1.0          # Lower cutoff frequency (Hz)
        highcut = 45.0        # Upper cutoff frequency (Hz)
        fs = 100.0            # Sampling frequency (Hz)
        order = 4             # Filter order (higher = steeper rolloff)
        self.sos = self.design_iir_bandpass(lowcut, highcut, fs, order)

        # Define a spectrogram
        nperseg = fs * 1  # Length of each segment, 2 seconds in this case
        hop = 3  # Overlap between segments, 50% hop in this case
        w = get_window(('gaussian', 2), nperseg)
        self.sft = ShortTimeFFT(w, hop, fs=fs, mfft=150)

        self.create_splits()

    def design_iir_bandpass(self, lowcut, highcut, fs, order=4):
        '''
        Designs the bandpass filer
        '''
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], btype='band', output='sos')  # Second-order sections
        return sos
        
    def create_splits(self):
        '''
        Uses mutlilabel stratification to split the data
        '''
        # Extract X and y
        X = self.dataset[['filename_lr', 'filename_hr']]
        y = self.dataset[['MI', 'STTC', 'CD', 'HYP', 'AD']]

        # Train/Test split
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.cfg.TEST_PERCENTAGE, random_state=0)
        train_indices, test_indices = next(splitter.split(X, y))

        # Train/Validation split
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.cfg.VALIDATE_PERCENTAGE / (1 - self.cfg.TEST_PERCENTAGE), random_state=0)
        train_indices, validation_indices = next(splitter.split(X.iloc[train_indices], y.iloc[train_indices]))

        # Assign splits
        self.train_df = pd.concat([X.iloc[train_indices], y.iloc[train_indices]], axis=1)
        self.validate_df = pd.concat([X.iloc[validation_indices], y.iloc[validation_indices]], axis=1)
        self.test_df = pd.concat([X.iloc[test_indices], y.iloc[test_indices]], axis=1)

        del X
        del y
        del self.dataset

    def load_batch(self, df):
        '''
        Loads signals from the signal database into tensors
        '''
        df_files = df.filename_lr if self.sampling_rate == 100 else df.filename_hr

        args = [
            (self.root_path, f, self.leads, self.sos, self.sft)
            for f in df_files
        ]

        with ProcessPoolExecutor() as executor:
            data = list(tqdm(executor.map(process_file, args), total=len(df_files), desc="Processing files"))

        labels = df[self.classes].to_numpy()
        
        return data, labels
    
    def give_data(self, mode):
        '''
        Creates tensorflow datasets for each  train, validation, and test splits
        '''
        dataset = None
        if mode == 'train':
            dataset = self.train_df
        elif mode == 'test':
            dataset = self.test_df
        else:
            dataset = self.validate_df

        data, labels = self.load_batch(dataset)

        labels = tf.cast(labels, tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((data , labels)).batch(self.cfg.BATCH_SIZE)
        
        return dataset

if __name__ == "__main__":
    cfg = config.Configuration()
    file_name = 'updated_ptbxl_database.json'
    root_path = '/home/lrbutler/Desktop/ECGSignalClassifer/ptb-xl/'

    dataset = PTBXLDataset(cfg=cfg, meta_file=file_name, root_path=root_path)
    train = dataset.give_data(mode='train')
    test = dataset.give_data(mode='test')
    validate = dataset.give_data(mode='validate')

    print(len(train))
    print(len(test))
    print(len(validate))