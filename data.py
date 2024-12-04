import tensorflow as tf
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from scipy.signal import butter, sosfilt, ShortTimeFFT, get_window
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
    tensor = np.transpose(spectrogram, (1, 2, 0)).astype(np.float32)
    return tensor

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

        self.storage_dir = 'ptbxl_tfrecords'

        # Define the filter
        lowcut = 1.0          # Lower cutoff frequency (Hz)
        highcut = 45.0        # Upper cutoff frequency (Hz)
        fs = 100.0            # Sampling frequency (Hz)
        order = 4             # Filter order (higher = steeper rolloff)
        self.sos = self.design_iir_bandpass(lowcut, highcut, fs, order)

        # Define a spectrogram
        nperseg = fs * 1    # Length of each segment, 2 seconds in this case
        hop = 2             # Overlap between segments, 50% hop in this case
        w = get_window(('gaussian', 15), nperseg)
        self.sft = ShortTimeFFT(w, hop, fs=fs, mfft=150)

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

        labels = df[self.classes].to_numpy().astype(np.int32)
        return data, labels
    
    def give_raw_dataset(self, mode):
        '''
        Creates tensorflow datasets for each  train, validation, and test splits
        '''
        self.create_splits()

        dataset = None
        if mode == 'train':
            dataset = self.train_df
        elif mode == 'test':
            dataset = self.test_df
        elif mode == 'validate':
            dataset = self.validate_df
        else:
            return

        data, labels = self.load_batch(dataset)
        dataset = tf.data.Dataset.from_tensor_slices((data , labels))
        return dataset
    
    def write_tfrecords(self, dataset, file_prefix):
        """Write data and labels to a TFRecord file."""
        filename = f'{file_prefix}.tfrecord'
        with tf.io.TFRecordWriter(filename) as writer:
            for sample, label in dataset:

                sample = tf.convert_to_tensor(sample)
                label = tf.convert_to_tensor(label)

                feature = {
                    'sample': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(sample).numpy()])),  
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label).numpy()]))
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())

    def read_tfrecords(self, mode, buffer_size):
        
        features = {
            'sample': tf.io.FixedLenFeature([], tf.string),  
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        
        def _parse_function(example_proto):
            """Parse a serialized Example."""
            parsed = tf.io.parse_single_example(example_proto, features)
            # Deserialize tensors
            sample = tf.io.parse_tensor(parsed['sample'], out_type=tf.float32)
            label = tf.io.parse_tensor(parsed['label'], out_type=tf.int32)

            return sample, label
        
        if mode == 'train':
            dataset = [os.path.join(self.storage_dir, 'train_dataset.tfrecord')]
        elif mode == 'test':
            dataset = [os.path.join(self.storage_dir, 'test_dataset.tfrecord')]
        elif mode == 'validate':
            dataset = [os.path.join(self.storage_dir, 'validate_dataset.tfrecord')]
        else:
            return

        data = tf.data.TFRecordDataset(dataset, buffer_size=buffer_size)
        dataset = data.map(_parse_function)

        return dataset

if __name__ == "__main__":
    cfg = config.Configuration()
    file_name = 'updated_ptbxl_database.json'
    root_path = '/home/lrbutler/Desktop/ECGSignalClassifer/ptb-xl/'

    dataset = PTBXLDataset(cfg=cfg, meta_file=file_name, root_path=root_path)

    # validate = dataset.give_raw_dataset(mode='validate')
    # dataset.write_tfrecords(validate, 'validate_dataset')

    train = dataset.give_raw_dataset(mode='train')
    dataset.write_tfrecords(train, 'train_dataset')

    test = dataset.give_raw_dataset(mode='test')
    dataset.write_tfrecords(test, 'test_dataset')

    data = dataset.read_tfrecords('validate')

    for sample, label in data.take(1):  # `take(1)` retrieves the first batch/example
        print(f"Sample Shape: {sample.shape}")
        print(f"Label Shape: {label.shape}")
        