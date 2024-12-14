import tensorflow as tf
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import butter, sosfilt, ShortTimeFFT, get_window
from PIL import Image
import wfdb
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import config
import gc

# Have to place this outside of the class inorder to use multiprocessing on it
def process_file(args):
    index, root_path, f, leads, sos, sft = args
    data = wfdb.rdsamp(os.path.join(root_path, os.path.splitext(f)[0]), channel_names=leads, return_res=32)[0].T
    filtered = sosfilt(sos, data, axis=-1)
    spectrogram = sft.spectrogram(filtered, axis=-1)
    tensor = np.transpose(spectrogram, (1, 2, 0)).astype(np.float32)
    return [index, tensor]

# Class for creatinng dataset
class PTBXLDataset(object):
    def __init__(self, cfg, meta_file, root_path):
        self.cfg = cfg
        self.meta_file = meta_file  # Full path to PTBXL database file
        self.root_path = root_path  # Full path to the dataset folder

        self.dataset = pd.read_json(os.path.join(root_path, meta_file))
        self.dataset = self.dataset[['filename_lr', 'filename_hr', 'AD']]

        self.leads = ['I', 'II', 'V2']
        self.classes = ['AD']
        self.sampling_rate = 100 # or 500

        self.storage_dir = 'ptbxl_tfrecords'

        # Define the filter
        lowcut = 1.0          # Lower cutoff frequency (Hz)
        highcut = 45.0        # Upper cutoff frequency (Hz)
        fs = 100.0            # Sampling frequency (Hz)
        order = 4             # Filter order (higher = steeper rolloff)
        self.sos = self.design_iir_bandpass(lowcut, highcut, fs, order)

        # Define a spectrogram
        nperseg = fs * 1    # Length of each segment in seconds
        hop = 2             # Similar to stride
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
        
    def create_splits(self, multilabel=False):
        """
        Creates train, validation, and test splits with stratification.
        """
        X = self.dataset[['filename_lr', 'filename_hr']]
        y = self.dataset[self.classes]

        # Choose the appropriate splitter
        if multilabel:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=self.cfg.TEST_PERCENTAGE, random_state=0
            )
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.cfg.TEST_PERCENTAGE, random_state=0
            )

        # First, split into train+validation and test
        train_val_indices, test_indices = next(splitter.split(X, y))

        # Isolate train+validation data
        X_train_val, y_train_val = X.iloc[train_val_indices], y.iloc[train_val_indices]

        # Now split train+validation into train and validation
        if multilabel:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=self.cfg.VALIDATE_PERCENTAGE / (1 - self.cfg.TEST_PERCENTAGE), random_state=0
            )
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.cfg.VALIDATE_PERCENTAGE / (1 - self.cfg.TEST_PERCENTAGE), random_state=0
            )

        train_indices, validation_indices = next(splitter.split(X_train_val, y_train_val))

        # Map back to original indices
        train_indices = train_val_indices[train_indices]
        validation_indices = train_val_indices[validation_indices]

        # Assertions to ensure no overlap
        assert not set(train_indices).intersection(validation_indices), "Train and validation overlap!"
        assert not set(train_indices).intersection(test_indices), "Train and test overlap!"
        assert not set(validation_indices).intersection(test_indices), "Validation and test overlap!"

        # Assign splits
        self.train_df = pd.concat([X.iloc[train_indices], y.iloc[train_indices]], axis=1)
        self.validate_df = pd.concat([X.iloc[validation_indices], y.iloc[validation_indices]], axis=1)
        self.test_df = pd.concat([X.iloc[test_indices], y.iloc[test_indices]], axis=1)

        del X, y
        
    def load_batch(self, df):
        '''
        Loads signals from the signal database into tensors
        '''
        df_files = df.filename_lr if self.sampling_rate == 100 else df.filename_hr

        args = [
            (i, self.root_path, f, self.leads, self.sos, self.sft)
            for i, f in enumerate(df_files)
        ]

        with ProcessPoolExecutor() as executor:
            data = list(tqdm(executor.map(process_file, args), total=len(df_files), desc="Processing files"))

        # Ensure that the ordering of the data is correct.
        data = sorted(data, key=lambda x: x[0])
        data = [d[1] for d in data]

        labels = df[self.classes].to_numpy().astype(np.int32)
        return data, labels
    
    def give_raw_dataset(self, mode):
        '''
        Creates tensorflow datasets for each  train, validation, and test splits
        '''

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
        '''
        Input:
            dataset:        tf.Dataset of data to write.
            file_prefix:    Name of the file to write the data.
        '''
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

    def read_tfrecords(self, file_name, buffer_size=1000):
        '''
        Input:
            file_name:  File name to read records from.
        Output:
            dataset:    TFRecordDataset.
        '''
        
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
        
        dataset = [file_name]
        data = tf.data.TFRecordDataset(dataset, buffer_size=buffer_size)
        dataset = data.map(_parse_function)

        return dataset
    
    def write_yolo_data(self, tensor_slices, output_dir):
        '''
        Input:
            tensor_slices:  tf.Dataset of (tensor, labels)
            output_dir:     directory to write images and labels
        '''
        # Create directories
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for i, (tensor, label) in enumerate(tensor_slices):
            # Normalize tensor to [0, 255]
            normalized_tensor = tf.cast((tensor / tf.reduce_max(tensor)) * 255, tf.uint8)

            # Save the image
            image_path = os.path.join(images_dir, f"image{i+1}.png")
            image = Image.fromarray(normalized_tensor.numpy())
            image.save(image_path)

            # Save the label (binary classification)
            label_path = os.path.join(labels_dir, f"image{i+1}.txt")
            with open(label_path, "w") as f:
                f.write(f"{label}\n")  # YOLO expects the class index

if __name__ == "__main__":
    cfg = config.Configuration()
    file_name = 'updated_ptbxl_database.json'
    root_path = '/home/lrbutler/Desktop/ECGSignalClassifer/ptb-xl/'

    dataset = PTBXLDataset(cfg=cfg, meta_file=file_name, root_path=root_path)

    dataset.create_splits()

    validate = dataset.give_raw_dataset(mode='validate')
    dataset.write_yolo_data(validate, 'val')
    del validate
    tf.keras.backend.clear_session()
    gc.collect()

    train = dataset.give_raw_dataset(mode='train')
    dataset.write_yolo_data(train, 'train')
    del train
    tf.keras.backend.clear_session()
    gc.collect()

    test = dataset.give_raw_dataset(mode='test')
    dataset.write_yolo_data(test, 'test')
    del test
    tf.keras.backend.clear_session()
    gc.collect()