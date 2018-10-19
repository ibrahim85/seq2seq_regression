from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np

from pathlib import Path


slim = tf.contrib.slim

dim_label = 28
dim_raw_audio = 1470
dim_mfcc = 20
dim_melspecs = 128
dim_spec_centr = 1

num_fms = 21
    
def decode_and_reshape(features, name, num_features):
    
    dec_var = tf.decode_raw(features[name], tf.float32)
    dec_var = tf.reshape(dec_var, (num_features, -1))
    dec_var = tf.cast(tf.transpose(dec_var, (1,0)), tf.float32)
    
    return dec_var

def noisy_decode_and_reshape(features, name, dim_features, seq_length,squeeze=False):
    
    dec_var = tf.decode_raw(features[name], tf.float32)
    dec_var = tf.reshape(dec_var, (seq_length, dim_features, num_fms))
    dec_var = tf.cast(tf.transpose(dec_var, (0,2,1)), tf.float32)
    
    return dec_var
    
def get_split(batch_size=32, is_training=True, split_name='train'):
    """Returns a data split of the BBC dataset.
    
    Args:
        batch_size: the number of batches to return.
        is_training: whether to shuffle the dataset.
        split_name: A train/test/valid split name.
    Returns:
        raw_audio: the raw audio examples.
        mfcc: the mfcc features. 
        label: the 3d components of each word. 
        num_examples: the number of audio samples in the set.
        word: the current word.
    """
    num_classes = 28
    base_path = Path('/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean/')
    if split_name == 'train':
        paths = np.loadtxt(str(base_path / 'train_set.csv'), dtype='<U150').tolist()
        print('Training examples : ', len(paths))
    elif split_name == 'devel':
        paths = np.loadtxt(str(base_path / 'valid_set.csv'), dtype='<U150').tolist()
        paths = paths[:10000]
        print('Evaluating examples : ', len(paths))
    elif split_name == 'test':
        paths = np.loadtxt(str(base_path / 'test_set.csv'), dtype='<U150').tolist()
        print('Testing examples : ', len(paths))
    else:
        raise TypeError('split_name should be one of [train], [devel] or [test]')
    
    num_examples = len(paths)
    
    filename_queue = tf.train.string_input_producer(paths, shuffle=is_training)
    
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)
    
    features = tf.parse_single_example(
        serialized_example,
        features={
            'subject_id': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
            'raw_audio': tf.FixedLenFeature([], tf.string),
            'frame_mfcc_overlap': tf.FixedLenFeature([], tf.string),
            'frame_mfcc': tf.FixedLenFeature([], tf.string),
            'delta_frame_mfcc': tf.FixedLenFeature([], tf.string),
            'delta2_frame_mfcc': tf.FixedLenFeature([], tf.string),
            'rmse': tf.FixedLenFeature([], tf.string),
            'frame_melspectrogram': tf.FixedLenFeature([], tf.string),
            'delta_frame_melspectrogram': tf.FixedLenFeature([], tf.string),
            'delta2_frame_melspectrogram': tf.FixedLenFeature([], tf.string),
            'frame_melspectrogram_overlap':tf.FixedLenFeature([], tf.string),
            'frame_spectral_centroid':tf.FixedLenFeature([], tf.string),
            'delta_frame_spectral_centroid':tf.FixedLenFeature([], tf.string),
            'delta2_frame_spectral_centroid':tf.FixedLenFeature([], tf.string),
            
            'noisy_frame_mfcc': tf.FixedLenFeature([], tf.string),
            'delta_noisy_frame_mfcc': tf.FixedLenFeature([], tf.string),
            'delta2_noisy_frame_mfcc': tf.FixedLenFeature([], tf.string),
            
            'noisy_frame_spectral_centroid': tf.FixedLenFeature([], tf.string),
            'delta_noisy_frame_spectral_centroid': tf.FixedLenFeature([], tf.string),
            'delta2_noisy_frame_spectral_centroid': tf.FixedLenFeature([], tf.string),
            
            'noisy_frame_melspectrogram': tf.FixedLenFeature([], tf.string),
            'delta_noisy_frame_melspectrogram': tf.FixedLenFeature([], tf.string),
            'delta2_noisy_frame_melspectrogram': tf.FixedLenFeature([], tf.string),
            
            'noisy_frame_rmse': tf.FixedLenFeature([], tf.string),
            'delta_noisy_frame_rmse': tf.FixedLenFeature([], tf.string),
            'delta2_noisy_frame_rmse': tf.FixedLenFeature([], tf.string),
            
            'word': tf.FixedLenFeature([], tf.string)
        }
    )
    
    subject_id = features['subject_id']
    label = tf.decode_raw(features['labels'], tf.float32)
    label = tf.reshape(label, (-1, num_classes))
    
    raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    raw_audio = tf.reshape(raw_audio, ([1, -1]))
    
    frame_mfcc = decode_and_reshape(features, 'frame_mfcc', dim_mfcc) 
    
    # frame mfcc overlap
    frame_mfcc_overlap = decode_and_reshape(features, 'frame_mfcc_overlap', dim_mfcc)
    delta_frame_mfcc = decode_and_reshape(features, 'delta_frame_mfcc', dim_mfcc)
    delta2_frame_mfcc = decode_and_reshape(features, 'delta2_frame_mfcc', dim_mfcc)
    
    # frame melspectrogram
    frame_melspectrogram = decode_and_reshape(features, 'frame_melspectrogram', dim_melspecs)
    delta_frame_melspectrogram = decode_and_reshape(features, 'delta_frame_melspectrogram', dim_melspecs)
    delta2_frame_melspectrogram = decode_and_reshape(features, 'delta2_frame_melspectrogram', dim_melspecs)
    
    frame_melspectrogram_overlap = decode_and_reshape(features, 'frame_melspectrogram_overlap', dim_melspecs)
    
    # frame spectral centroid
    frame_spectral_centroid = decode_and_reshape(features, 'frame_spectral_centroid', dim_spec_centr)
    delta_frame_spectral_centroid = decode_and_reshape(features, 'delta_frame_spectral_centroid', dim_spec_centr)
    delta2_frame_spectral_centroid = decode_and_reshape(features, 'delta2_frame_spectral_centroid', dim_spec_centr)
    
    seq_length = -1#tf.shape(frame_mfcc)[0]
    # noisy frame mfcc
    noisy_frame_mfcc = noisy_decode_and_reshape(features, 'noisy_frame_mfcc', dim_mfcc, seq_length)
    delta_noisy_frame_mfcc = noisy_decode_and_reshape(features, 'delta_noisy_frame_mfcc', dim_mfcc, seq_length)
    delta2_noisy_frame_mfcc = noisy_decode_and_reshape(features, 'delta2_noisy_frame_mfcc', dim_mfcc, seq_length)
    seq_len = tf.shape(frame_melspectrogram)[0]
    
    # noisy spectral centroid
    noisy_frame_spectral_centroid = noisy_decode_and_reshape(features, 'noisy_frame_spectral_centroid', dim_spec_centr, seq_length) 
    delta_noisy_frame_spectral_centroid = noisy_decode_and_reshape(features, 'delta_noisy_frame_spectral_centroid', dim_spec_centr, seq_length)
    delta2_noisy_frame_spectral_centroid = noisy_decode_and_reshape(features, 'delta2_noisy_frame_spectral_centroid', dim_spec_centr, seq_length)
    
    # noisy mel-spectrogram
    noisy_frame_melspectrogram = noisy_decode_and_reshape(features, 'noisy_frame_melspectrogram', dim_melspecs, seq_length) 
    delta_noisy_frame_melspectrogram = noisy_decode_and_reshape(features, 'delta_noisy_frame_melspectrogram', dim_melspecs, seq_length)
    delta2_noisy_frame_melspectrogram = noisy_decode_and_reshape(features, 'delta2_noisy_frame_melspectrogram', dim_melspecs, seq_length)
    
    # noisy rmse
    noisy_frame_rmse = noisy_decode_and_reshape(features, 'noisy_frame_rmse', 1, seq_length)
    delta_noisy_frame_rmse = noisy_decode_and_reshape(features, 'delta_noisy_frame_rmse', 1, seq_length)
    delta2_noisy_frame_rmse = noisy_decode_and_reshape(features, 'delta2_noisy_frame_rmse', 1, seq_length)
    
    rmse = tf.decode_raw(features['rmse'], tf.float32)
    rmse = tf.reshape(rmse, (-1, 1))
    word = features['word']
    
    dec_features = []
    batch_features = []
    
    subject_id, label, raw_audio, frame_mfcc, frame_mfcc_overlap, delta_frame_mfcc, delta2_frame_mfcc, frame_melspectrogram, delta_frame_melspectrogram, delta2_frame_melspectrogram, frame_melspectrogram_overlap, frame_spectral_centroid, delta_frame_spectral_centroid, delta2_frame_spectral_centroid, noisy_frame_mfcc, delta_noisy_frame_mfcc, delta2_noisy_frame_mfcc, noisy_frame_spectral_centroid, delta_noisy_frame_spectral_centroid, delta2_noisy_frame_spectral_centroid, noisy_frame_melspectrogram, delta_noisy_frame_melspectrogram, delta2_noisy_frame_melspectrogram, noisy_frame_rmse, delta_noisy_frame_rmse, delta2_noisy_frame_rmse, seq_len = tf.train.batch([subject_id, label, raw_audio, frame_mfcc, frame_mfcc_overlap, delta_frame_mfcc, delta2_frame_mfcc, frame_melspectrogram, delta_frame_melspectrogram, delta2_frame_melspectrogram, frame_melspectrogram_overlap, frame_spectral_centroid, delta_frame_spectral_centroid, delta2_frame_spectral_centroid, noisy_frame_mfcc, delta_noisy_frame_mfcc, delta2_noisy_frame_mfcc, noisy_frame_spectral_centroid, delta_noisy_frame_spectral_centroid, delta2_noisy_frame_spectral_centroid, noisy_frame_melspectrogram, delta_noisy_frame_melspectrogram, delta2_noisy_frame_melspectrogram, noisy_frame_rmse, delta_noisy_frame_rmse, delta2_noisy_frame_rmse, seq_len], 
        batch_size, num_threads=1, capacity=1000, dynamic_pad=True)

    subject_id = tf.reshape(subject_id, (batch_size, -1))
    seq_length = -1
    label = tf.reshape(label, (batch_size, -1, dim_label))
    raw_audio = tf.reshape(raw_audio, (batch_size, -1, dim_raw_audio))
    
    # frame mfcc
    frame_mfcc = tf.reshape(frame_mfcc, (batch_size, -1, dim_mfcc))
    delta_frame_mfcc = tf.reshape(delta_frame_mfcc, (batch_size, -1, dim_mfcc))
    delta2_frame_mfcc = tf.reshape(delta2_frame_mfcc, (batch_size, -1, dim_mfcc))
    
    frame_mfcc_overlap = tf.reshape(frame_mfcc_overlap, (batch_size, -1, dim_mfcc))
    
    # frame mel-spectrogram
    frame_melspectrogram = tf.reshape(frame_melspectrogram, (batch_size, -1, dim_melspecs))
    delta_frame_melspectrogram = tf.reshape(delta_frame_melspectrogram, (batch_size, -1, dim_melspecs))
    delta2_frame_melspectrogram = tf.reshape(delta2_frame_melspectrogram, (batch_size, -1, dim_melspecs))
    
    frame_melspectrogram_overlap = tf.reshape(frame_melspectrogram_overlap, (batch_size, -1, dim_melspecs))
    
    # frame spectral centroid
    frame_spectral_centroid = tf.reshape(frame_spectral_centroid, (batch_size, seq_length, dim_spec_centr))
    delta_frame_spectral_centroid = tf.reshape(delta_frame_spectral_centroid, (batch_size, seq_length, dim_spec_centr))
    delta2_frame_spectral_centroid = tf.reshape(delta2_frame_spectral_centroid, (batch_size, seq_length, dim_spec_centr))
    
    # noisy spectral centroid
    noisy_frame_spectral_centroid = tf.reshape(noisy_frame_spectral_centroid, (batch_size, seq_length, num_fms, dim_spec_centr))
    delta_noisy_frame_spectral_centroid = tf.reshape(delta_noisy_frame_spectral_centroid, (batch_size, seq_length, num_fms, dim_spec_centr))
    delta2_noisy_frame_spectral_centroid = tf.reshape(delta2_noisy_frame_spectral_centroid, (batch_size, seq_length, num_fms, dim_spec_centr))
    
    # noisy frame mfcc
    noisy_frame_mfcc = tf.reshape(noisy_frame_mfcc, (batch_size, seq_length, num_fms, dim_mfcc))
    delta_noisy_frame_mfcc = tf.reshape(delta_noisy_frame_mfcc, (batch_size, seq_length, num_fms, dim_mfcc))
    delta2_noisy_frame_mfcc = tf.reshape(delta2_noisy_frame_mfcc, (batch_size, seq_length, num_fms, dim_mfcc))
    
    # noisy frame melspectrogram
    noisy_frame_melspectrogram = tf.reshape(noisy_frame_melspectrogram, (batch_size, seq_length, num_fms, dim_melspecs))
    delta_noisy_frame_melspectrogram = tf.reshape(delta_noisy_frame_melspectrogram, (batch_size, seq_length, num_fms, dim_melspecs))
    delta2_noisy_frame_melspectrogram = tf.reshape(delta2_noisy_frame_melspectrogram, (batch_size, seq_length, num_fms, dim_melspecs))
    
    # noisy spectral centroid
    noisy_frame_rmse = tf.reshape(noisy_frame_rmse, (batch_size, seq_length, num_fms, 1))
    delta_noisy_frame_rmse = tf.reshape(delta_noisy_frame_rmse, (batch_size, seq_length, num_fms, 1))
    delta2_noisy_frame_rmse = tf.reshape(delta2_noisy_frame_rmse, (batch_size, seq_length, num_fms, 1))
    
    return subject_id, label, raw_audio, frame_mfcc, frame_mfcc_overlap, delta_frame_mfcc, delta2_frame_mfcc, frame_melspectrogram, delta_frame_melspectrogram, delta2_frame_melspectrogram, frame_melspectrogram_overlap, frame_spectral_centroid, delta_frame_spectral_centroid, delta2_frame_spectral_centroid, noisy_frame_mfcc, delta_noisy_frame_mfcc, delta2_noisy_frame_mfcc, noisy_frame_spectral_centroid, delta_noisy_frame_spectral_centroid, delta2_noisy_frame_spectral_centroid, noisy_frame_melspectrogram, delta_noisy_frame_melspectrogram, delta2_noisy_frame_melspectrogram, noisy_frame_rmse, delta_noisy_frame_rmse, delta2_noisy_frame_rmse, seq_len
