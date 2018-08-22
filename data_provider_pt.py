from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np

from pathlib import Path


slim = tf.contrib.slim

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
    
    base_path = Path('/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_enhanced/')
    if split_name == 'train':
        paths = np.loadtxt(str(base_path / 'train_set.csv'), dtype='<U150').tolist()
        print('Training examples : ', len(paths))
    elif split_name == 'devel':
        paths = np.loadtxt(str(base_path / 'valid_set.csv'), dtype='<U150').tolist()
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
            'raw_audio': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
            'subject_id': tf.FixedLenFeature([], tf.string),
            'word': tf.FixedLenFeature([], tf.string),
            'mfcc': tf.FixedLenFeature([], tf.string),
        }
    )
    
    raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    raw_audio = tf.reshape(raw_audio, ([1, -1]))
    
    mfcc = tf.decode_raw(features['mfcc'], tf.float32)
    mfcc = tf.reshape(mfcc, (20, -1))
    mfcc = tf.cast(tf.transpose(mfcc, (1,0)), tf.float32)
    
    label = tf.decode_raw(features['labels'], tf.float32)
    label = tf.reshape(label, (-1, 28))
    
    subject_id = features['subject_id']
    word = features['word']
    
    raw_audio, mfcc, label, subject_id, word = tf.train.batch(
        [raw_audio, mfcc, label, subject_id, word], batch_size, 
        num_threads=1, capacity=1000, dynamic_pad=True)
    
    label = tf.reshape(label, (batch_size, -1, 28))
    mfcc = tf.reshape(mfcc, (batch_size, -1, 20))
    raw_audio = tf.reshape(raw_audio, (batch_size, -1, 533))
    
    return raw_audio, mfcc, label, num_examples, word
