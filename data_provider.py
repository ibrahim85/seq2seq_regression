from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np

from pathlib import Path


slim = tf.contrib.slim


def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length


def get_split(batch_size=32, num_classes=7, is_training=True, split_name='train'):
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

    base_path = Path('/vol/atlas/homes/pt511/db/audio_to_3d/tf_records')
    if split_name == 'example':
        paths = np.loadtxt(str(base_path / 'example_set.csv'), dtype=str).tolist()
        print('Examples : ', len(paths))
    elif split_name == 'train':
        paths = np.loadtxt(str(base_path / 'train_set.csv'), dtype=str).tolist()
        print('Training examples : ', len(paths))
    elif  split_name == 'devel':
        paths = np.loadtxt(str(base_path / 'valid_set.csv'), dtype=str).tolist()
        print('Evaluating examples : ', len(paths))
    elif split_name == 'test':
        paths = np.loadtxt(str(base_path / 'test_set.csv'), dtype=str).tolist()
        print('Testing examples : ', len(paths))

    num_examples = len(paths)
    # print(num_examples)

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
            'mfcc': tf.FixedLenFeature([], tf.string)
        }
    )

    raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    #raw_audio = tf.reshape(raw_audio, ([1, 20480]))

    mfcc = tf.decode_raw(features['mfcc'], tf.float64)
    mfcc = tf.cast(mfcc, tf.float32)
    # mfcc = tf.reshape(mfcc, ([20, 129]))
    #mfcc = tf.cast(tf.transpose(mfcc, (1,0)), tf.float32)

    label = tf.decode_raw(features['labels'], tf.float32)
    # label = tf.reshape(label, ([1, 7]))
    # label = tf.transpose(label, (1,0))

    subject_id = features['subject_id']
    word = features['word']

    raw_audio, mfcc, label, subject_id, word = tf.train.batch(
        [raw_audio, mfcc, label, subject_id, word], batch_size,
        num_threads=1, capacity=1000, dynamic_pad=True)

    label = tf.reshape(label, (batch_size, -1, num_classes))
    mfcc = tf.reshape(mfcc, (batch_size, -1, 20))
    raw_audio = tf.reshape(raw_audio, (batch_size, -1, 256))
    #
    # sos_slice = tf.constant(0., dtype=tf.float32, shape=[batch_size, 1, num_classes])
    sos_token = tf.constant(0, dtype=tf.int32, shape=[batch_size, 1])
    sos_slice = tf.one_hot(sos_token, num_classes)
    decoder_inputs = tf.concat([sos_slice, label], axis=1)
    #
    # eos_token = tf.constant(1, dtype=tf.int32, shape=[batch_size, 1])
    # eos_slice = tf.one_hot(eos_token, num_classes)
    eos_token = tf.zeros([batch_size, num_classes])
    eos_slice = tf.expand_dims(eos_token, [1])
    target_labels = tf.concat([label, eos_slice], axis=1)

    label_lengths = length(label)
    mfcc_lengths = length(mfcc)
    decoder_inputs_lengths = length(decoder_inputs)

    return raw_audio, mfcc, target_labels, num_examples, word, decoder_inputs, label_lengths, mfcc_lengths, decoder_inputs_lengths

