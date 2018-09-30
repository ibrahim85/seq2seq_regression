from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# import os
import numpy as np

from pathlib import Path


slim = tf.contrib.slim


def get_paths(base_path, split_name):
    if split_name == 'example':
        paths = np.loadtxt(str(base_path / 'example_set.csv'), dtype='<U150').tolist()
        print('Examples : ', len(paths))
    elif split_name == 'train':
        paths = np.loadtxt(str(base_path / 'train_set.csv'), dtype='<U150').tolist()
        print('Training examples : ', len(paths))
    elif  split_name == 'devel':
        paths = np.loadtxt(str(base_path / 'valid_set.csv'), dtype='<U150').tolist()
        print('Evaluating examples : ', len(paths))
    elif split_name == 'test':
        paths = np.loadtxt(str(base_path / 'test_set.csv'), dtype='<U150').tolist()
        print('Testing examples : ', len(paths))
    return paths


def length(sequence):
  used = tf.abs(tf.sign(tf.reduce_max(tf.abs(sequence), 2)))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length


def get_split(options):
    """
    Experiment 1:
        mfcc and raw_audio for seq2seq model
    Data:
        - 'raw_audio'
        - 'labels'
        - 'subject_id'
        - 'word'
        - 'mfcc'
    """
    batch_size = options['batch_size']
    num_classes = options['num_classes']
    mfcc_num_features = options['mfcc_num_features']
    raw_audio_num_features = options['raw_audio_num_features']

    base_path = Path(options['data_root_dir'])
    split_name = options['split_name']
    paths = get_paths(base_path, split_name)

    num_examples = len(paths)

    # Make queue
    filename_queue = tf.train.string_input_producer(paths, shuffle=options['shuffle'])

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

    # decode data
    # decoded_features = decode_features(features, ['raw_audio', 'mfcc', 'label', 'subject_id', 'word'])
    # raw_audio = decoded_features['raw_audio']
    # mfcc = decoded_features['mfcc']
    # label = decoded_features['label']
    # subject_id = decoded_features['subject_id']
    # word = decoded_features['word']
    raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    raw_audio = tf.reshape(raw_audio, ([1, -1]))

    mfcc = tf.decode_raw(features['mfcc'], tf.float32)
    mfcc = tf.reshape(mfcc, ([mfcc_num_features, -1]))
    mfcc = tf.cast(tf.transpose(mfcc, (1,0)), tf.float32)

    label = tf.decode_raw(features['labels'], tf.float32)
    label = tf.reshape(label, ([-1, num_classes]))

    subject_id = features['subject_id']
    word = features['word']

    raw_audio, mfcc, label, subject_id, word = tf.train.batch(
        [raw_audio, mfcc, label, subject_id, word], batch_size,
        num_threads=1, capacity=1000, dynamic_pad=True)

    label = tf.reshape(label, (batch_size, -1, num_classes))
    mfcc = tf.reshape(mfcc, (batch_size, -1, mfcc_num_features))
    raw_audio = tf.reshape(raw_audio, (batch_size, -1, raw_audio_num_features))

    # Add random Noise
    if options['mfcc_gaussian_noise_std'] != 0.0:
        mfcc = tf.add(mfcc,
                      tf.random_normal(tf.shape(mfcc), mean=0.0,
                          stddev=options['mfcc_gaussian_noise_std']))

    if options['label_gaussian_noise_std'] != 0.0:
        label = tf.add(label,
                       tf.random_normal(tf.shape(label), mean=0.0,
                           stddev=options['label_gaussian_noise_std']))

    # sos_token
    sos_token = tf.constant(1, dtype=tf.float32, shape=[batch_size, num_classes])
    sos_slice = tf.expand_dims(sos_token, [1])
    decoder_inputs = tf.concat([sos_slice, label], axis=1)

    # eos_token
    eos_token = tf.zeros([batch_size, num_classes])
    eos_slice = tf.expand_dims(eos_token, [1])
    target_labels = tf.concat([label, eos_slice], axis=1)

    label_lengths = length(label)
    mfcc_lengths = length(mfcc)
    decoder_inputs_lengths = length(decoder_inputs)

    if options['reverse_time']:
        raw_audio = tf.reverse(raw_audio, axis=[1])
        mfcc = tf.reverse(mfcc, axis=[1])

    return raw_audio, mfcc, target_labels, num_examples, word, decoder_inputs,\
           label_lengths, mfcc_lengths, decoder_inputs_lengths


def get_split2(options):
    """
    Experiment 2:
        frame_mfcc and raw_audio for seq2seq model
    Data:
        - 'raw_audio'
        - 'labels'
        - 'subject_id'
        - 'word'
        - 'mfcc'
        - 'frame_mfcc'
        - 'frame_mfcc_overlap'
        - 'delta_frame_mfcc'
        - 'delta2_frame_mfcc'
        - 'rmse'
    """
    batch_size = options['batch_size']
    num_classes = options['num_classes']
    mfcc_num_features = options['mfcc_num_features']
    # raw_audio_num_features = options['raw_audio_num_features']

    base_path = Path(options['data_root_dir'])
    split_name = options['split_name']
    paths = get_paths(base_path, split_name)

    num_examples = len(paths)

    filename_queue = tf.train.string_input_producer(paths, shuffle=options['shuffle'])

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
            'frame_mfcc': tf.FixedLenFeature([], tf.string),
            'frame_mfcc_overlap': tf.FixedLenFeature([], tf.string),
            'delta_frame_mfcc': tf.FixedLenFeature([], tf.string),
            'delta2_frame_mfcc': tf.FixedLenFeature([], tf.string),
            'rmse': tf.FixedLenFeature([], tf.string),
        }
    )

    #raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    #raw_audio = tf.reshape(raw_audio, ([1, -1]))

    #mfcc = tf.decode_raw(features['mfcc'], tf.float32)
    #mfcc = tf.cast(mfcc, tf.float32)
    #mfcc = tf.reshape(mfcc, ([mfcc_num_features, -1]))
    #mfcc = tf.cast(tf.transpose(mfcc, (1,0)), tf.float32)

    frame_mfcc = tf.decode_raw(features['frame_mfcc'], tf.float32)
    frame_mfcc = tf.reshape(frame_mfcc, (mfcc_num_features, -1))
    frame_mfcc = tf.cast(tf.transpose(frame_mfcc, (1,0)), tf.float32)

    #frame_mfcc_overlap = tf.decode_raw(features['frame_mfcc_overlap'], tf.float32)
    #frame_mfcc_overlap = tf.reshape(frame_mfcc_overlap, (mfcc_num_features, -1))
    #frame_mfcc_overlap = tf.cast(tf.transpose(frame_mfcc_overlap, (1,0)), tf.float32)

    #delta_frame_mfcc = tf.decode_raw(features['delta_frame_mfcc'], tf.float32)
    #delta_frame_mfcc = tf.reshape(delta_frame_mfcc, (mfcc_num_features, -1))
    #delta_frame_mfcc = tf.cast(tf.transpose(delta_frame_mfcc, (1,0)), tf.float32)

    label = tf.decode_raw(features['labels'], tf.float32)
    label = tf.reshape(label, ([-1, num_classes]))
    #label = tf.transpose(label, (1,0))

    #rmse = tf.decode_raw(features['rmse'], tf.float32)
    #rmse = tf.reshape(rmse, (-1, 1))

    subject_id = features['subject_id']
    word = features['word']

    ########## Augment Data
    if options['random_crop']:  # split_name == 'train':

        maxval = tf.cast(tf.shape(label)[0], tf.float32)
        s = tf.random_uniform([1], minval=0, maxval=0.5, dtype=tf.float32)
        s = tf.cast(tf.floor(s * maxval), tf.int32)[0]

        e = tf.random_uniform([1], minval=0.5, maxval=1, dtype=tf.float32)
        e = tf.cast(tf.floor(e * maxval - tf.cast(s, tf.float32) + 1), tf.int32)[0]
        e = tf.cond(e > s, lambda: e, lambda: s - e + 1)

        label = tf.slice(label, [s, 0], [e, 28])
        frame_mfcc = tf.slice(frame_mfcc, [s, 0], [e, 20])
    ##########


    frame_mfcc, label, subject_id, word = tf.train.batch(
        [frame_mfcc, label, subject_id, word], batch_size,
      num_threads=1, capacity=2048, dynamic_pad=True)

    # Add random Noise
    if options['mfcc_gaussian_noise_std'] != 0.0:
        frame_mfcc = tf.add(frame_mfcc,
                      tf.random_normal(tf.shape(frame_mfcc), mean=0.0,
                          stddev=options['mfcc_gaussian_noise_std']))

    if options['label_gaussian_noise_std'] != 0.0:
        label = tf.add(label,
                       tf.random_normal(tf.shape(label), mean=0.0,
                           stddev=options['label_gaussian_noise_std']))

    label = tf.reshape(label, (batch_size, -1, num_classes))

    #mfcc = tf.reshape(mfcc, (batch_size, -1, mfcc_num_features))
    #raw_audio = tf.reshape(raw_audio, (batch_size, -1, raw_audio_num_features))
    frame_mfcc = tf.reshape(frame_mfcc, (batch_size, -1, mfcc_num_features))
    #frame_mfcc_overlap = tf.reshape(frame_mfcc_overlap, (batch_size, -1, mfcc_num_features))
    #delta_frame_mfcc = tf.reshape(delta_frame_mfcc, (batch_size, -1, ))
    #rmse = tf.reshape(rmse, (batch_size, -1, 1))

    # sos_token
    sos_token = tf.constant(1, dtype=tf.float32, shape=[batch_size, num_classes])
    #sos_slice = tf.one_hot(sos_token, num_classes)
    #sos_token = tf.zeros([batch_size, num_classes])
    sos_slice = tf.expand_dims(sos_token, [1])
    decoder_inputs = tf.concat([sos_slice, label], axis=1)

    # eos_token
    eos_token = tf.zeros([batch_size, num_classes])
    eos_slice = tf.expand_dims(eos_token, [1])
    target_labels = tf.concat([label, eos_slice], axis=1)

    label_lengths = length(target_labels) + 1
    mfcc_lengths = length(frame_mfcc)
    decoder_inputs_lengths = length(decoder_inputs)

    if options['reverse_time']:
        #raw_audio = tf.reverse(raw_audio, axis=[1])
        frame_mfcc = tf.reverse(frame_mfcc, axis=[1])

    return frame_mfcc, target_labels, num_examples, word, decoder_inputs,\
           label_lengths, mfcc_lengths, decoder_inputs_lengths


def get_split3(options):
    # batch_size=32, num_classes=28, is_training=True, split_name='train'
    batch_size = options['batch_size']
    num_classes = options['num_classes']
    mfcc_num_features = options['mfcc_num_features']
    raw_audio_num_features = options['raw_audio_num_features']

    base_path = Path(options['data_root_dir'])
    split_name = options['split_name']
    paths = get_paths(base_path, split_name)

    num_examples = len(paths)

    filename_queue = tf.train.string_input_producer(paths, shuffle=options['shuffle'])

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
            'frame_mfcc': tf.FixedLenFeature([], tf.string),
            'frame_mfcc_overlap': tf.FixedLenFeature([], tf.string),
            'delta_frame_mfcc': tf.FixedLenFeature([], tf.string),
            'delta2_frame_mfcc': tf.FixedLenFeature([], tf.string),
            'rmse': tf.FixedLenFeature([], tf.string),
            'frame_melspectrogram': tf.FixedLenFeature([], tf.string),
            'frame_melspectrogram_overlap': tf.FixedLenFeature([], tf.string)
        }
    )

    # raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
    # raw_audio = tf.reshape(raw_audio, ([1, -1]))

    # mfcc = tf.decode_raw(features['mfcc'], tf.float32)
    # mfcc = tf.reshape(mfcc, (20, -1))
    # mfcc = tf.cast(tf.transpose(mfcc, (1, 0)), tf.float32)

    frame_mfcc = tf.decode_raw(features['frame_mfcc'], tf.float32)
    frame_mfcc = tf.reshape(frame_mfcc, (mfcc_num_features, -1))
    frame_mfcc = tf.cast(tf.transpose(frame_mfcc, (1, 0)), tf.float32)

    # frame_mfcc_overlap = tf.decode_raw(features['frame_mfcc_overlap'], tf.float32)
    # frame_mfcc_overlap = tf.reshape(frame_mfcc_overlap, (20, -1))
    # frame_mfcc_overlap = tf.cast(tf.transpose(frame_mfcc_overlap, (1, 0)), tf.float32)
    #
    # delta_frame_mfcc = tf.decode_raw(features['delta_frame_mfcc'], tf.float32)
    # delta_frame_mfcc = tf.reshape(delta_frame_mfcc, (20, -1))
    # delta_frame_mfcc = tf.cast(tf.transpose(delta_frame_mfcc, (1, 0)), tf.float32)
    #
    # delta2_frame_mfcc = tf.decode_raw(features['delta2_frame_mfcc'], tf.float32)
    # delta2_frame_mfcc = tf.reshape(delta2_frame_mfcc, (20, -1))
    # delta2_frame_mfcc = tf.cast(tf.transpose(delta2_frame_mfcc, (1, 0)), tf.float32)
    #
    # frame_melspectrogram = tf.decode_raw(features['frame_melspectrogram'], tf.float32)
    # frame_melspectrogram = tf.reshape(frame_melspectrogram, (128, -1))
    # frame_melspectrogram = tf.cast(tf.transpose(frame_melspectrogram, (1, 0)), tf.float32)
    #
    # frame_melspectrogram_overlap = tf.decode_raw(features['frame_melspectrogram_overlap'], tf.float32)
    # frame_melspectrogram_overlap = tf.reshape(frame_melspectrogram_overlap, (128, -1))
    # frame_melspectrogram_overlap = tf.cast(tf.transpose(frame_melspectrogram_overlap, (1, 0)), tf.float32)

    label = tf.decode_raw(features['labels'], tf.float32)
    label = tf.reshape(label, (-1, num_classes))

    # when used without <eos>
    #decoder_inputs = label[:-1, :]

    rmse = tf.decode_raw(features['rmse'], tf.float32)
    rmse = tf.reshape(rmse, (-1, 1))

    subject_id = features['subject_id']
    word = features['word']

    ########## Augment Data
    if options['random_crop']:  # split_name == 'train':

        maxval = tf.cast(tf.shape(label)[0], tf.float32)
        s = tf.random_uniform([1], minval=0, maxval=0.5, dtype=tf.float32)
        s = tf.cast(tf.floor(s * maxval), tf.int32)[0]

        e = tf.random_uniform([1], minval=0.5, maxval=1, dtype=tf.float32)
        e = tf.cast(tf.floor(e * maxval - tf.cast(s, tf.float32) + 1), tf.int32)[0]
        e = tf.cond(e > s, lambda: e, lambda: s - e + 1)

        label = tf.slice(label, [s, 0], [e, 28])
        #decoder_inputs = tf.slice(decoder_inputs, [s, 0], [e, num_classes])
        frame_mfcc = tf.slice(frame_mfcc, [s, 0], [e, mfcc_num_features])
        # delta_frame_mfcc = tf.slice(delta_frame_mfcc, [s, 0], [e, 20])
        # delta2_frame_mfcc = tf.slice(delta2_frame_mfcc, [s, 0], [e, 20])
        rmse = tf.slice(rmse, [s, 0], [e, 1])
    ##########

    frame_mfcc, rmse, label, subject_id, word = tf.train.batch(
        [frame_mfcc, rmse, label, subject_id, word], batch_size,
        num_threads=1, capacity=1000, dynamic_pad=True, allow_smaller_final_batch=True)

    label = tf.reshape(label, (batch_size, -1, num_classes))
    #mfcc = tf.reshape(mfcc, (batch_size, -1, 20))
    frame_mfcc = tf.reshape(frame_mfcc, (batch_size, -1, 20))
    #frame_mfcc_overlap = tf.reshape(frame_mfcc_overlap, (batch_size, -1, 20))
    #delta_frame_mfcc = tf.reshape(delta_frame_mfcc, (batch_size, -1, 20))
    rmse = tf.reshape(rmse, (batch_size, -1, 1))
    #raw_audio = tf.reshape(raw_audio, (batch_size, -1, 735))

    label_lengths = length(label)
    mfcc_lengths = length(frame_mfcc)
    decoder_inputs_lengths = label_lengths

    # standardize data per feature
    # for now only for frame_mfcc and label
    if options['standardize_inputs_and_labels']:
        eim = tf.convert_to_tensor(
            [-242.95416, 81.43694, -11.771074, 27.070665,
             -13.133402, 7.853387, -13.930464, 3.237301,
             -7.58117, -0.85082525, -1.0349289, -4.017565,
             1.1383969, -6.5288205, 3.3377466, -6.3484197,
             0.562516, -4.093935, -0.8670171, -2.6639743], dtype=tf.float32)
        eivar = tf.convert_to_tensor(
            [4.73080000e+04, 6.62778369e+03, 7.94231201e+02, 1.01425513e+03,
             5.15957764e+02, 4.11880280e+02, 3.49256989e+02, 1.65152344e+02,
             1.58384720e+02, 1.00354126e+02, 7.98662262e+01, 7.84846039e+01,
             7.00963669e+01, 9.50163040e+01, 7.33705063e+01, 9.03896027e+01,
             5.94251404e+01, 5.85693321e+01, 4.78590660e+01, 4.46491356e+01], dtype=tf.float32)
        tlm = tf.convert_to_tensor(
            [0.08709314, 0.20119551, 0.01583645, -0.07990878, -0.03016437,
             -0.02380619, -0.00410177, 0.16347252, 0.13605969, 0.06855655,
             -0.01053068, -0.11301234, 0.10808781, 0.01175822, -0.18842568,
             0.0545788, -0.01378771, 0.03019104, -0.04349428, 0.08825737,
             0.04038208, -0.01783869, 0.07432479, 0.03237964, 0.09984641,
             0.09162843, 0.0726966, -0.05494107], dtype=tf.float32)
        tlvar = tf.convert_to_tensor(
            [0.96776265, 0.6284481, 0.24534242, 0.24819732, 0.042043,
             0.09614109, 0.06104016, 0.04933477, 0.0866574, 0.04351515,
             0.03135443, 0.05627843, 0.04494405, 0.03083666, 0.09743688,
             0.02250072, 0.02160346, 0.02263445, 0.02424745, 0.04163726,
             0.01505357, 0.01757062, 0.02050258, 0.0164365, 0.01612102,
             0.0303383, 0.0188257, 0.04577498], dtype=tf.float32)
        frame_mfcc = tf.reshape(frame_mfcc, [-1, mfcc_num_features])
        frame_mfcc = (frame_mfcc - eim) / tf.sqrt(eivar)
        frame_mfcc = tf.reshape(frame_mfcc, (batch_size, -1, mfcc_num_features))
        label = tf.reshape(label, [-1, num_classes])
        label = (label - tlm) / tf.sqrt(tlvar)
        label = tf.reshape(label, (batch_size, -1, num_classes))

    if options['use_rmse']:
        encoder_inputs = tf.concat([frame_mfcc, rmse], axis=-1)
    else:
        encoder_inputs = frame_mfcc

    decoder_inputs = label[:, :-1, :]
    # sos_token
    sos_token = tf.constant(0, dtype=tf.float32, shape=[batch_size, num_classes])
    sos_slice = tf.expand_dims(sos_token, [1])
    decoder_inputs = tf.concat([sos_slice, decoder_inputs], axis=1)

    target_labels = label

    # label_lengths = length(target_labels)
    # mfcc_lengths = length(frame_mfcc)
    # decoder_inputs_lengths = length(decoder_inputs) + 1
    #rmse_lengths = length(rmse)

    return encoder_inputs, target_labels, num_examples, word, decoder_inputs,\
           label_lengths, mfcc_lengths, decoder_inputs_lengths




# def decode_features(features, requested_features, options):
#     features_dict = {}
#     if 'raw_audio' in requested_features:
#         raw_audio = tf.decode_raw(features['raw_audio'], tf.float32)
#         raw_audio = tf.reshape(raw_audio, ([1, -1]))
#         features_dict['raw_audio'] = raw_audio
#     if 'mfcc' in requested_features:
#         mfcc = tf.decode_raw(features['mfcc'], tf.float32)
#         mfcc = tf.cast(mfcc, tf.float32)
#         mfcc = tf.reshape(mfcc, ([options['mfcc_num_features'], -1]))
#         mfcc = tf.cast(tf.transpose(mfcc, (1,0)), tf.float32)
#         features_dict['mfcc'] = mfcc
#     if 'frame_mfcc' in requested_features:
#         frame_mfcc = tf.decode_raw(features['frame_mfcc'], tf.float32)
#         frame_mfcc = tf.reshape(frame_mfcc, (options['mfcc_num_features'], -1))
#         frame_mfcc = tf.cast(tf.transpose(frame_mfcc, (1, 0)), tf.float32)
#         features_dict['frame_mfcc'] = frame_mfcc
#     if 'frame_mfcc_overlap' in requested_features:
#         frame_mfcc_overlap = tf.decode_raw(features['frame_mfcc_overlap'], tf.float32)
#         frame_mfcc_overlap = tf.reshape(frame_mfcc_overlap, (options['mfcc_num_features'], -1))
#         frame_mfcc_overlap = tf.cast(tf.transpose(frame_mfcc_overlap, (1,0)), tf.float32)
#         features_dict['frame_mfcc_overlap'] = frame_mfcc_overlap
#     if 'delta_frame_mfcc' in requested_features:
#         delta_frame_mfcc = tf.decode_raw(features['delta_frame_mfcc'], tf.float32)
#         delta_frame_mfcc = tf.reshape(delta_frame_mfcc, (20, -1))
#         delta_frame_mfcc = tf.cast(tf.transpose(delta_frame_mfcc, (1, 0)), tf.float32)
#         features_dict['delta_frame_mfcc'] = delta_frame_mfcc
#     if 'delta2_frame_mfcc' in requested_features:
#         delta2_frame_mfcc = tf.decode_raw(features['delta2_frame_mfcc'], tf.float32)
#         delta2_frame_mfcc = tf.reshape(delta2_frame_mfcc, (20, -1))
#         delta2_frame_mfcc = tf.cast(tf.transpose(delta2_frame_mfcc, (1, 0)), tf.float32)
#         features_dict['delta2_frame_mfcc'] = delta2_frame_mfcc
#     if 'frame_melspectrogram' in requested_features:
#         frame_melspectrogram = tf.decode_raw(features['frame_melspectrogram'], tf.float32)
#         frame_melspectrogram = tf.reshape(frame_melspectrogram, (128, -1))
#         frame_melspectrogram = tf.cast(tf.transpose(frame_melspectrogram, (1, 0)), tf.float32)
#         features_dict['frame_melspectrogram'] = frame_melspectrogram
#     if 'frame_melspectrogram_overlap' in requested_features:
#         frame_melspectrogram_overlap = tf.decode_raw(features['frame_melspectrogram_overlap'], tf.float32)
#         frame_melspectrogram_overlap = tf.reshape(frame_melspectrogram_overlap, (128, -1))
#         frame_melspectrogram_overlap = tf.cast(tf.transpose(frame_melspectrogram_overlap, (1, 0)), tf.float32)
#         features_dict['frame_melspectrogram_overlap'] = frame_melspectrogram_overlap
#     if 'label' in requested_features:
#         label = tf.decode_raw(features['labels'], tf.float32)
#         label = tf.reshape(label, (-1, options['num_classes']))
#         features_dict['label'] = label
#     if 'rmse' in requested_features:
#         rmse = tf.decode_raw(features['rmse'], tf.float32)
#         rmse = tf.reshape(rmse, (-1, 1))
#         features_dict['rmse'] = rmse
#     if 'subject_id' in requested_features:
#         subject_id = features['subject_id']
#         features_dict['subject_id'] = subject_id
#     if 'word' in requested_features:
#         word = features['word']
#         features_dict['word'] = word
#     return features_dict
