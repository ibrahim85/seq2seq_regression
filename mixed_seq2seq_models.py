from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from basic_models import BasicModel

from model_utils import stacked_lstm, temp_res_conv_network, get_attention_cell, get_decoder_init_state, RNMTplus_net
from transformer_model import SelfAttentionEncoder

class CNNRNNSeq2SeqModel(BasicModel):
    """

    """
    def __init__(self, options):
        super(CNNRNNSeq2SeqModel, self).__init__(options=options)
        # additional tensors for seq2seq
        self.decoder_inputs = tf.identity(self.target_labels)[:, :-1, :]
        self.decoder_inputs_lengths = tf.identity(self.target_labels_lengths) - 1
        self.target_labels = tf.identity(self.target_labels)[:, 1:, :]
        self.target_labels_lengths = tf.identity(self.target_labels_lengths) - 1
        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()
        else:
            self.build_train_graph()
        self.make_savers()

    def build_train_graph(self):
        # if self.options['has_encoder']:
        with tf.variable_scope('encoder'):
            self.encoder_out = temp_res_conv_network(self.encoder_inputs, self.options)

        # if self.options['has_decoder']:
        with tf.variable_scope('decoder'):
            ss_prob = self.options['ss_prob']
            self.sampling_prob = tf.constant(ss_prob, dtype=tf.float32)
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
                self.decoder_inputs,
                self.decoder_inputs_lengths,
                self.sampling_prob)
            decoder_cell = stacked_lstm(
                num_layers=self.options['decoder_num_layers'],
                num_hidden=self.options['decoder_num_hidden'],
                # layer_norm=self.options['decoder_layer_norm'],
                # dropout_keep_prob=self.options['decoder_dropout_keep_prob'],
                is_training=self.is_training,
                residual=self.options['residual_decoder'],
                # use_peepholes=True,
                input_forw=None,
                return_cell=True)
            attention_cell = get_attention_cell(cell=decoder_cell,
                                                options=self.options,
                                                memories=self.encoder_out,
                                                memories_lengths=self.encoder_inputs_lengths)
            decoder_init_state = get_decoder_init_state(cell=attention_cell,
                                                        init_state=None,
                                                        options=self.options)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attention_cell,
                helper=helper,
                initial_state=decoder_init_state,
                output_layer=tf.layers.Dense(self.options['num_classes']))
            outputs, self.final_state, final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=None)
            self.decoder_outputs = outputs.rnn_output

        self.define_loss()
        self.define_training_params()


class TransRNNSeq2SeqModel(BasicModel):
    """

    """
    def __init__(self, options):
        super(TransRNNSeq2SeqModel, self).__init__(options=options)
        # additional tensors for seq2seq
        self.decoder_inputs = tf.identity(self.target_labels)[:, :-1, :]
        self.decoder_inputs_lengths = tf.identity(self.target_labels_lengths) - 1
        self.target_labels = tf.identity(self.target_labels)[:, 1:, :]
        self.target_labels_lengths = tf.identity(self.target_labels_lengths) - 1
        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()
        else:
            self.build_train_graph()
        self.make_savers()

    def build_train_graph(self):
        # if self.options['has_encoder']:
        with tf.variable_scope('encoder'):
            transformer_encoder = SelfAttentionEncoder(self.options)
            self.encoder_out = transformer_encoder.encoder_outputs

        # if self.options['has_decoder']:
        with tf.variable_scope('decoder'):
            ss_prob = self.options['ss_prob']
            self.sampling_prob = tf.constant(ss_prob, dtype=tf.float32)
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
                self.decoder_inputs,
                self.decoder_inputs_lengths,
                self.sampling_prob)
            decoder_cell = stacked_lstm(
                num_layers=self.options['decoder_num_layers'],
                num_hidden=self.options['decoder_num_hidden'],
                # layer_norm=self.options['decoder_layer_norm'],
                # dropout_keep_prob=self.options['decoder_dropout_keep_prob'],
                is_training=self.is_training,
                residual=self.options['residual_decoder'],
                # use_peepholes=True,
                input_forw=None,
                return_cell=True)
            attention_cell = get_attention_cell(cell=decoder_cell,
                                                options=self.options,
                                                memories=self.encoder_out,
                                                memories_lengths=self.encoder_inputs_lengths)
            decoder_init_state = get_decoder_init_state(cell=attention_cell,
                                                        init_state=None,
                                                        options=self.options)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attention_cell,
                helper=helper,
                initial_state=decoder_init_state,
                output_layer=tf.layers.Dense(self.options['num_classes']))
            outputs, self.final_state, final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=None)
            self.decoder_outputs = outputs.rnn_output

        self.define_loss()
        self.define_training_params()