from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from basic_models import BasicModel

from model_utils import stacked_lstm, blstm_encoder, get_attention_cell, get_decoder_init_state, RNMTplus_net


# from losses import batch_masked_concordance_cc, batch_masked_mse, L2loss, masked_concordance_cc
# from data_provider import get_split, get_split2, get_split3
# import numpy as np
# import pandas as pd

class RNNModel(BasicModel):
    """

    """

    def __init__(self, options):
        super(RNNModel, self).__init__(options=options)
        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()
        else:
            self.build_train_graph()
        self.make_savers()

    def build_train_graph(self):
        # if self.options['has_encoder']:
        with tf.variable_scope('encoder'):
            if self.options['bidir_encoder']:
                self.encoder_out, self.encoder_hidden = blstm_encoder(
                    input_forw=self.encoder_inputs, options=self.options)
            else:
                self.encoder_out, self.encoder_hidden = stacked_lstm(
                    num_layers=self.options['encoder_num_layers'],
                    num_hidden=self.options['encoder_num_hidden'],
                    input_forw=self.encoder_inputs,
                    layer_norm=self.options['encoder_layer_norm'],
                    dropout_keep_prob=self.options['encoder_dropout_keep_prob'],
                    is_training=True,
                    residual=self.options['residual_encoder'],
                    use_peepholes=True,
                    return_cell=False)
                print("Encoder hidden:", self.encoder_hidden)
        with tf.variable_scope('decoder'):
            self.decoder_outputs = tf.layers.dense(
                self.encoder_out, self.options['num_classes'], activation=None)

        self.define_loss()
        self.define_training_params()


class RNNplusModel(BasicModel):
    """

    """

    def __init__(self, options):
        super(RNNplusModel, self).__init__(options=options)
        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()
        else:
            self.build_train_graph()
        self.make_savers()

    def build_train_graph(self):
        # with tf.variable_scope('input_linear_projection'):
        self.encoder_inputs = tf.layers.dense(
            inputs=self.encoder_inputs,
            units=2*self.options['num_hidden'], activation=None, use_bias=True,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None, trainable=True,
            name=None, reuse=None)
        self.encoder_inputs = tf.layers.batch_normalization(self.encoder_inputs,
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=self.is_training)
        self.encoder_inputs = tf.nn.relu(self.encoder_inputs)

        with tf.variable_scope('encoder'):
            self.encoder_out = RNMTplus_net(self.encoder_inputs, self.options)

        with tf.variable_scope('decoder'):
            self.decoder_outputs = tf.layers.dense(
                self.encoder_out, self.options['num_classes'], activation=None)

        self.define_loss()
        self.define_training_params()