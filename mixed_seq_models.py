from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from basic_models import BasicModel

from model_utils import stacked_lstm, temp_res_conv_network
from transformer_model import SelfAttentionEncoder

class CNNRNNModel(BasicModel):
    """

    """
    def __init__(self, options):
        super(CNNRNNModel, self).__init__(options=options)
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
            self.decoder_outputs = stacked_lstm(
                    num_layers=self.options['decoder_num_layers'],
                    num_hidden=self.options['decoder_num_hidden'],
                    input_forw=self.encoder_out,
                    layer_norm=self.options['decoder_layer_norm'],
                    dropout_keep_prob=self.options['decoder_dropout_keep_prob'],
                    is_training=True,
                    residual=self.options['residual_decoder'],
                    use_peepholes=True,
                    return_cell=False)

        self.define_loss()
        self.define_training_params()


class TransRNNModel(BasicModel):
    """

    """
    def __init__(self, options):
        super(TransRNNModel, self).__init__(options=options)
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
                self.decoder_outputs = stacked_lstm(
                    num_layers=self.options['decoder_num_layers'],
                    num_hidden=self.options['decoder_num_hidden'],
                    input_forw=self.encoder_out,
                    layer_norm=self.options['decoder_layer_norm'],
                    dropout_keep_prob=self.options['decoder_dropout_keep_prob'],
                    is_training=True,
                    residual=self.options['residual_decoder'],
                    use_peepholes=True,
                    return_cell=False)

        self.define_loss()
        self.define_training_params()