from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from basic_models import BasicModel

from model_utils import stacked_lstm, get_attention_cell, get_decoder_init_state, temp_res_conv_network


class CNNModel(BasicModel):
    """

    """

    def __init__(self, options):
        super(CNNModel, self).__init__(options=options)
        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()
        else:
            self.build_train_graph()
        self.make_savers()

    def build_train_graph(self):
        if self.options['has_encoder']:
            with tf.variable_scope('encoder'):
                self.encoder_out = temp_res_conv_network(self.encoder_inputs, self.options)
                #self.encoder_out = temp_conv_network(self.encoder_inputs, self.options)
                #self.encoder_out = temp_conv_network2(self.encoder_inputs, self.options)
                print("Encoder out:", self.encoder_out)
        else:
            self.encoder_out = self.encoder_inputs
            self.encoder_hidden = None

        if self.options['has_decoder']:
            with tf.variable_scope('decoder_lstm'):
                ss_prob = self.options['ss_prob']
                self.sampling_prob = tf.constant(ss_prob, dtype=tf.float32)
                helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
                    self.decoder_inputs,
                    self.decoder_inputs_lengths,
                    self.sampling_prob)
                decoder_cell = stacked_lstm(
                    num_layers=self.options['decoder_num_layers'],
                    num_hidden=self.options['decoder_num_hidden'],
                    layer_norm=self.options['decoder_layer_norm'],
                    dropout_keep_prob=self.options['decoder_dropout_keep_prob'],
                    is_training=True,
                    residual=self.options['residual_decoder'],
                    use_peepholes=True,
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
                        maximum_iterations=self.options['max_out_len'])
                self.decoder_outputs = outputs.rnn_output
        else:
            self.decoder_outputs = tf.layers.dense(
                self.encoder_out, self.options['num_classes'], activation=None)
            # the following two lines are for compatability (needs to print)
            # there is no use for sampling_prob when there is no decoder
            ss_prob = self.options['ss_prob']
            self.sampling_prob = tf.constant(ss_prob, dtype=tf.float32)

        self.define_loss()
        self.define_training_params()
