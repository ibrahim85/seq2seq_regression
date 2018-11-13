from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from basic_models import BasicModel

from model_utils import temp_res_conv_network, dense_1d_conv_network


class CNNModel(BasicModel):
    """
    Basic 1d CNN model
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
        with tf.variable_scope('1dcnn'):
            self.encoder_out = temp_res_conv_network(self.encoder_inputs, self.options)
            #self.encoder_out = temp_conv_network(self.encoder_inputs, self.options)
            #self.encoder_out = temp_conv_network2(self.encoder_inputs, self.options)
            print("Encoder out:", self.encoder_out)
        with tf.variable_scope('output_layer'):
            self.decoder_outputs = tf.layers.dense(
                self.encoder_out, self.options['num_classes'], activation=None)
            # the following two lines are for compatability (needs to print)
            # there is no use for sampling_prob when there is no decoder
            ss_prob = self.options['ss_prob']
            self.sampling_prob = tf.constant(ss_prob, dtype=tf.float32)
        self.define_loss()
        self.define_training_params()


class DenseNet1D(BasicModel):
    """

    """

    def __init__(self, options):
        super(DenseNet1D, self).__init__(options=options)
        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()
        else:
            self.build_train_graph()
        self.make_savers()

    def build_train_graph(self):
        # self.encoder_inputs = tf.layers.batch_normalization(self.encoder_inputs,
        #     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
        #     beta_initializer=tf.zeros_initializer(),
        #     gamma_initializer=tf.ones_initializer(),
        #     moving_mean_initializer=tf.zeros_initializer(),
        #     moving_variance_initializer=tf.ones_initializer(),
        #     training=self.is_training)
        with tf.variable_scope('1dcnn'):
            self.encoder_out = dense_1d_conv_network(self.encoder_inputs, self.options)
            #self.encoder_out = temp_conv_network(self.encoder_inputs, self.options)
            #self.encoder_out = temp_conv_network2(self.encoder_inputs, self.options)
            print("Encoder out:", self.encoder_out)
        with tf.variable_scope('output_layer'):
            self.decoder_outputs = tf.layers.dense(
                self.encoder_out, self.options['num_classes'], activation=None)

        self.define_loss()
        self.define_training_params()
