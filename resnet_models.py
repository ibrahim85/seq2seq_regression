from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from basic_models import BasicModel

from model_utils import stacked_lstm, backend_resnet
# from resnet_model_utils import ResNet
from data_provider import get_split

class ResNetModel(BasicModel):
    """

    """

    def __init__(self, options):
        super(ResNetModel, self).__init__(options=options)
        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()
        else:
            self.build_train_graph()
        self.make_savers()

    def build_train_graph(self):
        self.encoder_out = backend_resnet(self.encoder_inputs, resnet_size=6,
                                          final_size=512, num_classes=None,
                                          frontend_3d=False, training=True, name="resnet")
        print("Encoder out:", self.encoder_out)
        self.decoder_out, _ = stacked_lstm(
            num_layers=self.options['decoder_num_layers'],
            num_hidden=self.options['decoder_num_hidden'],
            input_forw=self.encoder_out,
            layer_norm=self.options['decoder_layer_norm'],
            dropout_keep_prob=self.options['decoder_dropout_keep_prob'],
            is_training=True,
            residual=self.options['residual_decoder'],
            use_peepholes=True,
            return_cell=False)
        print("Decoder out:", self.decoder_out)
        self.decoder_outputs = tf.layers.dense(
                self.decoder_out, self.options['num_classes'], activation=None)

        self.define_loss()
        self.define_training_params()
