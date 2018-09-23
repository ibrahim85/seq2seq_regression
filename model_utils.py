from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from helper_functions import *
import numpy as np

# import six
# from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
# from tensorflow.python.layers import utils
# from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops


def stacked_lstm(num_layers, num_hidden, is_training, input_forw=None,
                 layer_norm=False, dropout_keep_prob=1.0, residual=False,
                 return_cell=False, use_peepholes=True):
    """
    input_forw : (tensor) input tensor forward in time
    num_layers : (int) depth of stacked LSTM
    num_hidden : (int, list, tuple) number of units at each LSTM layer
    """
    if type(num_hidden) is int:
        num_hidden = [num_hidden] * num_layers
    if not is_training:
        dropout_keep_prob = 1.0
    # print(len(num_hidden))
    # assert len(num_hidden) == num_layers \
    #     "length of num_hidden %d, must match num_layers %d" % (len(num_hidden), num_layers)
    # with tf.name_scope(name):
    # input_back = tf.reverse(input_forw, axis=[1])
    #if residual:
    #    rnn_layers = [tf.contrib.rnn.ResidualWrapper(
    #                  tf.contrib.rnn.LSTMCell(
    #                     num_units=layer_size_,
    #                     use_peepholes=use_peepholes, cell_clip=None, initializer=None, num_proj=None, proj_clip=None,
    #                     forget_bias=1.0, state_is_tuple=True,
    #                     activation=tf.tanh, reuse=None))
    #                  for _, layer_size_ in enumerate(num_hidden)]
    #else:
    #    rnn_layers = [tf.contrib.rnn.LSTMCell(
    #                     num_units=layer_size_,
    #                     use_peepholes=use_peepholes, cell_clip=None, initializer=None, num_proj=None, proj_clip=None,
    #                     forget_bias=1.0, state_is_tuple=True,
    #                     activation=tf.tanh, reuse=None)
    #                  for _, layer_size_ in enumerate(num_hidden)]
    def cellfn(layer_size):
        return tf.contrib.rnn.LayerNormBasicLSTMCell(
                         num_units=layer_size,
                         forget_bias=1.0,
                         activation=tf.tanh,
                         layer_norm=layer_norm, 
                         norm_gain=1.0,
                         norm_shift=0.0,
                         dropout_keep_prob=dropout_keep_prob)
    if residual:
        rnn_layers = [tf.contrib.rnn.ResidualWrapper(cellfn(layer_size))
                      for _, layer_size in enumerate(num_hidden)]
    else:
        rnn_layers = [cellfn(layer_size) for _, layer_size in enumerate(num_hidden)]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    if return_cell:
        return multi_rnn_cell
    assert input_forw is not None, "RNN input is None"
    outputs, states = tf.nn.dynamic_rnn(multi_rnn_cell, input_forw, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    #states = tf.concat(states, 2)
    return outputs, states


def blstm_encoder(input_forw, options):
    """
    input_forw : input tensor forward in time
    """
    if 'encoder_dropout_keep_prob' in options:
        dropout_keep_prob = options['encoder_dropout_keep_prob']
    else:
        dropout_keep_prob = 1.0

    """
    # with tf.name_scope(name):
    # input_back = tf.reverse(input_forw, axis=[1])
    if model_options['residual_encoder']:
        print('encoder : residual BLSTM')
        rnn_layers = [tf.contrib.rnn.ResidualWrapper(
                      tf.contrib.rnn.LayerNormBasicLSTMCell(model_options['encoder_num_hidden'],
                                                            forget_bias=1.0,
                                                            activation=tf.tanh,
                                                            layer_norm=True,
                                                            norm_gain=1.0,
                                                            norm_shift=0.0,))
                      for _ in range(model_options['encoder_num_layers'])]
    else:
        print('encoder : BLSTM')
        rnn_layers = [tf.contrib.rnn.LayerNormBasicLSTMCell(model_options['encoder_num_hidden'],
                                                            forget_bias=1.0,
                                                            activation=tf.tanh,
                                                            layer_norm=True,
                                                            norm_gain=1.0,
                                                            norm_shift=0.0,
                                                            dropout_keep_prob=dropout_keep_prob)
                      for _ in range(model_options['encoder_num_layers'])]
    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(
                                multi_rnn_cell_forw, multi_rnn_cell_back,
                                input_forw, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    # hidden_states = tf.concat(hidden_states, 3)
    # hidden_states_list = []
    # for layer_id in range(model_options['encoder_num_layers']):
    #     hidden_states_list.append(hidden_states[0][layer_id])  # forward
    #     hidden_states_list.append(hidden_states[1][layer_id])  # backward
    # hidden_states = tuple(hidden_states_list)
    return outputs, hidden_states
    """
    dense_out_forw = input_forw #tf.squeeze(dense_out, axis=2)
    dense_out_back = tf.reverse(dense_out_forw, axis=[1])
    # create 2 layer LSTMCells
    rnn_layers = [tf.contrib.rnn.LayerNormBasicLSTMCell(options['encoder_num_hidden'],
                                                            forget_bias=1.0,
                                                            input_size=None,
                                                            activation=tf.tanh,
                                                            layer_norm=True,
                                                            norm_gain=1.0,
                                                            norm_shift=0.0,
                                                            #dropout_keep_prob=dropout_keep_prob,
                                                            dropout_prob_seed=None,
                                                            reuse=None) for _ in range(options['encoder_num_layers'])]
    # rnn_layers = [tf.contrib.rnn.LSTMCell(size) for size in [256, 256]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs_forw, hidden_forw = tf.nn.dynamic_rnn(cell=multi_rnn_cell_forw,
                                            inputs=dense_out_forw,
                                            dtype=tf.float32)
    outputs_back, hidden_back = tf.nn.dynamic_rnn(cell=multi_rnn_cell_back,
                                            inputs=dense_out_back,
                                            dtype=tf.float32)    
    bilstm_out = tf.concat([outputs_forw, tf.reverse(outputs_back, axis=[1])], axis=-1)
    bilstm_hidden = tf.concat([hidden_forw, tf.reverse(hidden_back, axis=[1])], axis=-1)
    return bilstm_out, bilstm_hidden  




def blstm_2layer(x_input, name="blstm_2layer"):
    with tf.name_scope(name):
        # 2-layer BiLSTM
        # dense_out = tf.concat(dense_out, axis=1)
        # Define input for forward and backward LSTM
        dense_out_forw = x_input #tf.squeeze(dense_out, axis=2)
        dense_out_back = tf.reverse(dense_out_forw, axis=[1])
        # create 2 layer LSTMCells
        # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]
        rnn_layers = [tf.contrib.rnn.LayerNormBasicLSTMCell(size) for size in [256, 256]]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        outputs_forw, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_forw,
                                            inputs=dense_out_forw,
                                            dtype=tf.float32)
        outputs_back, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_back,
                                            inputs=dense_out_back,
                                            dtype=tf.float32)

        # get only the last output from lstm
        # lstm_out = tf.transpose(lstm_out, [1, 0, 2])
        last_forw = tf.gather(outputs_forw, indices=int(outputs_forw.get_shape()[1]) - 1, axis=1)
        last_back = tf.gather(outputs_back, indices=int(outputs_forw.get_shape()[1]) - 1, axis=1)

        bilstm_out = tf.concat([last_forw, last_back], axis=1)
        print("shape of bilstm output is %s" % bilstm_out.get_shape())
        return bilstm_out

def lstm_1layer(x_input, size=256, name="lstm_1layer"):
    with tf.name_scope(name):
        # 2-layer BiLSTM
        # dense_out = tf.concat(dense_out, axis=1)
        # Define input for forward and backward LSTM
        #dense_out_forw = x_input #tf.squeeze(dense_out, axis=2)
        #dense_out_back = tf.reverse(dense_out_forw, axis=[1])
        # create 2 layer LSTMCells
        # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]
        rnn_layers = tf.contrib.rnn.LSTMCell(size)

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        #multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        lstm_out, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_forw,
                                            inputs=x_input,
                                            dtype=tf.float32)
        # outputs_back, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell_back,
        #                                     inputs=dense_out_back,
        #                                     dtype=tf.float32)

        # get only the last output from lstm
        # lstm_out = tf.transpose(lstm_out, [1, 0, 2])
        #lstm_out = tf.gather(outputs_forw, indices=int(outputs_forw.get_shape()[1]) - 1, axis=1)
        #last_back = tf.gather(outputs_back, indices=int(outputs_forw.get_shape()[1]) - 1, axis=1)

        #bilstm_out = tf.concat([last_forw, last_back], axis=1)
        print("shape of lstm output is %s" % lstm_out.get_shape())
        return lstm_out



def get_model_variables():
    vars = [v for v in tf.trainable_variables() if "batch_normalization" not in v.name
                                                             and "gamma" not in v.name
                                                              and "beta" not in v.name]
    vars = [v for v in tf.global_variables()+tf.local_variables() if "batch_normalization" not in v.name
                                                             and "gamma" not in v.name
                                                              and "beta" not in v.name
                                                              and "Adam" not in v.name]
    return vars


def get_decoder_init_state(cell, init_state, options):
    """
    initial values for (unidirectional lstm) decoder network from (equal depth bidirectional lstm)
    encoder hidden states. initially, the states of the forward and backward networks are concatenated
    and a fully connected layer is defined for each lastm parameter (c, h) mapping from encoder to
    decoder hidden size state
    """
    if options['bidir_encoder']:
        raise NotImplemented
    else:
        if options['encoder_state_as_decoder_init']:  # use encoder state for decoder init
            decoder_init_state = init_state
            #    decoder_init_state = cell.zero_state(
            #        dtype=tf.float32,
            #        batch_size=self.options['batch_size'] * self.options['beam_width']).clone(
            #                cell_state=tf.contrib.seq2seq.tile_batch(init_state, self.options['beam_width']))
        else:  # use zero state
            decoder_init_state = cell.zero_state(
                    dtype=tf.float32,
                    batch_size=options['batch_size'])
    return decoder_init_state


def get_attention_cell(cell, options, memories=None, memories_lengths=None):
    if options['attention_type'] is None:
        assert options['encoder_state_as_decoder_init'], \
            ("Decoder must use encoder final hidden state if"
             "no Attention mechanism is defined")
        attention_cell = cell
        return attention_cell
    assert (memories is not None) and (memories_lengths is not None), \
        "memory and memory_lengths tensors must be provided for attention cell"
    if options['attention_type'] is "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=options['decoder_num_hidden'],  # The depth of the query mechanism.
            memory=memories,  # The memory to query; usually the output of an RNN encoder
            memory_sequence_length=memories_lengths,  # Sequence lengths for the batch
            # entries in memory. If provided, the memory tensor rows are masked with zeros for values
            # past the respective sequence lengths.
            normalize=options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
            name='BahdanauAttention')
    elif options['attention_type'] is "monotonic_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(
            num_units=options['decoder_num_hidden'],  # The depth of the query mechanism.
            memory=memories,  # The memory to query; usually the output of an RNN encoder
            memory_sequence_length=memories_lengths,  # Sequence lengths for the batch
            # entries in memory. If provided, the memory tensor rows are masked with zeros for values
            # past the respective sequence lengths.
            normalize=options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
            name='BahdanauMonotonicAttention')
    elif options['attention_type'] is "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=options['decoder_num_hidden'],  # The depth of the query mechanism.
            memory=memories,  # The memory to query; usually the output of an Rif self.options['mode'] == 'train':
            memory_sequence_length=memories_lengths,  # Sequence lengths for the batch
            scale=options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
            name='LuongAttention')
    elif options['attention_type'] is "monotonic_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongMonotonicAttention(
            num_units=options['decoder_num_hidden'],  # The depth of the query mechanism.
            memory=memories,  # The memory to query; usually the output of an RNN encoder
            memory_sequence_length=memories_lengths,  # Sequence lengths for the batch
            scale=options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
            sigmoid_noise=0.0,
            score_bias_init=0.0,
            mode='parallel',
            name='LuongMonotonicAttention')
    else:
        raise ValueError("attention_type value not understood")
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell=cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=options['attention_layer_size'],
        alignment_history=options['alignment_history'],
        output_attention=options['output_attention'])
    return attention_cell


def get_attention_weights(self, sess):
    assert self.options['alignment_history']
    input_lengths, label_lengths, alignments = sess.run(
        [self.encoder_inputs_lengths, self.target_labels_lengths, self.final_state.alignment_history.stack()])
    return input_lengths, label_lengths, alignments


def lengths_mask(inputs, inputs_lengths, options):
    """
    makes a boolean mask for an inputs tensor
    length is assumed at dimension 1
    """
    lengths_transposed = tf.expand_dims(inputs_lengths, 1)
    range_ = tf.range(0, tf.shape(inputs)[1], 1)
    range_row = tf.expand_dims(range_, 0)
    # Use the logical operations to create a mask
    mask = tf.less(range_row, lengths_transposed)
    mask = tf.expand_dims(mask, -1)
    multiply = tf.constant([1, 1, options['num_classes']])
    mask = tf.tile(mask, multiply)
    return mask

class MultiLayerOutput(base.Layer):
    """2x Densely-connected layers class.
    Implements the operation:
    `outputs = activation(inputs * kernel + bias)`
    twice
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then it is
    flattened prior to the initial matrix multiply by `kernel`.
    Arguments:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
    Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the kernel matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the kernel matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel_constraint: Constraint function for the kernel matrix.
    bias_constraint: Constraint function for the bias.
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
    """

    def __init__(self, units,
               activation=[tf.nn.relu, None],
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(MultiLayerOutput, self).__init__(trainable=trainable, name=name,
                                    activity_regularizer=activity_regularizer,
                                    **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel1 = self.add_variable('kernel1',
                                        shape=[input_shape[-1].value, self.units[0]],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.kernel2 = self.add_variable('kernel2',
                                        shape=[self.units[0], self.units[1]],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.bias1 = self.add_variable('bias1',
                                    shape=[self.units[0],],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True) if self.use_bias else None
        self.bias2 = self.add_variable('bias2',
                                     shape=[self.units[1], ],
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint,
                                     dtype=self.dtype,
                                     trainable=True) if self.use_bias else None

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel1, [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units[0]]
                outputs.set_shape(output_shape)
            if self.use_bias:
                outputs = nn.bias_add(outputs, self.bias)
            outputs = tf.layers.batch_normalization(outputs)
            outputs = self.activation[0](outputs)
            outputs = standard_ops.tensordot(outputs, self.kernel2, [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units[1]]
                outputs.set_shape(output_shape)
            if self.use_bias:
                outputs = nn.bias_add(outputs, self.bias2)
            if self.activation[1]:
                outputs = self.activation[1](outputs)
        else:
            outputs = standard_ops.matmul(inputs, self.kernel1)
            if self.use_bias:
                outputs = nn.bias_add(outputs, self.bias1)
            outputs = tf.layers.batch_normalization(outputs)
            outputs = self.activation[0](outputs)
            outputs = standard_ops.matmul(outputs, self.kernel2)
            if self.use_bias:
                outputs = nn.bias_add(outputs, self.bias2)
            if self.activation[1]:
                outputs = self.activation[1](outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units[1])
