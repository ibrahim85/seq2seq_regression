from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from resnet_model_utils import ResNet, get_block_sizes


def backend_resnet(x_input, resnet_size=34, final_size=512, num_classes=None, frontend_3d=False, training=True, name="resnet"):

    with tf.name_scope(name):
        BATCH_SIZE, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS = x_input.get_shape().as_list()

        # RESNET
        video_input = tf.reshape(x_input, (-1, HEIGHT, WIDTH, NUM_CHANNELS))  # BATCH_SIZE*NUM_FRAMES

        #  = tf.cast(video_input, tf.float32)
        resnet = ResNet(resnet_size=resnet_size, bottleneck=False, num_classes=num_classes, num_filters=64,
                        kernel_size=7, conv_stride=2, first_pool_size=3, first_pool_stride=2,
                        second_pool_size=7, second_pool_stride=1, block_sizes=get_block_sizes(resnet_size),
                        block_strides=[1, 2, 2, 2], final_size=final_size, frontend_3d=frontend_3d)
        features = resnet.__call__(video_input, training=training)
        # features, end_points = resnet_v2.resnet_v1_50(video_input, None)
        features = tf.reshape(features, (BATCH_SIZE, -1, int(features.get_shape()[1])))  # NUM_FRAMES

        print("shape after resnet is %s" % features.get_shape())

    return features


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
        return tf.nn.rnn_cell.LSTMCell(layer_size,
                               use_peepholes=True,
                               cell_clip=100,
                               state_is_tuple=True)
        #return tf.contrib.rnn.LayerNormBasicLSTMCell(
        #                 num_units=layer_size,
        #                 forget_bias=1.0,
        #                 activation=tf.tanh,
        #                 layer_norm=layer_norm, 
        #                 norm_gain=1.0,
        #                 norm_shift=0.0,
        #                 dropout_keep_prob=dropout_keep_prob)
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


def temp_conv_block(inputs, out_dim, training, final_layer=False):
    inputs = tf.layers.conv1d(inputs=inputs, filters=out_dim, kernel_size=3, strides=1,
                              padding='same',  # 'same'
                              data_format='channels_last', dilation_rate=1, activation=None,
                              use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              bias_initializer=None,
                              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                              kernel_constraint=None, bias_constraint=None, trainable=True,
                              name=None, reuse=None)
    # print(inputs.get_shape())
    if not final_layer:
        inputs = tf.layers.batch_normalization(inputs,
                              axis=-1,
                              momentum=0.99,
                              epsilon=0.001,
                              center=True,
                              scale=True,
                              beta_initializer=tf.zeros_initializer(),
                              gamma_initializer=tf.ones_initializer(),
                              moving_mean_initializer=tf.zeros_initializer(),
                              moving_variance_initializer=tf.ones_initializer(),
                              beta_regularizer=None,
                              gamma_regularizer=None,
                              beta_constraint=None,
                              gamma_constraint=None,
                              training=training,
                              trainable=True,
                              name=None,
                              reuse=None,
                              renorm=False,
                              renorm_clipping=None,
                              renorm_momentum=0.99,
                              fused=None)
        inputs = tf.nn.relu(inputs)
    return inputs

def temp_res_conv_block(inputs, out_dim, training):
    """
    2x conv layers of size out_dim, batch norm and relu followed by adding initial input
    """
    inputs0 = tf.identity(inputs)
    inputs = temp_conv_block(inputs, out_dim, training, final_layer=False)
    inputs = temp_conv_block(inputs, out_dim, training, final_layer=False)
    return inputs + inputs0


def temp_conv_network(inputs, options):
    training = options['is_training']
    input_dim = int(inputs.get_shape()[-1])  # .as_list()
    print('Temporal convolution')
    print('input shape %s' % inputs.get_shape())
    for i, layer_dim in enumerate(options['1dcnn_features_dims']):
        inputs = temp_conv_block(inputs, layer_dim, training)
        print('input shape after %d temp conv layer %s' % (i, inputs.get_shape().as_list()))
    # print(inputs.get_shape())
    inputs = tf.layers.dense(inputs=inputs, units=128, activation=None, use_bias=True,  # shape[-1]
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None, trainable=True,
                             name=None, reuse=None)
    inputs = tf.layers.batch_normalization(inputs,
                              axis=-1,
                              momentum=0.99,
                              epsilon=0.001,
                              center=True,
                              scale=True,
                              beta_initializer=tf.zeros_initializer(),
                              gamma_initializer=tf.ones_initializer(),
                              moving_mean_initializer=tf.zeros_initializer(),
                              moving_variance_initializer=tf.ones_initializer(),
                              beta_regularizer=None,
                              gamma_regularizer=None,
                              beta_constraint=None,
                              gamma_constraint=None,
                              training=training,
                              trainable=True,
                              name=None,
                              reuse=None,
                              renorm=False,
                              renorm_clipping=None,
                              renorm_momentum=0.99,
                              fused=None)
    inputs = tf.nn.relu(inputs)
    print('input shape after 1st linear layer %s' % inputs.get_shape().as_list())
    inputs = tf.layers.dense(inputs=inputs, units=options['num_classes'], activation=None, use_bias=True,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None, trainable=True,
                             name=None, reuse=None)
    print('input shape after 2nd linear layer %s' % inputs.get_shape().as_list())
    return inputs


def temp_conv_network2(inputs, options):
    """
    does not include dense output network but instead has a 1dconv final layer with
    no normalization or nonlinearity   
    """
    training = options['is_training']
    input_dim = int(inputs.get_shape()[-1])  # .as_list()
    print('Temporal convolution')
    print('input shape %s' % inputs.get_shape())
    for i, layer_dim in enumerate(options['1dcnn_features_dims']):
        inputs = temp_conv_block(inputs, layer_dim, training)
        print('input shape after %d temp conv layer %s' % (i, inputs.get_shape().as_list()))
    inputs = temp_conv_block(inputs, options['num_classes'], training, final_layer=True)
    return inputs


def temp_res_conv_network(inputs, options):
    """
    does not include dense output network but instead has a 1dconv final layer with
    no normalization or nonlinearity
    """
    training = options['is_training']
    input_dim = int(inputs.get_shape()[-1])  # .as_list()
    print('Temporal convolution')
    print('input shape %s' % inputs.get_shape())
    for i, layer_dim in enumerate(options['1dcnn_features_dims']):
        if tf.shape(inputs)[-1] != layer_dim:
            inputs = temp_conv_block(inputs, layer_dim, training)
        inputs = temp_res_conv_block(inputs, layer_dim, training)
        print('input shape after %d temp conv layer %s' % (i, inputs.get_shape().as_list()))
    return inputs


def dense_1d_block(inputs, training, out_dim, bottleneck=False):
    """
    https://arxiv.org/pdf/1608.06993.pdf
    """
    inputs_l = tf.layers.batch_normalization(inputs, axis=-1, momentum=0.99, epsilon=0.001,
        center=True, scale=True, beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
        gamma_constraint=None,
        training=training, trainable=True, renorm=False, renorm_clipping=None,
        renorm_momentum=0.99)
    inputs_ = tf.nn.relu(inputs_l)
    if bottleneck:
        inputs_l = tf.layers.conv1d(inputs=inputs_l, filters=4*out_dim, kernel_size=1, strides=1,
                                  padding='same',  # 'same'
                                  data_format='channels_last', dilation_rate=1, activation=None,
                                  use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                  bias_initializer=None,
                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                  kernel_constraint=None, bias_constraint=None, trainable=True,
                                  name=None, reuse=None)
    outputs = tf.layers.conv1d(inputs=inputs_l, filters=out_dim, kernel_size=3, strides=1,
        padding='same',  # 'same'
        data_format='channels_last', dilation_rate=1, activation=None,
        use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(seed=None),
        bias_initializer=None,
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, trainable=True,
        name=None, reuse=None)
    inputs_next = tf.concat([inputs, outputs], axis=-1)
    return outputs, inputs_next


def dense_1d_conv_network(inputs_0, options):
    """
    https://arxiv.org/pdf/1608.06993.pdf
    """
    training = options['is_training']
    growth_rate = options['growth_rate']
    num_layers = options['num_layers']
    final_layer_dim = options['final_layer_dim']
    print('Dense Temporal Convolution')
    print('input shape %s' % inputs_0.get_shape())
    outputs = tf.layers.conv1d(inputs=inputs_0, filters=growth_rate, kernel_size=3, strides=1,
        padding='same',  # 'same'
        data_format='channels_last', dilation_rate=1, activation=None,
        use_bias=True, kernel_initializer=tf.keras.initializers.he_normal(seed=None),
        bias_initializer=None,
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, trainable=True,
        name=None, reuse=None)
    inputs_next = tf.concat([inputs_0, outputs], axis=-1)
    print('output shape after 1 temp conv layer %s' % (outputs.get_shape().as_list()))
    print('next feature map shape after 1 temp conv layer %s' % (inputs_next.get_shape().as_list()))
    for i in range(2, num_layers):
        outputs, inputs_next = dense_1d_block(inputs_next, training, growth_rate, bottleneck=False)
        print('output shape after %d temp conv layer %s' % (i, outputs.get_shape().as_list()))
        print('next feature map shape after %d temp conv layer %s' % (i, inputs_next.get_shape().as_list()))
    outputs, inputs_next = dense_1d_block(inputs_next, training, final_layer_dim, bottleneck=False)
    print('output shape after %d temp conv layer %s' % (i+1, outputs.get_shape().as_list()))
    return outputs


def bilstm(inputs, options):
    inputs_reverse = tf.reverse(inputs, axis=[1])
    with tf.variable_scope('forw_cell'):
        outputs_forw, _ = tf.nn.dynamic_rnn(
            cell=tf.contrib.rnn.LayerNormBasicLSTMCell(
                options['num_hidden'],
                forget_bias=1.0,
                activation=tf.tanh,
                layer_norm=True,
                norm_gain=1.0,
                norm_shift=0.0,
                dropout_keep_prob=options['dropout_keep_prob']),
            inputs=inputs,
            dtype=tf.float32)
    with tf.variable_scope('back_cell'):
        outputs_back, _ = tf.nn.dynamic_rnn(
            cell=tf.contrib.rnn.LayerNormBasicLSTMCell(
                options['num_hidden'],
                forget_bias=1.0,
                activation=tf.tanh,
                layer_norm=True,
                norm_gain=1.0,
                norm_shift=0.0,
                dropout_keep_prob=options['dropout_keep_prob']),
            inputs=inputs_reverse,
            dtype=tf.float32)
    outputs_back = tf.reverse(outputs_back, axis=[1])
    bilstm_out = tf.concat([outputs_forw, outputs_back], axis=-1)
    return bilstm_out


def RNMTplus_cell(inputs, options):
    res_inputs = bilstm(inputs, options)
    res_inputs = tf.layers.dropout(res_inputs,
                                   rate=1 - options['dropout_keep_prob'],  # drop probability
                                   training=options['is_training'])
    return inputs + res_inputs


def RNMTplus_net(inputs, options):
    assert int(inputs.get_shape()[-1]) == 2*options['num_hidden'], \
    "number of inputs in RNNplus cell must equal numbed of hidden units"
    inputs = tf.layers.dropout(inputs,
                               rate=1 - options['dropout_keep_prob'],  # drop probability
                               training=options['is_training'])
    for i in range(options['num_blocks']):
        with tf.variable_scope('layer_%d' % i):
            inputs = RNMTplus_cell(inputs, options)
    inputs = tf.layers.dense(inputs=inputs, units=options['num_classes'], activation=None, use_bias=True,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None, trainable=True,
                             name=None, reuse=None)
    return inputs


def cnn_audio_model2d(audio_frames, batch_size, nfilters=64, batch_norm=False, raw_model=False):
    """

    """
    # normalize_fn = tf.layers.batch_normalization
    with tf.variable_scope("audio_model"):
        ks = 3
        print("1.0", audio_frames)
        net = tf.layers.conv2d(audio_frames,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.1", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.2", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.3", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.4", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.5", net)
        if not raw_model:
            net = tf.layers.conv2d(net,
                                   filters=nfilters,
                                   kernel_size=(1, ks),
                                   strides=(1, 2),
                                   padding='valid',
                                   activation=tf.nn.relu)
            print("1.6", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(3, 1),
                               strides=(2, 1),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("2.1", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(3, 1),
                               strides=(2, 1),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("2.2", net)
        if not raw_model:
            net = tf.layers.conv2d(net,
                                   filters=nfilters,
                                   kernel_size=(3, 1),
                                   strides=(2, 1),
                                   padding='valid')
            if batch_norm:
                net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)
            print("2.3", net)
            #net = tf.layers.conv2d(net,
            #                       filters=nfilters,
            #                       kernel_size=(3, 1),
            #                       strides=(1, 1),
            #                       padding='valid')
            #if batch_norm:
            #    net = tf.layers.batch_normalization(net)
            #net = tf.nn.relu(net)
            #print("2.4", net)
        # net = tf.layers.flatten(net)
        #print(net)
        net = tf.reshape(net, (batch_size, -1, nfilters))
        print("3.1", net)
        net = tf.contrib.layers.fully_connected(net, 128)  # , activation_fn=None)
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("3.2", net)
        net = tf.reshape(net, [batch_size, -1, 128])
    return net


def cnn_audio_model3(audio_frames, batch_size, nfilters=64, batch_norm=False):
    """
    cnn audio model with no aggregation over results, kernels take individual time steps as inputs
    """
    # normalize_fn = tf.layers.batch_normalization
    with tf.variable_scope("audio_model"):
        ks = 3
        #nfilters = 64
        print("1.0", audio_frames)
        net = tf.layers.conv2d(audio_frames,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.1", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.2", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.3", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.4", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.5", net)
        net = tf.layers.conv2d(net,
                               filters=nfilters,
                               kernel_size=(1, ks),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("1.6", net)
        # net = tf.layers.flatten(net)
        #print(net)
        net = tf.reshape(net, (batch_size, -1, nfilters))
        print("3.1", net)
        net = tf.contrib.layers.fully_connected(net, 128)  # , activation_fn=None)
        print("3.2", net)
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        #net = tf.reshape(net, [batch_size, -1, 128])
    return net


def cnn_audio_model2d_res(audio_frames, batch_size, nfilters=256, batch_norm=False):
    """
    cnn audio model with no aggregation over results, kernels take individual time steps as inputs
    """
    # normalize_fn = tf.layers.batch_normalization
    with tf.variable_scope("audio_model"):
        ks = 3
        print("0.0", audio_frames)
        net = tf.layers.conv2d(audio_frames,
                               filters=32,
                               kernel_size=(1, 7),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("0.1", net)
        net0 = tf.layers.max_pooling2d(net,
                               pool_size=(1, 3),
                               strides=(1, 2),
                               padding='valid')
        print("1.0", net0)
        net = tf.layers.conv2d(net0,
                               filters=32,
                               kernel_size=(1, 3),
                               strides=(1, 1),
                               padding='same')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net,
                               filters=32,
                               kernel_size=(1, 3),
                               strides=(1, 1),
                               padding='same')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        # print("1.1", net)
        # net = tf.layers.conv2d(net,
        #                        filters=64,
        #                        kernel_size=(1, 3),
        #                        strides=(1, 1),
        #                        padding='same',
        #                        activation=tf.nn.relu)
        net1 = net0 + net
        print("1.2", net1)
        net1 = tf.layers.conv2d(net1,
                               filters=64,
                               kernel_size=(1, 3),
                               strides=(1, 2),
                               padding='valid')
        if batch_norm:
            net1 = tf.layers.batch_normalization(net1)
        net1 = tf.nn.relu(net1)
        print("2.0", net1)
        net = tf.layers.conv2d(net1,
                               filters=64,
                               kernel_size=(1, 3),
                               strides=(1, 1),
                               padding='same')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=(1, 3),
                               strides=(1, 1),
                               padding='same')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        # print("2.1", net)
        # net = tf.layers.conv2d(net,
        #                        filters=128,
        #                        kernel_size=(1, 3),
        #                        strides=(1, 1),
        #                        padding='same',
        #                        activation=tf.nn.relu)
        net2 = net1 + net
        print("2.2", net2)

        #net2 = tf.layers.conv2d(net2,
        #                       filters=128,
        #                       kernel_size=(1, 3),
        #                       strides=(1, 2),
        #                       padding='valid',
        #                       activation=tf.nn.relu)
        #print("3.0", net2)
        #net = tf.layers.conv2d(net2,
        #                       filters=128,
        #                       kernel_size=(1, 3),
        #                       strides=(1, 1),
        #                       padding='same',
        #                       activation=tf.nn.relu)
        # print("3.1", net)
        # net = tf.layers.conv2d(net,
        #                        filters=256,
        #                        kernel_size=(1, 3),
        #                        strides=(1, 1),
        #                        padding='same',
        #                        activation=tf.nn.relu)
        #net3 = net2 + net
        #print("3.2", net3)
        #net3 = tf.layers.conv2d(net3,
        #                        filters=256,
        #                        kernel_size=(1, 3),
        #                        strides=(1, 2),
        #                        padding='valid',
        #                        activation=tf.nn.relu)
        #print("4.0", net3)
        #net = tf.layers.conv2d(net3,
        #                       filters=256,
        #                       kernel_size=(1, 3),
        #                       strides=(1, 1),
        #                       padding='same',
        #                       activation=tf.nn.relu)
        # print("3.1", net)
        # net = tf.layers.conv2d(net,
        #                        filters=256,
        #                        kernel_size=(1, 3),
        #                        strides=(1, 1),
        #                        padding='same',
        #                        activation=tf.nn.relu)
        #net4 = net3 + net
        #print("4.2", net4)
        net4 = tf.layers.conv2d(net2,
                                filters=64,
                                kernel_size=(3, 1),
                                strides=(2, 1),
                                padding='valid')
        if batch_norm:
            net4 = tf.layers.batch_normalization(net4)
        net4 = tf.nn.relu(net4)
        print("5.0", net4)
        net = tf.layers.conv2d(net4,
                               filters=64,
                               kernel_size=(3, 1),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu)
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=(3, 1),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu)
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        # print("3.1", net)
        # net = tf.layers.conv2d(net,
        #                        filters=256,
        #                        kernel_size=(1, 3),
        #                        strides=(1, 1),
        #                        padding='same',
        #                        activation=tf.nn.relu)
        net5 = net4 + net
        print("5.2", net5)

        net5 = tf.layers.conv2d(net5,
                                filters=64,
                                kernel_size=(3, 1),
                                strides=(2, 1),
                                padding='valid')
        if batch_norm:
            net5 = tf.layers.batch_normalization(net5)
        net5 = tf.nn.relu(net5)
        print("5.0", net4)
        net = tf.layers.conv2d(net5,
                               filters=64,
                               kernel_size=(3, 1),
                               strides=(1, 1),
                               padding='same')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=(3, 1),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu)
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net6 = net5 + net
        print("5.2", net6)
        net = tf.layers.conv2d(net6,
                               filters=64,
                               kernel_size=(3, 1),
                               strides=(1, 1),
                               padding='valid')
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        # net = tf.layers.flatten(net)
        print("6.1", net)
        net = tf.layers.max_pooling2d(net,
                                       pool_size=(2, 2),
                                       strides=(1, 1),
                                       padding='valid')
        print("7.1", net)
        net = tf.reshape(net, (batch_size, -1, 64))
        print("5.1", net)
        net = tf.contrib.layers.fully_connected(net, 128)  # , activation_fn=None)
        if batch_norm:
            net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("5.2", net)
        net = tf.reshape(net, [batch_size, -1, 128])
    return net


def cnn1d_block_raw(tensor_in, filters_out, kernel_size, strides,
                    padding='valid', batch_norm=True, activation_fn=tf.nn.relu):
    net = tf.layers.conv1d(inputs=tensor_in, filters=filters_out,
                           kernel_size=kernel_size, strides=strides,
                           padding=padding)
    if batch_norm: net = tf.layers.batch_normalization(net)
    net = activation_fn(net)
    return net


def cnn_raw_audio1(audio_frames, return_mean=False):
    # front end
    net = cnn1d_block_raw(tensor_in=audio_frames, filters_out=16, kernel_size=7, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    net = tf.layers.max_pooling1d(inputs=net, pool_size=3, strides=2, padding='valid')
    # f32
    net = cnn1d_block_raw(tensor_in=net, filters_out=32, kernel_size=3, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    net = cnn1d_block_raw(tensor_in=net, filters_out=32, kernel_size=3, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    # f64
    net = cnn1d_block_raw(tensor_in=net, filters_out=64, kernel_size=3, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    net = cnn1d_block_raw(tensor_in=net, filters_out=64, kernel_size=3, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    # f128
    net = cnn1d_block_raw(tensor_in=net, filters_out=128, kernel_size=3, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    net = cnn1d_block_raw(tensor_in=net, filters_out=128, kernel_size=3, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    # f256
    net = cnn1d_block_raw(tensor_in=net, filters_out=256, kernel_size=3, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    net = cnn1d_block_raw(tensor_in=net, filters_out=256, kernel_size=3, strides=2,
                          padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    #net = cnn1d_block_raw(tensor_in=net, filters_out=256, kernel_size=3, strides=2,
    #                      padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
    # mean
    if return_mean:
        net = tf.reduce_mean(net, axis=1)
    return net


# def cnn_raw_audio_feature_extractor2(audio_frames):
#     # front end
#     net = cnn1d_block_raw(tensor_in=audio_frames, filters_out=16, kernel_size=7, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     net = tf.layers.max_pooling1d(inputs=net, pool_size=3, strides=2, padding='valid')
#     # f32
#     net = cnn1d_block_raw(tensor_in=net, filters_out=32, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     net = cnn1d_block_raw(tensor_in=net, filters_out=32, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     # f64
#     net = cnn1d_block_raw(tensor_in=net, filters_out=64, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     net = cnn1d_block_raw(tensor_in=net, filters_out=64, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     # f128
#     net = cnn1d_block_raw(tensor_in=net, filters_out=128, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     net = cnn1d_block_raw(tensor_in=net, filters_out=128, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     # f256
#     net = cnn1d_block_raw(tensor_in=net, filters_out=256, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     net = cnn1d_block_raw(tensor_in=net, filters_out=256, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     net = cnn1d_block_raw(tensor_in=net, filters_out=256, kernel_size=3, strides=2,
#                           padding='valid', batch_norm=True, activation_fn=tf.nn.relu)
#     # mean
#     net = tf.reduce_mean(net, axis=1)
#     return net
