from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from model_utils import blstm_encoder
from metrics import char_accuracy, flatten_list
from data_provider import get_split
from tqdm import tqdm
from tensorflow.contrib.rnn import LSTMStateTuple
import numpy as np

class RegressionModel:
    """
    03/07/18 : The 2nd seq2seq model trained on MVLRS
    Additions:
    ---------
    - mean encoder out as initial decoder hidden state. fc layer transforms concatenated
        blstm state to decoder size
    - added dropout after resnet
    """

    def __init__(self, options):

        self.options = options
        self.is_training = options['is_training']
        self.split_name = options['split_name']
        self.batch_size = options['batch_size']
        self.base_path = options['data_root_dir']
  
        self.epsilon = tf.constant(1e-10, dtype=tf.float32)
        # self.data_paths = get_data_paths(self.options)
        #
        # self.number_of_steps_per_epoch, self.number_of_steps = \
        #     get_number_of_steps(self.data_paths, self.options)
        _, \
            self.encoder_inputs, \
            self.target_labels, \
            self.num_examples, \
            self.words, \
            self.decoder_inputs, \
            self.target_labels_lengths, \
            self.encoder_inputs_lengths, \
            self.decoder_inputs_lengths = \
            get_split(batch_size=self.batch_size, base_path=self.base_path, 
                      split_name=self.split_name, is_training=self.is_training)
        
        # THSI SHOULD GO!!!
        #self.target_labels = tf.clip_by_value(self.target_labels, clip_value_min=-15., clip_value_max=15.)
        
        self.number_of_steps_per_epoch = self.num_examples // self.batch_size
        self.number_of_steps = self.number_of_steps_per_epoch * options['num_epochs']

        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            # self.encoder_inputs, self.target_labels, self.decoder_inputs, self.encoder_inputs_lengths, \
            # self.target_labels_lengths, self.decoder_inputs_lengths, self.max_input_len = \
            #     get_training_data_batch(self.data_paths, self.options)
            self.build_train_graph()

        else:
            # self.encoder_inputs, self.target_labels, self.decoder_inputs, self.encoder_inputs_lengths, \
            # self.target_labels_lengths, self.decoder_inputs_lengths, self.max_input_len = \
            #     get_inference_data_batch(self.data_paths, self.options)
            self.max_decoding_steps = tf.to_int32(
                tf.round(self.options['max_out_len_multiplier'] *
                         tf.to_float(tf.reduce_max(self.encoder_inputs_lengths))))
            self.build_inference_graph()

        # else:
        #     raise ValueError("options.mode must be either 'train' or 'test'")

        if self.options['save'] or self.options['restore']:
            self.saver = tf.train.Saver(var_list=tf.global_variables(),
                                        max_to_keep=self.options['num_models_saved'])

        # if self.options['restore']:
        #     self.restore_model(sess)
   
    def build_train_graph(self):
        ss_prob = self.options['ss_prob']

        # with tf.variable_scope('resnet'):
        #     features_res = backend_resnet(x_input=self.encoder_inputs,
        #                                   resnet_size=self.options['resnet_size'],
        #                                   num_classes=self.options['resnet_num_features'],
        #                                   training=True)
        #     if self.options['res_features_keep_prob'] != 1.0:
        #         features_res = tf.layers.dropout(features_res,
        #                                          rate=1. - self.options['res_features_keep_prob'],
        #                                          training=True,
        #                                          name='features_res_dropout')

        with tf.variable_scope('encoder_blstm'):
            # encoder_out, encoder_hidden = blstm_encoder(self.encoder_inputs, self.options)
            # rnn_layers = [tf.contrib.rnn.LayerNormBasicLSTMCell(self.options['encoder_num_hidden'], layer_norm=self.options['encoder_layer_norm']) for _ in range(self.options['encoder_num_layers'])]
            # create a RNN cell composed sequentially of a number of RNNCells
            # multi_rnn_cell_forw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            # multi_rnn_cell_back = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            # outputs, encoder_hidden = tf.nn.bidirectional_dynamic_rnn(
            #                 multi_rnn_cell_forw, multi_rnn_cell_back,
            #                 self.encoder_inputs, dtype=tf.float32)
            # encoder_out = tf.concat(outputs, 2)
            encoder_cell = [tf.contrib.rnn.LSTMCell(self.options['encoder_num_hidden']) for _ in range(self.options['encoder_num_layers'])]
            # encoder_cell = [tf.contrib.rnn.LayerNormBasicLSTMCell(self.options['encoder_num_hidden'], layer_norm=self.options['encoder_layer_norm']) for _ in range(self.options['encoder_num_layers'])]
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_cell, state_is_tuple=True)
            encoder_out, encoder_hidden = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs, dtype=tf.float32)

        with tf.variable_scope('decoder_lstm'):
            self.sampling_prob = tf.constant(ss_prob, dtype=tf.float32)
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
                self.decoder_inputs,
                self.decoder_inputs_lengths,
                self.sampling_prob)
            # decoder_cell = [tf.contrib.rnn.LayerNormBasicLSTMCell(self.options['decoder_num_hidden'], layer_norm=self.options['decoder_layer_norm']) for _ in range(self.options['decoder_num_layers'])]
            decoder_cell = [tf.contrib.rnn.LSTMCell(self.options['decoder_num_hidden']) for _ in range(self.options['decoder_num_layers'])]
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell, state_is_tuple=True)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.options['decoder_num_hidden'],  # The depth of the query mechanism.
                memory=encoder_out,  # The memory to query; usually the output of an RNN encoder
                memory_sequence_length=self.encoder_inputs_lengths,  # Sequence lengths for the batch
                # entries in memory. If provided, the memory tensor rows are masked with zeros for values
                # past the respective sequence lengths.
                normalize=self.options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
                name='BahdanauAttention')
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.options['attention_layer_size'],
                alignment_history=False,
                cell_input_fn=None,
                output_attention=False, # Luong: True, Bahdanau: False ?
                initial_cell_state=None,
                name=None)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, self.options['num_classes'])

            if self.options['encoder_state_as_decoder_init']:  # use encoder state for decoder init
                init_state = self.get_decoder_init_state(encoder_hidden)
                decoder_init_state = out_cell.zero_state(dtype=tf.float32, batch_size=self.options['batch_size']
                                                         ).clone(cell_state=init_state)
            else:  # use zero state
                decoder_init_state = out_cell.zero_state(dtype=tf.float32,
                                                         batch_size=self.options['batch_size'])

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell,
                helper=helper,
                initial_state=decoder_init_state)
            outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.options['max_out_len'])
            self.decoder_outputs = outputs.rnn_output
            # decoder_greedy_pred = tf.argmax(decoder_outputs, axis=2)

        with tf.variable_scope('loss_function'):
            # target_weights = tf.sequence_mask(self.target_labels_lengths,  # +1 for <eos> token
            #                                   maxlen=None,  # data_options['max_out_len']+1,
            #                                   dtype=tf.float32)
            # self.train_loss = tf.losses.mean_squared_error self.target_labels, self.decoder_outputs)  # , weights=target_weights)
            target_labels_ = tf.reshape(self.target_labels, [-1, 7]) # + self.epsilon
            predictions_ = tf.reshape(self.decoder_outputs, [-1, 7]) # + self.epsilon
            # self.train_loss = tf.abs(tf.losses.cosine_distance(target_labels_, predictions_, dim=0, weights=1.0 , reduction=tf.losses.Reduction.MEAN))/self.batch_size  # , wdim=0, eights=target_weights)
            # self.train_loss = tf.losses.mean_squared_error(target_labels_, predictions_)
            if self.options['loss_fun'] is "mse":
                # self.train_loss = tf.reduce_mean(tf.pow(predictions_ - target_labels_, 2))
                self.l2_loss = self.options['reg_constant'] * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
                self.train_loss = tf.reduce_mean(tf.pow(predictions_ - target_labels_, 2)) + self.l2_loss
            elif self.options['loss_fun'] is "cos":
                # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                # reg_constant = 0.01  # Choose an appropriate one.
                # loss = my_normal_loss + reg_constant * sum(reg_losses)
                self.l2_loss = self.options['reg_constant'] * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
                self.train_loss = tf.abs(tf.reduce_mean(tf.losses.cosine_distance(target_labels_, predictions_, dim=0))) + self.l2_loss

        with tf.variable_scope('training_parameters'):
            params = tf.trainable_variables()
            # clip by gradients
            max_gradient_norm = tf.constant(self.options['max_grad_norm'], dtype=tf.float32, name='max_gradient_norm')
            self.gradients = tf.gradients(self.train_loss, params)
            self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, max_gradient_norm)
            # self.clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients]
            # Optimization
            self.global_step = tf.Variable(0, trainable=False)
            self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)
            initial_learn_rate = tf.constant(self.options['learn_rate'], tf.float32)
            if self.options['decay_steps'] is None:
                decay_steps = self.number_of_steps_per_epoch
            else:
                decay_steps = self.options['decay_steps']
            learn_rate = tf.train.exponential_decay(learning_rate=initial_learn_rate, global_step=self.global_step,
                                                    decay_steps=decay_steps, 
                                                    decay_rate=self.options['learn_rate_decay'],
                                                    staircase=self.options['staircase_decay'])
            # self.optimizer = tf.train.MomentumOptimizer(learn_rate, momentum=0.9)#.minimize(cross_entropy)
            # learn_rate = tf.constant(self.options['learn_rate'], tf.float32)
            self.optimizer = tf.train.AdamOptimizer(learn_rate)
            # self.optimizer = tf.train.GradientDescentOptimizer(learn_rate)
            # self.update_step = self.optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            self.update_step = self.optimizer.apply_gradients(zip(self.clipped_gradients, params), global_step=self.global_step)

    def train(self, sess, number_of_steps=None):

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = self.optimizer.minimize(self.train_loss)

        if number_of_steps is not None:
            assert type(number_of_steps) is int
            start_epoch = 0
            num_epochs = 1
        else:
            number_of_steps = self.number_of_steps_per_epoch
            start_epoch = self.options['start_epoch']
            num_epochs = self.options['num_epochs']

        if self.options['restore']:
            self.restore_model(sess)

        if self.options['reset_global_step']:
            initial_global_step = self.global_step.assign(0)
            sess.run(initial_global_step)

        for epoch in range(start_epoch, start_epoch + num_epochs):

            for step in range(number_of_steps):

                _, ei, do, tl, gstep, loss, l2loss, lr, sp = sess.run(
                    [self.update_step,
                     # self.increment_global_step,
                     self.encoder_inputs,
                     self.decoder_outputs,
                     self.target_labels,
                     self.global_step,
                     self.train_loss,
                     self.l2_loss,
                     # self.accuracy,
                     # self.accuracy2,
                     self.optimizer._lr,
                     self.sampling_prob])
                print("%d,%d,%d,%d,%d,%.4f,%.4f,%.8f,%.4f"
                      % (gstep, epoch,
                         self.options['num_epochs'],
                         step,
                         self.number_of_steps_per_epoch,
                         loss, l2loss, lr, sp))
                
                if np.isinf(loss):
                    self.ei = ei
                    self.do = do
                    self.tl = tl
                    return None
      
                if (self.train_era_step % self.options['save_steps'] == 0) and self.options['save']:
                    # print("saving model at global step %d..." % global_step)
                    self.save_model(sess=sess, save_path=self.options['save_model'] + "_epoch%d_step%d" % (epoch, step))
                    # print("model saved.")

                self.train_era_step += 1

        # save before closing
        if self.options['save']:
            self.save_model(sess=sess, save_path=self.options['save_model'] + "_final")

    def build_inference_graph(self):
        """
        with beam search decoder
        """
        # with tf.variable_scope('resnet'):
        #     features_res = backend_resnet(x_input=self.encoder_inputs,
        #                                   resnet_size=self.options['resnet_size'],
        #                                   num_classes=self.options['resnet_num_features'],
        #                                   training=False)
        #     if self.options['res_features_keep_prob'] != 1.0:
        #         features_res = tf.layers.dropout(features_res,
        #                                          rate=1.-self.options['res_features_keep_prob'],
        #                                          training=False,
        #                                          name='features_res_dropout')

        with tf.variable_scope('encoder_blstm'):
            encoder_out, encoder_hidden = blstm_encoder(self.encoder_inputs, self.options)

        with tf.variable_scope('decoder_lstm'):

            decoder_cell = [tf.contrib.rnn.LSTMCell(self.options['decoder_num_hidden'])
                            for _ in range(self.options['encoder_num_layers'])]
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell, state_is_tuple=True)
            encoder_out_beam = tf.contrib.seq2seq.tile_batch(
                encoder_out, multiplier=self.options['beam_width'])
            encoder_inputs_lengths_beam = tf.contrib.seq2seq.tile_batch(
                self.encoder_inputs_lengths, multiplier=self.options['beam_width'])
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.options['decoder_num_hidden'],
                memory=encoder_out_beam,
                memory_sequence_length=encoder_inputs_lengths_beam,
                normalize=True,
                dtype=None,
                name='BahdanauAttention')
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.options['attention_layer_size'],
                alignment_history=False,
                cell_input_fn=None,
                output_attention=False,
                initial_cell_state=None,
                name=None)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, self.options['num_classes'])

            if self.options['encoder_state_as_decoder_init']:  # use encoder state for decoder init
                init_state = self.get_decoder_init_state(encoder_hidden)
                decoder_init_state = out_cell.zero_state(
                    dtype=tf.float32, batch_size=self.options['batch_size'] * self.options['beam_width']).clone(
                        cell_state=tf.contrib.seq2seq.tile_batch(init_state, self.options['beam_width']))
            else:  # use zero state
                decoder_init_state = out_cell.zero_state(
                    dtype=tf.float32, batch_size=self.options['batch_size'] * self.options['beam_width'])

            # decoder_init_state = out_cell.zero_state(
            #     dtype=tf.float32, batch_size=self.options['batch_size'] * self.options['beam_width'])
            # init_state = self.get_decoder_init_state(encoder_hidden)
            # decoder_init_state = out_cell.zero_state(
            #     dtype=tf.float32, batch_size=self.options['batch_size'] * self.options['beam_width']).clone(
            #     cell_state=tf.contrib.seq2seq.tile_batch(init_state, self.options['beam_width']))
            embedding_decoder = tf.diag(tf.ones(self.options['num_classes']))
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=out_cell,
                embedding=embedding_decoder,
                start_tokens=tf.fill([self.options['batch_size']], 27),
                end_token=28,
                initial_state=decoder_init_state,
                beam_width=self.options['beam_width'],
                length_penalty_weight=0.0)
            self.outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=False, maximum_iterations=self.max_decoding_steps,
                swap_memory=True)
            beam_search_outputs = self.outputs.predicted_ids
            self.best_output = beam_search_outputs[:, :, 0]   ### IS THIS THE BEST???
            # CHECK : https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/FinalBeamSearchDecoderOutput

    def predict(self, sess, num_steps=None):
        # sometimes decoder outputs -1s at the end of the sequence, replace those with 0s
        # def replace_(s, vout=-1, vin=0):
        #     s[s == vout] = vin
        #     return s
        if self.options['restore']:
            self.restore_model(sess)
        if num_steps is None:
            num_steps = self.number_of_steps_per_epoch
        res = []
        for step in tqdm(range(num_steps)):
            tl, pred = sess.run([self.target_labels, self.best_output])
            res.append([tl, pred])
        # labels_ = flatten_list([decrypt(res_[0]) for res_ in res])
        # predictions_ = flatten_list([decrypt(replace_(res_[1])) for res_ in res])
        return res


    def restore_model(self, sess):
        print("reading model %s ..." % self.options['restore_model'])
        self.saver.restore(sess, self.options['restore_model'])
        print("model restored.")

    def save_model(self, sess, save_path):
        print("saving model %s ..." % save_path)
        self.saver.save(sess=sess, save_path=save_path)
        print("model saved.")

    def get_decoder_init_state(self, encoder_states):
        """
        initial values for (unidirectional lstm) decoder network from (equal depth bidirectional lstm)
        encoder hidden states. initially, the states of the forward and backward networks are concatenated
        and a fully connected layer is defined for each lastm parameter (c, h) mapping from encoder to
        decoder hidden size state
        """

        def encoder2decoder_init_state(encoder_hidden, decoder_hidden_size, name='encoder_decoder_hidden'):
            decoder_hidden = tf.layers.dense(inputs=encoder_hidden, units=decoder_hidden_size,
                                                 activation=tf.nn.relu, name=name)
            return decoder_hidden

        encoder_depth_ = len(encoder_states[0])
        # init_state = [LSTMStateTuple(c=tf.concat([encoder_states[0][i].c, encoder_states[1][i].c], axis=1),
        #                              h=tf.concat([encoder_states[0][i].h, encoder_states[1][i].h], axis=1))
        #               for i in range(encoder_depth_)]
        init_state = [[tf.concat([encoder_states[0][i].c, encoder_states[1][i].c], axis=1),
                       tf.concat([encoder_states[0][i].h, encoder_states[1][i].h], axis=1)]
                      for i in range(encoder_depth_)]
        init_state1 = [
            [encoder2decoder_init_state(state[0], self.options['decoder_num_hidden'], name="enc_c2dec_c_%d" % (i+1)),
             encoder2decoder_init_state(state[1], self.options['decoder_num_hidden'], name="enc_h2dec_h_%d" % (i+1))]
                       for i, state in enumerate(init_state)]
        init_state2 = [LSTMStateTuple(c=eh_state[0], h=eh_state[1]) for eh_state in init_state1]

        return tuple(init_state2)


