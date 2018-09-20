from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from model_utils import stacked_lstm, blstm_encoder
from metrics import char_accuracy, flatten_list
from data_provider import get_split, get_split2, get_split3
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

        if self.options['data_split'] == 'split1':
            _, self.encoder_inputs, \
            self.target_labels, \
            self.num_examples, \
            self.words, \
            self.decoder_inputs, \
            self.target_labels_lengths, \
            self.encoder_inputs_lengths, \
            self.decoder_inputs_lengths = get_split(options)
        elif self.options['data_split'] == 'split2':
            self.encoder_inputs, \
            self.target_labels, \
            self.num_examples, \
            self.words, \
            self.decoder_inputs, \
            self.target_labels_lengths, \
            self.encoder_inputs_lengths, \
            self.decoder_inputs_lengths = get_split2(options)
        elif self.options['data_split'] == 'split3':
            self.encoder_inputs, \
            self.target_labels, \
            self.num_examples, \
            self.words, \
            self.decoder_inputs, \
            self.target_labels_lengths, \
            self.encoder_inputs_lengths, \
            self.decoder_inputs_lengths = get_split3(options)

        self.number_of_steps_per_epoch = self.num_examples // self.batch_size
        self.number_of_steps = self.number_of_steps_per_epoch * options['num_epochs']

        if self.options['save_steps'] is None:
            self.save_steps = self.number_of_steps_per_epoch
        else:
            self.save_steps = self.options['save_steps']        

        self.init_global_step()

        self.max_decoding_steps = tf.reduce_max(self.encoder_inputs_lengths)

        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()

        else:
            self.build_train_graph()
            #self.build_inference_graph()

        if self.options['save_summaries']:
            self.merged_summaries = tf.summary.merge_all()

        if self.options['save'] or self.options['restore']:
            self.saver = tf.train.Saver(var_list=tf.global_variables(),
                                        max_to_keep=self.options['num_models_saved'])

        if self.options['save_graph'] or self.options['save_summaries']:
            self.writer = tf.summary.FileWriter(self.options['save_dir'])

    def build_train_graph(self):
        if self.options['has_encoder']:
            with tf.variable_scope('encoder'):
                if self.options['bidir_encoder']:
                    self.encoder_out, self.encoder_hidden = blstm_encoder(
                        input_forw=self.encoder_inputs, options=self.options)
                else:
                    self.encoder_out, self.encoder_hidden = stacked_lstm(
                        num_layers=self.options['encoder_num_layers'],
                        num_hidden=self.options['encoder_num_hidden'],
                        residual=self.options['residual_encoder'],
                        use_peepholes=True,
                        input_forw=self.encoder_inputs,
                        return_cell=False)
                    print("Encoder hidden:", self.encoder_hidden)

        else:
            self.encoder_out = self.encoder_inputs
            self.encoder_hidden = None

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
                residual=self.options['residual_decoder'],
                use_peepholes=True,
                input_forw=None,
                return_cell=True)

            if self.options['attention_type'] is None:
                assert self.options['encoder_state_as_decoder_init'], \
                    ("Decoder must use encoder final hidden state if"
                    "no Attention mechanism is defined")
                attn_cell = decoder_cell
            else:
                attention_mechanism = self.get_attention_mechanism()

                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=self.options['attention_layer_size'],
                    alignment_history=self.options['alignment_history'],
                    output_attention=self.options['output_attention']) # Luong: True, Bahdanau: False ?

            decoder_init_state = self.get_decoder_init_state(
                self.encoder_hidden, attn_cell)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell,
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

        with tf.variable_scope('loss_function'):

            lengths_transposed = tf.expand_dims(self.target_labels_lengths, 1)
            range_ = tf.range(0, tf.shape(self.target_labels)[1], 1)
            range_row = tf.expand_dims(range_, 0)
            # Use the logical operations to create a mask
            self.mask = tf.less(range_row, lengths_transposed)
            # Use the select operation to select between 1 or 0 for each value.
            # mask = tf.where(mask,
            #                 tf.ones(tf.shape(self.target_labels)[:2]),
            #                 tf.zeros(tf.shape(self.target_labels)[:2]))
            #mask = tf.reshape(mask, [-1, ])
            #target_labels_ = tf.reshape(self.target_labels, [-1, ])
            #predictions_ = tf.reshape(self.decoder_outputs, [-1, ])
            #target_labels_ = tf.boolean_mask(tf.reshape(self.target_labels, [-1, ]), mask)
            #predictions_ = tf.boolean_mask(tf.reshape(self.decoder_outputs, [-1, ]), mask)

            self.l2_loss = self.options['reg_constant'] * \
                           tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) \
                           for v in tf.trainable_variables()])

            if self.options['loss_fun'] is "mse":
                #self.mask = tf.reshape(self.mask, [-1, ])
                self.mask = tf.expand_dims(self.mask, -1)
                multiply = tf.constant([1, 1, self.options['num_classes']])
                self.mask = tf.reshape(tf.tile(self.mask, multiply), [-1, ])
                self.target_labels_ = tf.boolean_mask(
                    tf.reshape(self.target_labels, [-1, ]), self.mask)
                self.predictions_ = tf.boolean_mask(
                    tf.reshape(self.decoder_outputs, [-1, ]), self.mask)
                self.train_loss = tf.reduce_mean(
                    tf.pow(self.predictions_ - self.target_labels_, 2))
            # elif self.options['loss_fun'] is "cos":
            #     self.train_loss = tf.abs(tf.reduce_mean(
            #         tf.losses.cosine_distance(target_labels_, predictions_, dim=0)))
            elif self.options['loss_fun'] is 'concordance_cc':
                self.mask = tf.expand_dims(self.mask, -1)
                multiply = tf.constant([1, 1, self.options['num_classes']])
                self.mask = tf.tile(self.mask, multiply)
                if not self.options['ccc_loss_per_batch']:
                    ### MT - Loss (mean CCC loss per component, separately for all samples)
                    self.mask = tf.reshape(
                        tf.transpose(self.mask, (0, 2, 1)),
                        (self.options['batch_size']*self.options['num_classes'], -1))
                    self.target_labels_ = tf.reshape(
                        tf.transpose(self.target_labels, (0, 2, 1)),
                        (self.options['batch_size']*self.options['num_classes'], -1))
                    self.decoder_outputs_ = tf.reshape(
                        tf.transpose(self.decoder_outputs, (0, 2, 1)),
                        (self.options['batch_size']*self.options['num_classes'], -1))
                    self.train_loss = tf.reduce_mean(
                        tf.map_fn(
                            fn=self.concordance_cc,
                            elems=(self.decoder_outputs_, self.target_labels_, self.mask),
                            dtype=tf.float32))
                else:
                    ### PT - loss (mean CCC loss per component, per batch)
                    self.train_loss = tf.reduce_mean(
                        [self.concordance_cc(
                            (tf.reshape(self.decoder_outputs[:, :, i], (-1,)),
                             tf.reshape(self.target_labels[:, :, i], (-1,)),
                             tf.reshape(self.mask[:, :, i], (-1,))))
                         for i in range(self.options['num_classes'])])

            self.train_loss = self.train_loss + self.l2_loss

            if self.options['save_summaries']:
                tf.summary.scalar('train_loss', self.train_loss)
                tf.summary.scalar('l2_loss', self.l2_loss)

        with tf.variable_scope('training_parameters'):
            params = tf.trainable_variables()
            # clip by gradients
            max_gradient_norm = tf.constant(
                self.options['max_grad_norm'],
                dtype=tf.float32,
                name='max_gradient_norm')
            self.gradients = tf.gradients(self.train_loss, params)
            self.clipped_gradients, _ = tf.clip_by_global_norm(
                self.gradients, max_gradient_norm)
            # self.clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients]
            # Optimization
            self.global_step = tf.Variable(0, trainable=False)
            self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)
            initial_learn_rate = tf.constant(self.options['learn_rate'], tf.float32)
            if self.options['decay_steps'] is None:
                decay_steps = self.number_of_steps_per_epoch
            elif type(self.options['decay_steps']) is float:
                decay_steps = self.options['decay_steps'] * self.number_of_steps_per_epoch
            else:
                decay_steps = self.options['decay_steps']
            learn_rate = tf.train.exponential_decay(
                learning_rate=initial_learn_rate,
                global_step=self.global_step,
                decay_steps=decay_steps,
                decay_rate=self.options['learn_rate_decay'],
                staircase=self.options['staircase_decay'])
            self.optimizer = tf.train.AdamOptimizer(learn_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.update_step = self.optimizer.apply_gradients(
                    zip(self.clipped_gradients, params),
                    global_step=self.global_step)

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

                if np.isinf(loss) or np.isnan(loss):
                    self.ei = ei
                    self.do = do
                    self.tl = tl
                    return None

                if (self.train_era_step % self.save_steps == 0) \
                    and self.options['save']:
                    # print("saving model at global step %d..." % global_step)
                    self.save_model(
                        sess=sess,
                        save_path=self.options['save_model'] + "_epoch%d_step%d" % (epoch, step))
                    # print("model saved.")

                self.train_era_step += 1

        # save before closing
        if self.options['save']:
            self.save_model(sess=sess, save_path=self.options['save_model'] + "_final")
        if self.options['save_summaries']:
            self.save_summaries(sess=sess, summaries=self.merged_summaries)

    def build_inference_graph(self):
        """
        No differences between train and test graphs
        """
        #self.build_train_graph()
        encoder_inputs_pl = tf.placeholder(tf.float32, shape=(None, None, 20))  # placeholder for encoder inputs
        decoder_inputs_pl = tf.placeholder(tf.float32, shape=(None, None, 28))
        decoder_inputs_lengths_pl = tf.placeholder(tf.int32, shape=(None,))

        with tf.variable_scope('encoder'):

            if self.options['bidir_encoder']:
                self.encoder_out, encoder_hidden = blstm_encoder(input_forw=encoder_inputs_pl, options=self.options)
            else:
                self.encoder_out, encoder_hidden = stacked_lstm(
                    num_layers=self.options['encoder_num_layers'], num_hidden=self.options['encoder_num_hidden'],
                    residual=self.options['residual_encoder'], use_peepholes=True,
                    input_forw=encoder_inputs_pl, return_cell=False)
                print("Encoder hidden:", encoder_hidden)

        with tf.variable_scope('decoder_lstm'):
            ss_prob = self.options['ss_prob']
            self.sampling_prob = tf.constant(ss_prob, dtype=tf.float32)
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
                decoder_inputs_pl,
                decoder_inputs_lengths_pl,
                self.sampling_prob)
            #helper = tf.contrib.seq2seq.InferenceHelper(
            #    sample_fn=lambda outputs: outputs,
            #    sample_shape=[1],  # again because dim=1
            #    sample_dtype=dtypes.float32,
            #    start_inputs=start_tokens,
            #    end_fn=lambda sample_ids: False)
            decoder_cell = stacked_lstm(
                num_layers=self.options['decoder_num_layers'],num_hidden=self.options['decoder_num_hidden'],
                residual=self.options['residual_decoder'], use_peepholes=True,
                input_forw=None, return_cell=True)


            if self.options['attention_type'] is None:

                assert self.options['encoder_state_as_decoder_init'], "Decoder must use encoder final hidden state if no Attention mechanism is defined"
                attn_cell = decoder_cell

            else:

                attention_mechanism = self.get_attention_mechanism()
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=self.options['attention_layer_size'],
                    alignment_history=self.options['alignment_history'],
                    output_attention=self.options['output_attention'])

        decoder_init_state = self.get_decoder_init_state(encoder_hidden, attn_cell)

        decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell,
                helper=helper,
                initial_state=decoder_init_state,
                output_layer=tf.layers.Dense(self.options['num_classes']))

        outputs, self.final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.options['max_out_len'])
        self.decoder_outputs = outputs.rnn_output

    def predict(self, sess, num_steps=None, return_words=False):
        if num_steps is None:
            num_steps = self.number_of_steps_per_epoch
        loss_ = []
        if return_words:
            assert self.batch_size == 1, "batch_size must be set to 1 for getting loss per word"
            for i in range(num_steps):
                l_, w_ = sess.run([self.train_loss, self.words])
                loss_.append([l_, w_[0].decode("utf-8")])
                print("%d, %d, %.4f, %s" % (i, num_steps, l_, w_))
            loss_ = pd.DataFrame(loss_, columns=["loss", "word"])
            # return results aggregated per word
            loss_ = loss_.groupby("word").agg({"loss": [np.mean, np.sum]}).reset_index(drop=False)
        else:
            for i in range(num_steps):
                l_ = sess.run(self.train_loss)
                loss_.append(l_)
                print("%d, %d, %.4f" % (i, num_steps, l_))
        return loss_

    def predict_from_array(self, sess, mfcc_path, num_steps=None):
        mfcc = np.loadtxt(mfcc_path)
        mfcc = np.expand_dims(mfcc, 0)
        seq_length = mfcc.shape[1]
        if num_steps is not None:
            pred = []
            step_length = int(seq_length/num_steps)
            rem_length = seq_length - step_length * num_steps
            for i in range(num_steps+1):
                start_ = i*step_length
                print("start_ %d" % start_)
                if i != num_steps:
                    end_ = (i+1)*step_length
                    len_ = step_length
                else:
                    end_ = seq_length
                    len_ = rem_length
                print("end_ %d" % end_)
                print("len_ %d" % len_)
                feed_dict={self.encoder_inputs: mfcc[:, start_:end_, :],
                           self.decoder_inputs: np.ones((1, len_, self.options['num_classes'])),
                           self.decoder_inputs_lengths: [len_]}
                pred.append(sess.run(self.decoder_outputs, feed_dict=feed_dict))
        else:
            feed_dict={self.encoder_inputs: mfcc,
                       self.decoder_inputs: np.ones((1, seq_length, self.options['num_classes'])),
                       self.decoder_inputs_lengths: [seq_length]}
            pred = sess.run(self.decoder_outputs, feed_dict=feed_dict)
        return pred

    @property
    def learn_rate_decay_steps(self):
        if self.options['num_decay_steps'] is None:  # decay every epoch
            num_decay_steps = self.number_of_steps_per_epoch
        elif type(self.options['num_decay_steps']) is float:   # decay at a proportion to steps per epoch
            num_decay_steps = int(self.options['num_decay_steps'] * self.number_of_steps_per_epoch)
        else:  # explicitly specify decay steps
            num_decay_steps = self.options['num_decay_steps']
        return num_decay_steps

    def init_global_step(self, value=0):
        print("initializing global step at %d" % value)
        self.global_step = tf.Variable(value, trainable=False)
        self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)

    def restore_model(self, sess):
        print("reading model %s ..." % self.options['restore_model'])
        self.saver.restore(sess, self.options['restore_model'])
        print("model restored.")

    def save_model(self, sess, save_path):
        print("saving model %s ..." % save_path)
        self.saver.save(sess=sess, save_path=save_path)
        print("model saved.")

    def save_graph(self, sess):
        # writer = tf.summary.FileWriter(self.options['graph_save_path'])
        self.writer.add_graph(sess.graph)
        # writer = tf.summary.FileWriter(logdir='logdir', graph=graph)
        self.writer.flush()
        # self.writer.close()

    def save_summaries(self, sess, summaries):
        s, gs = sess.run([summaries, self.global_step])
        self.writer.add_summary(s, gs)
        self.writer.flush()

    def get_decoder_init_state(self, encoder_states, cell):
        """
        initial values for (unidirectional lstm) decoder network from (equal depth bidirectional lstm)
        encoder hidden states. initially, the states of the forward and backward networks are concatenated
        and a fully connected layer is defined for each lastm parameter (c, h) mapping from encoder to
        decoder hidden size state
        """
        if not self.options['bidir_encoder']:

            if self.options['encoder_state_as_decoder_init']:  # use encoder state for decoder init
                init_state = encoder_states

                #if self.options['mode'] == 'train':
                decoder_init_state = cell.zero_state(
                        dtype=tf.float32, batch_size=self.options['batch_size']).clone(
                        cell_state=init_state)
                #elif self.options['mode'] == 'test':
                #    decoder_init_state = cell.zero_state(
                #        dtype=tf.float32,
                #        batch_size=self.options['batch_size'] * self.options['beam_width']).clone(
                #                cell_state=tf.contrib.seq2seq.tile_batch(init_state, self.options['beam_width']))
            else:  # use zero state
                #if self.options['mode'] == 'train':
                decoder_init_state = cell.zero_state(
                        dtype=tf.float32,
                        batch_size=self.options['batch_size'])
                #elif self.options['mode'] == 'test':
                #    decoder_init_state = cell.zero_state(
                #        dtype=tf.float32,
                #        batch_size=self.options['batch_size'] * self.options['beam_width'])
            return decoder_init_state

        else:
            raise NotImplemented

    @staticmethod
    def concordance_cc(values_in):
        """Defines concordance loss for training the model.
        Args:
           prediction: prediction of the model.
           ground_truth: ground truth values.
           mask: True for non padded elements, False for padded elements
        Returns:
           The concordance value.
        """
        prediction, ground_truth, mask = values_in
        # apply mask to predictions and ground truth
        prediction = tf.boolean_mask(prediction, mask)
        ground_truth = tf.boolean_mask(ground_truth, mask)
        # compute CCC with masked tensors
        pred_mean, pred_var = tf.nn.moments(prediction, (0,))
        gt_mean, gt_var = tf.nn.moments(ground_truth, (0,))
        mean_cent_prod = tf.reduce_mean((prediction - pred_mean) * (ground_truth - gt_mean))
        return - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

    def get_attention_mechanism(self):
        if self.options['attention_type'] is "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.options['decoder_num_hidden'],  # The depth of the query mechanism.
                memory=self.encoder_out,  # The memory to query; usually the output of an RNN encoder
                memory_sequence_length=self.encoder_inputs_lengths,  # Sequence lengths for the batch
                # entries in memory. If provided, the memory tensor rows are masked with zeros for values
                # past the respective sequence lengths.
                normalize=self.options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
                name='BahdanauAttention')
        elif self.options['attention_type'] is "monotonic_bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(
                num_units=self.options['decoder_num_hidden'],  # The depth of the query mechanism.
                memory=self.encoder_out,  # The memory to query; usually the output of an RNN encoder
                memory_sequence_length=self.encoder_inputs_lengths,  # Sequence lengths for the batch
                # entries in memory. If provided, the memory tensor rows are masked with zeros for values
                # past the respective sequence lengths.
                normalize=self.options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
                name='BahdanauMonotonicAttention')
        elif self.options['attention_type'] is "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.options['decoder_num_hidden'],  # The depth of the query mechanism.
                memory=self.encoder_out,  # The memory to query; usually the output of an Rif self.options['mode'] == 'train':
                memory_sequence_length=self.encoder_inputs_lengths,  # Sequence lengths for the batch
                scale=self.options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
                name='LuongAttention')
        elif self.options['attention_type'] is "monotonic_luong":
            attention_mechanism = tf.contrib.seq2seq.LuongMonotonicAttention(
                num_units=self.options['decoder_num_hidden'],  # The depth of the query mechanism.
                memory=self.encoder_out,  # The memory to query; usually the output of an RNN encoder
                memory_sequence_length=self.encoder_inputs_lengths,  # Sequence lengths for the batch
                scale=self.options['attention_layer_norm'],  # boolean. Whether to normalize the energy term.
                sigmoid_noise=0.0,
                score_bias_init=0.0,
                mode='parallel',
                name='LuongMonotonicAttention')
        return attention_mechanism

    def get_attention_weights(self, sess):
        assert self.options['alignment_history']
        input_lengths, label_lengths, alignments = sess.run(
            [self.encoder_inputs_lengths, self.target_labels_lengths, self.final_state.alignment_history.stack()])
        return input_lengths, label_lengths, alignments
