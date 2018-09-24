from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from model_utils import stacked_lstm, blstm_encoder, get_attention_cell, get_decoder_init_state, lengths_mask
from metrics import char_accuracy, flatten_list
from losses import batch_masked_concordance_cc, batch_masked_mse, L2loss
from data_provider import get_split, get_split2, get_split3
from tqdm import tqdm
from tensorflow.contrib.rnn import LSTMStateTuple
import numpy as np
import pandas as pd


class BasicModel:
    """
    Model class with basic functionality
    options: (dict) all  model and training options/parameters
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

    def build_train_graph(self):
        pass

    def train(self, sess, number_of_steps=None, reset_global_step=False):
        pass

    def build_inference_graph(self):
        pass

    def predict(self, sess, num_steps=None):
        pass

    @property
    def learn_rate_decay_steps(self):
        if self.options['num_decay_steps'] is None:  # decay every epoch
            num_decay_steps = self.number_of_steps_per_epoch
        elif type(self.options['num_decay_steps']) is float:  # decay at a proportion to steps per epoch
            num_decay_steps = int(self.options['num_decay_steps'] * self.number_of_steps_per_epoch)
        else:  # explicitly specify decay steps
            num_decay_steps = self.options['num_decay_steps']
        return num_decay_steps

    def init_global_step(self, value=0):
        print("initializing global step at %d" % value)
        self.global_step = tf.Variable(value, trainable=False)
        self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)

    def make_savers(self):
        """
        makes all tensorflow saver objects as defined in self.options dict
        to be run after graph is defined
        """
        if self.options['save'] or self.options['restore']:
            self.saver = tf.train.Saver(var_list=tf.global_variables(),
                                        max_to_keep=self.options['num_models_saved'])
        if self.options['save_graph']:
            self.graph_writer = tf.summary.FileWriter(self.options['save_dir'])
        if self.options['save_summaries']:
            self.summary_writer = tf.summary.FileWriter(self.options['save_dir'])
            self.merged_summaries = tf.summary.merge_all()

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
        self.graph_writer.add_graph(sess.graph)
        self.graph_writer.flush()
        # self.writer.close()

    def save_summaries(self, sess, summaries):
        s, gs = sess.run([summaries, self.global_step])
        self.summary_writer.add_summary(s, gs)
        self.summary_writer.flush()


class RegressionModel(BasicModel):
    """

    """

    def __init__(self, options):
        super(RegressionModel, self).__init__(options=options)
        if self.is_training:
            self.train_era_step = self.options['train_era_step']
            self.build_train_graph()
        else:
            self.build_train_graph()
        self.make_savers()

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
                        input_forw=self.encoder_inputs,
                        layer_norm=self.options['encoder_layer_norm'],
                        dropout_keep_prob=self.options['encoder_dropout_keep_prob'],
                        is_training=True,
                        residual=self.options['residual_encoder'],
                        use_peepholes=True,
                        return_cell=False)
                    print("Encoder hidden:", self.encoder_hidden)
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
                                                            init_state=self.encoder_hidden,
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

        with tf.variable_scope('loss_function'):
            self.mask = lengths_mask(self.target_labels, self.target_labels_lengths, self.options)
            if self.options['loss_fun'] is "mse":
                self.train_loss = batch_masked_mse(
                    (self.decoder_outputs, self.target_labels, self.mask), self.options)
            elif self.options['loss_fun'] is 'concordance_cc':
                self.train_loss = batch_masked_concordance_cc(
                    (self.decoder_outputs, self.target_labels, self.mask), self.options)
            self.l2_loss = L2loss(self.options['reg_constant'])
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
                     self.encoder_inputs,
                     self.decoder_outputs,
                     self.target_labels,
                     self.global_step,
                     self.train_loss,
                     self.l2_loss,
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
        if self.options['save'] and (self.save_steps != self.number_of_steps_per_epoch):
            self.save_model(sess=sess, save_path=self.options['save_model'] + "_final")
        if self.options['save_summaries']:
            self.save_summaries(sess=sess, summaries=self.merged_summaries)

    def build_inference_graph(self):
        """
        No differences between train and test graphs
        """
        self.build_train_graph()

    # make this into eval
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
            loss_ = loss_.groupby("word").agg({"loss": [np.mean, np.std]}).reset_index(drop=False)
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
