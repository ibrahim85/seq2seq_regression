from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import attention_layer
import ffn_layer
import transformer_model_utils
from models import BasicModel
from model_utils import lengths_mask
from losses import batch_masked_concordance_cc, batch_masked_mse, L2loss

class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size
    self.current_name = "test"

  def build(self):
    self.scale = tf.get_variable(self.current_name+"_layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable(self.current_name+"_layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def __call__(self, x, epsilon=1e-6):
    self.build()
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params, train):
    self.layer = layer
    self.postprocess_dropout = params["layer_postprocess_dropout"]
    self.train = train

    # Create normalization layer
    self.layer_norm = LayerNormalization(params["hidden_size"])

  def __call__(self, x, *args, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if self.train:
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
    return x + y


class EncoderStack(tf.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params, train):
    super(EncoderStack, self).__init__()
    self.layers = []
    for _ in range(params["num_hidden_layers"]):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"], train)
      feed_forward_network = ffn_layer.FeedFowardNetwork(
          params["hidden_size"], params["filter_size"],
          params["relu_dropout"], train, params["allow_ffn_pad"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params, train),
          PrePostProcessingWrapper(feed_forward_network, params, train)])

    # Create final layer normalization layer.
    self.output_normalization = LayerNormalization(params["hidden_size"])  # , scope_name="encoder_output")

  def __call__(self, encoder_inputs, attention_bias, inputs_padding):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer.
        [batch_size, 1, 1, input_length]
      inputs_padding: P

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("encoder_layer_%d" % n):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

    with tf.variable_scope("encoder_output"):
      output = self.output_normalization(encoder_inputs)

    return output  # self.output_normalization(encoder_inputs)


class SelfAttentionEncoder(BasicModel):
    def __init__(self, options):
        super(SelfAttentionEncoder, self).__init__(options=options)

        self.zero_mask = tf.cast(
            lengths_mask(self.encoder_inputs, self.encoder_inputs_lengths, self.options)[:,:,0],
            dtype=tf.int32)
        self.encoder_stack = EncoderStack(options, self.is_training)
        self.initializer = tf.variance_scaling_initializer(
            options["initializer_gain"], mode="fan_avg", distribution="uniform")

        # with tf.variable_scope("Transformer", initializer=initializer):
        self.attention_bias = transformer_model_utils.get_padding_bias(self.zero_mask)

        self.encoder_outputs = self.encode()

        self.decoder_outputs = tf.layers.dense(
                inputs=self.encoder_outputs,
                units=self.options['num_classes'], activation=None, use_bias=True,
                kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None, trainable=True,
                name=None, reuse=None)

        self.make_savers()
        self.get_loss()
        self.get_training_parameters()

    def encode(self):
        """Generate continuous representation for inputs.

        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            self.encoder_inputs = tf.layers.dense(
                inputs=self.encoder_inputs,
                units=self.options['hidden_size'], activation=None, use_bias=True,
                kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None, trainable=True,
                name=None, reuse=None)
            self.encoder_inputs = tf.layers.batch_normalization(self.encoder_inputs,
                                                   axis=-1,
                                                   momentum=0.99,
                                                   epsilon=0.001,
                                                   center=True,
                                                   scale=True,
                                                   beta_initializer=tf.zeros_initializer(),
                                                   gamma_initializer=tf.ones_initializer(),
                                                   moving_mean_initializer=tf.zeros_initializer(),
                                                   moving_variance_initializer=tf.ones_initializer(),
                                                   training=self.is_training,
                                                   trainable=True,
                                                   renorm=False,
                                                   renorm_momentum=0.99)
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            # embedded_inputs = self.embedding_softmax_layer(inputs)
            #
            inputs_padding = transformer_model_utils.get_padding(tf.cast(
                tf.reduce_max(100*self.encoder_inputs, [-1]),
                dtype=tf.int32))

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(self.encoder_inputs)[1]
                pos_encoding = transformer_model_utils.get_position_encoding(
                    length, self.options["hidden_size"])
                encoder_inputs = self.encoder_inputs + pos_encoding

            if self.is_training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, 1 - self.options["layer_postprocess_dropout"])

            return self.encoder_stack(encoder_inputs, self.attention_bias, inputs_padding)

    def get_loss(self):
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

    def get_training_parameters(self):
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
                _, ei, do, tl, gstep, loss, l2loss, lr = sess.run(
                    [self.update_step,
                     self.encoder_inputs,
                     self.decoder_outputs,
                     self.target_labels,
                     self.global_step,
                     self.train_loss,
                     self.l2_loss,
                     self.optimizer._lr])
                print("%d,%d,%d,%d,%d,%.4f,%.4f,%.8f"
                      % (gstep, epoch,
                         self.options['num_epochs'],
                         step,
                         self.number_of_steps_per_epoch,
                         loss, l2loss, lr))

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
    def eval(self, sess, num_steps=None, return_words=False):
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

    def predict(self, sess, mfcc_path, num_steps=None):
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
