import tensorflow as tf
# from data_provider2 import get_split
from tf_utils import start_interactive_session, set_gpu
from mixed_seq2seq_models import TransRNNSeq2SeqModel
import numpy as np
from collections import defaultdict

set_gpu(3)

options = defaultdict(
    lambda: None,  # Set default value to None.
    # Input params
    #default_batch_size=2,  # Maximum number of tokens per batch of examples.
    max_length=200,  # Maximum number of tokens per example.
    # Model params
    initializer_gain=1.0,  # Used in trainable variable initialization.
    vocab_size=28,  # Number of tokens defined in the vocabulary file.
    hidden_size=256,  # Model dimension in the hidden layers.
    num_hidden_layers=1,  # Number of layers in the encoder and decoder stacks.
    num_heads=4,  # Number of heads to use in multi-headed attention.
    filter_size=256,  # Inner layer dimension in the feedforward network.
    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    #----------------------------------------------------------------------------------------------------------------

    data_root_dir= "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtwN", 
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",

    is_training= True,
    max_seq_len=-3,
    split_name='train',
    data_in='mfcc',  # mcc, melf, melf_2d
    #use_rmse=False,
    batch_size= 128,   # number of examples in queue either for training or inference
    #'reverse_time False,
    #'shuffle True,
    #'random_crop': False,
    #'standardize_inputs_and_labels': True,
    mfcc_num_features= 20,  # 20,
    raw_audio_num_features= 533,  # 256,
    num_classes= 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    #'max_out_len_multiplier': 1.0,  # max_out_len = max_out_len_multiplier * max_in_len
    
    #'mfcc_gaussian_noise_std': 0.0,  # 0.05,
    #'label_gaussian_noise_std':0.0,

    has_decoder= True,
    decoder_num_layers= 1,  # number of hidden layers in decoder lstm
    residual_decoder= False,  #
    decoder_num_hidden= 256,  # number of hidden units in decoder lstm
    encoder_state_as_decoder_init= False,  # bool. encoder state is used for decoder init state, else zero state
    decoder_layer_norm= True,
    decoder_dropout_keep_prob= 1.0,
    attention_type= 'bahdanau',
    output_attention= True,
    attention_layer_size= 128,  # number of hidden units in attention layer
    attention_layer_norm= True,
    num_hidden_out= 128,  # number of hidden units in output fcn
    alignment_history= False,

    #'max_in_len': None,  # maximum number of frames in input videos
    #'max_out_len': None,  # maximum number of characters in output text

    loss_fun= "concordance_cc",
    ccc_loss_per_batch= False,  # set True for PT loss (mean per component/batch), False (mean per component per sample)
    reg_constant= 0.00,
    max_grad_norm= 10.0,
    num_epochs= 100,  # number of epochs over dataset for training
    start_epoch= 1,  # epoch to start
    reset_global_step= True,
    train_era_step= 1,  # start train step during current era, value of 0 saves the current model

    learn_rate= 0.001,  # initial learn rate corresponing top global step 0, or max lr for Adam
    learn_rate_decay= 0.975,
    staircase_decay= True,
    decay_steps= 0.5,

    ss_prob= 1.0,  # scheduled sampling probability for training. probability of passing decoder output as next
   
    restore= False, # boolean. restore model from disk
    restore_model= "/data/mat10/Projects/audio23d/Models/seq2seq_transf_lstm/seq2seq_transf_lstm_seq10_era1_epoch10_step604",

    save= True,  # boolean. save model to disk during current era
    save_model= "/data/mat10/Projects/audio23d/Models/dtwN/seq2seq_transf_lstm/seq2seq_transf_lstm_all_mfcc_cc_era1",
    num_models_saved= 100,  # total number of models saved
    save_steps= None,  # every how many steps to save model

    save_graph= False,
    save_dir= "/data/mat10/Projects/audio23d/Models/dtwN/seq2seq_transf_lstm/summaries",
    save_summaries= True

)

#from data_provider import get_split
#raw_audio, mfcc, target_labels, \
#num_examples, word, decoder_inputs, \
#label_lengths, mfcc_lengths, decoder_inputs_lengths = get_split(options)
#raw_audio, mfcc, label, num_examples, word = get_split()

model = TransRNNSeq2SeqModel(options)

sess = start_interactive_session()

#if options['save_graph']:
#    model.save_graph(sess)

if options['restore']:
    model.restore_model(sess)

if options['is_training']:
    model.train(sess)
else:
    loss = model.predict(sess, num_steps=None, return_words=True)

#pred = model.predict_from_array(sess, feed_dict)


# ra, mf, l, w, di, ll, mfl, dil = sess.run([raw_audio, mfcc, label, word,
#                                            decoder_inputs,
#                                            label_lengths, mfcc_lengths, decoder_inputs_lengths])
# print(ra.shape)
# print(mf.shape)
# print(l.shape)
# print(w.shape)
# print(di.shape)
# print(ll)

# raw_audio, mfcc, label, num_examples, word = \
#     get_split(batch_size=100, split_name='train', is_training=True)
# sess = start_interactive_session()
# ra, mf, l, w = sess.run([raw_audio, mfcc, label, word])


# from data_provider import get_paths, get_split3
# from pathlib import Path
# from model_utils import stacked_lstm, blstm_encoder
#
# #
# # p = get_paths(Path(options['data_root_dir']), options['split_name'])
# #
# encoder_inputs, tl, ne, w, di, tll, eil, dil = get_split3(options)
# # eo, eh = stacked_lstm(num_layers=options['encoder_num_layers'],
# #                         num_hidden=options['encoder_num_hidden'],
# #                         input_forw=encoder_inputs,
# #                         layer_norm=options['encoder_layer_norm'],
# #                         dropout_keep_prob=options['encoder_dropout_keep_prob'],
# #                         is_training=True,
# #                         residual=options['residual_encoder'],
# #                         use_peepholes=True,
# #                         return_cell=False)
#
# # sess = start_interactive_session()
# # ei_ = sess.run(ei)
# # ei = sess.run(model.encoder_inputs)
#
#
# #
# # num_layers=options['encoder_num_layers']
# # num_hidden=options['encoder_num_hidden']
# # input_forw=encoder_inputs
# # layer_norm=options['encoder_layer_norm']
# # dropout_keep_prob=1.0
# # is_training=True
# # residual=options['residual_encoder']
# # use_peepholes=True
# # return_cell=False
# #
# # if type(num_hidden) is int:
# #     num_hidden = [num_hidden] * num_layers
# # if not is_training:
# #     dropout_keep_prob = 1.0
# #
# # def cellfn(layer_size):
# #     return tf.contrib.rnn.LayerNormBasicLSTMCell(
# #                      num_units=layer_size,
# #                      forget_bias=1.0,
# #                      activation=tf.tanh,
# #                      layer_norm=layer_norm,
# #                      norm_gain=1.0,
# #                      norm_shift=0.0,
# #                      dropout_keep_prob=dropout_keep_prob)
# #
# # if residual:
# #     rnn_layers = [tf.contrib.rnn.ResidualWrapper(cellfn(layer_size))
# #                   for _, layer_size in enumerate(num_hidden)]
# # else:
# #     rnn_layers = [cellfn(layer_size) for _, layer_size in enumerate(num_hidden)]
# #
# # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
# # outputs, states = tf.nn.dynamic_rnn(multi_rnn_cell, input_forw, dtype=tf.float32)
#
# # ei = sess.run(encoder_inputs)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
