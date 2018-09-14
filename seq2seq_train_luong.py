import tensorflow as tf
# from data_provider2 import get_split
from tf_utils import start_interactive_session, set_gpu
from regression_model import RegressionModel
import numpy as np

set_gpu(5)

options = {
    'data_root_dir': "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",

    'is_training' : False,
    'split_name': 'devel',
    'batch_size': 512,   # number of examples in queue either for training or inference
    'reverse_time': False,
    'shuffle': True,
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    'max_out_len_multiplier': 0.70,  # max_out_len = max_out_len_multiplier * max_in_len

    'encoder_num_layers': 3,  # number of hidden layers in encoder lstm
    'residual_encoder': False,  # 
    'encoder_num_hidden': 256,  # number of hidden units in encoder lstm
    'encoder_dropout_keep_prob' : None,  # probability of keeping neuron, deprecated
    'encoder_layer_norm': True,
    'bidir_encoder': False,

    'decoder_num_layers': 3,  # number of hidden layers in decoder lstm
    'residual_decoder': False,  # 
    'decoder_num_hidden': 256,  # number of hidden units in decoder lstm
    'encoder_state_as_decoder_init' : False,  # bool. encoder state is used for decoder init state, else zero state
    'decoder_layer_norm': True,

    'attention_type': 'luong',
    'output_attention': True,
    'attention_layer_size': 256,  # number of hidden units in attention layer
    'attention_layer_norm': True,
    'num_hidden_out': 128,  # number of hidden units in output fcn
    'alignment_history': True,

    # 'beam_width': 20,  # number of best solutions used in beam decoder
    'max_in_len': None,  # maximum number of frames in input videos
    'max_out_len': None,  # maximum number of characters in output text

    'loss_fun': "concordance_cc",  # "mse", "cos", "concordance_cc"
    'reg_constant': 0.00,
    'max_grad_norm': 1.0, 
    'num_epochs': 10,  # number of epochs over dataset for training
    'start_epoch': 1,  # epoch to start
    'reset_global_step': False,
    'train_era_step': 1,  # start train step during current era, value of 0 saves the current model
    
    'learn_rate': 0.00,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'learn_rate_decay': 0.9,
    'staircase_decay': True,
    'decay_steps': 0.5,

    'ss_prob': 1.0,  # scheduled sampling probability for training. probability of passing decoder output as next
   
    'restore': True, # boolean. restore model from disk
    'restore_model': "/data/mat10/MSc_Project/audio_to_3dvideo/Models/model6_clean/luong/seq2seq_train_largemodel_cc_ss100_luong_0init_outstate_era1_final",  #"/data/mat10/MSc_Project/audio_to_3dvideo/Models/model4/luong/seq2seq_train_largemodel_cc_ss100_luong_0init_outstate_era1_final",  #/seq2seq_train_cc_ss015_luong_0init_outstate_era2_epoch4_step2627",  # path to model to restore

    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/MSc_Project/audio_to_3dvideo/Models/model6_clean/luong/seq2seq_train_largemodel_cc_ss100_luong_0init_outstate_era1",
    'num_models_saved': 50,  # total number of models saved
    'save_steps': 2000,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/MSc_Project/audio_to_3dvideo/Models/model3_all_mfcc/summaries",
    'save_summaries': True

          }


#from data_provider import get_split
#raw_audio, mfcc, target_labels, \
#num_examples, word, decoder_inputs, \
#label_lengths, mfcc_lengths, decoder_inputs_lengths = get_split(options)
#raw_audio, mfcc, label, num_examples, word = get_split()

model = RegressionModel(options)

sess = start_interactive_session()

if options['restore']:
    model.restore_model(sess)

#model.train(sess)

loss = model.predict(sess)
#np.mean(loss)


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
