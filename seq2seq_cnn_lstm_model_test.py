import tensorflow as tf
# from data_provider2 import get_split
from tf_utils import start_interactive_session, set_gpu
from mixed_seq2seq_models import CNNRNNSeq2SeqModel
import numpy as np

<<<<<<< HEAD
set_gpu(2)

options = {
    'data_root_dir': "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtw_antonio",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtwN",
=======
set_gpu(-1)

options = {
    'data_root_dir': '/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtwN',
>>>>>>> f798981d5c303deabd8107e5086cbc23a1985d2f
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",

    'is_training' : False,
<<<<<<< HEAD
    'data_in': 'melf',  # mcc, melf, melf_2d
    'split_name': 'devel',
    'batch_size': 1,   # number of examples in queue either for training or inference
    'random_crop': False,
=======
    'data_in': 'mfcc',  # mcc, melf, melf_2d
    #'max_seq_len': -20,
    'split_name': 'train',  # 'devel',
    #'use_rmse': False,
    'batch_size': 1,   # number of examples in queue either for training or inference
    #'reverse_time': False,
    #'shuffle': True,
    'random_crop': False,
    #'standardize_inputs_and_labels': True,
>>>>>>> f798981d5c303deabd8107e5086cbc23a1985d2f
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    
    '1dcnn_features_dims': [256, 256, 256],
    
    'has_decoder': True,
    'decoder_num_layers': 1,  # number of hidden layers in decoder lstm
    'residual_decoder': False,  # 
    'decoder_num_hidden': 256,  # number of hidden units in decoder lstm
    'encoder_state_as_decoder_init': False,  # bool. encoder state is used for decoder init state, else zero state
    'decoder_layer_norm': True,
    'decoder_dropout_keep_prob': 1.0,
    'attention_type': 'bahdanau',
    'output_attention': True,
    'attention_layer_size': 128,  # number of hidden units in attention layer
    'attention_layer_norm': True,
    'num_hidden_out': 128,  # number of hidden units in output fcn
    'alignment_history': False,

<<<<<<< HEAD
    'loss_fun': "concordance_cc",
=======
    #'max_in_len': None,  # maximum number of frames in input videos
    #'max_out_len': None,  # maximum number of characters in output text

    'loss_fun': "concordance_cc",
    #'ccc_loss_per_batch': False,  # set True for PT loss (mean per component/batch), False (mean per component per sample)
>>>>>>> f798981d5c303deabd8107e5086cbc23a1985d2f
    'reg_constant': 0.00,
    'max_grad_norm': 10.0, 
    'num_epochs': 30,  # number of epochs over dataset for training
    'start_epoch': 3,  # epoch to start
    'reset_global_step': False,
    'train_era_step': 1,  # start train step during current era, value of 0 saves the current model
    
    'learn_rate': 0.001,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'learn_rate_decay': 0.975,
    'staircase_decay': True,
    'decay_steps': 0.5,

    'ss_prob': 1.0,  # scheduled sampling probability for training. probability of passing decoder output as next
   
    'restore': True, # boolean. restore model from disk
    'restore_model': "/data/mat10/Projects/audio23d/Models/seq2seq_cnn_lstm/seq2seq_cnn_lstm_all_melf_cc_era1",
#"/data/mat10/Projects/audio23d/Models/seq2seq_cnn_lstm/seq2seq_cnn_lstm_seq10_era1_epoch10_step604",

    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/Projects/audio23d/Models/seq2seq_cnn_lstm/seq2seq_cnn_lstm_all_melf_cc_era1",
    'num_models_saved': 100,  # total number of models saved
    'save_steps': None,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/Projects/audio23d/Models/seq2seq_cnn_lstm/summaries",
    'save_summaries': False

          }

if __name__ == "__main__":
    ep =4 
    options['restore_model'] = options["save_model"] + "_epoch%d_step1254" % ep
    model = CNNRNNSeq2SeqModel(options)
    sess = start_interactive_session()
    model.restore_model(sess)
    loss = model.eval(sess, num_steps=None, return_words=False)
    print("loss at epoch %d is %.4f" % (ep, np.mean(loss)))
 
if False:
    model = CNNRNNSeq2SeqModel(options)
    losses = {}
<<<<<<< HEAD
    for ep in range(1, 54):
        options['restore_model'] = options["save_model"] + "_epoch%d_step6504" % ep
        #model = CNNRNNSeq2SeqModel(options)
=======
    for ep in range(1, 79):
        options['restore_model'] = "/data/mat10/Projects/audio23d/Models/dtwN/seq2seq_cnn_lstm/1dconv_res_mfcc_all_era1_epoch%d_step3536.index" % ep
        model = CNNRNNSeq2SeqModel(options)
>>>>>>> f798981d5c303deabd8107e5086cbc23a1985d2f
        sess = start_interactive_session()
        model.restore_model(sess)
        loss = model.eval(sess, num_steps=None, return_words=False)
        losses[ep] = np.mean(loss)
        tf.reset_default_graph()



