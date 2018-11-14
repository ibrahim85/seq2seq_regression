import tensorflow as tf
# from data_provider2 import get_split
from tf_utils import start_interactive_session, set_gpu
from mixed_seq_models import CNNRNNModel3
import numpy as np

set_gpu(0)

options = {
    'data_root_dir': "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtw_antonio",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtwN",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",

    'is_training' : True,
    'data_in': 'melf',  # mcc, melf, melf_2d
    'split_name': 'train',
    'batch_size': 20,   # number of examples in queue either for training or inference
    'random_crop': False,
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    
    #'1dcnn_features_dims': [256, 256, 256],
    'batch_norm': True,

    'has_decoder': True,
    'decoder_num_layers': 1,  # number of hidden layers in decoder lstm
    'residual_decoder': False,  # 
    'decoder_num_hidden': 256,  # number of hidden units in decoder lstm
    'decoder_layer_norm': True,
    'decoder_dropout_keep_prob': 1.0,
    'num_hidden_out': 128,  # number of hidden units in output fcn

    'loss_fun': "concordance_cc",
    'reg_constant': 0.00,
    'max_grad_norm': 10.0, 
    'num_epochs': 30,  # number of epochs over dataset for training
    'start_epoch': 1,  # epoch to start
    'reset_global_step': True,
    'train_era_step': 1,  # start train step during current era, value of 0 saves the current model
    
    'learn_rate': 0.001,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'learn_rate_decay': 0.975,
    'staircase_decay': True,
    'decay_steps': 0.5,

    'ss_prob': 1.0,  # scheduled sampling probability for training. probability of passing decoder output as next
   
    'restore': False, # boolean. restore model from disk
    'restore_model': "/data/mat10/Projects/audio23d/Models/seq_cnn3_lstm/seq2seq_cnn3_lstm_all_melf_cc_era1_epoch7_step4459",
#"/data/mat10/Projects/audio23d/Models/seq2seq_cnn_lstm/seq2seq_cnn_lstm_seq10_era1_epoch10_step604",

    'save': True,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/Projects/audio23d/Models/seq_cnn3_lstm/seq2seq_cnn3_lstm_batchnorm_all_melf_cc_era2",
    'num_models_saved': 100,  # total number of models saved
    'save_steps': 500,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/Projects/audio23d/Models/seq_cnn3_lstm/summaries",
    'save_summaries': True

          }

if __name__ == "__main__":

    model = CNNRNNModel3(options)

    sess = start_interactive_session()

    if options['save_graph']:
       model.save_graph(sess)

    if options['restore']:
        model.restore_model(sess)

    if options['is_training']:
        model.train(sess)
    else:
        loss = model.eval(sess, num_steps=None, return_words=False)
