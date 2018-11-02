import tensorflow as tf
# from data_provider2 import get_split
from tf_utils import start_interactive_session, set_gpu
from rnn_seq2seq_models import RNNSeq2SeqModel
import numpy as np

set_gpu(0)

options = {
    'data_root_dir': "/home/michaeltrs/Projects/audio23d/data",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtwN",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",
    'is_training': True,
    'random_crop': True,
    'split_name': "train",
    'data_in': 'melf',  # mcc, melf, melf_2d
    'batch_size': 128,   # number of examples in queue either for training or inference
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|

    'has_encoder': True,
    'encoder_num_layers': 1,  # number of hidden layers in encoder lstm
    'residual_encoder': False,  # 
    'encoder_num_hidden': 256,  # number of hidden units in encoder lstm
    'encoder_dropout_keep_prob' : 1.0,  # probability of keeping neuron, deprecated
    'encoder_layer_norm': True,
    'bidir_encoder': False,
    
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
    'alignment_history': True,

    'loss_fun': "mse",  # "concordance_cc",
    'reg_constant': 0.00,
    'max_grad_norm': 10.0, 
    'num_epochs': 100,  # number of epochs over dataset for training
    'start_epoch': 1,  # epoch to start
    'reset_global_step': True,
    'train_era_step': 1,  # start train step during current era, value of 0 saves the current model
    
    'learn_rate': 0.001,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'learn_rate_decay': 0.975,
    'staircase_decay': True,
    'decay_steps': 0.5,

    'ss_prob': 1.0,  # scheduled sampling probability for training. probability of passing decoder output as next
   
    'restore': False, # boolean. restore model from disk
    'restore_model': "/data/mat10/Projects/audio23d/Models/seq2seq_lstm_lstm/seq2seq_lstm_lstm_all_era1_epoch30_step1208",

    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/Projects/audio23d/Models/dtwN/seq2seq_lstm_lstm/seq2seq_lstm_lstm_all_mfcc_cc_era1",
    'num_models_saved': 100,  # total number of models saved
    'save_steps': None,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/Projects/audio23d/Models/dtwN/seq2seq_lstm_lstm/summaries",
    'save_summaries': False

          }

if __name__ == "__main__":

    model = RNNSeq2SeqModel(options)

    sess = start_interactive_session()

    if options['save_graph']:
       model.save_graph(sess)

    if options['restore']:
        model.restore_model(sess)

    if options['is_training']:
        model.train(sess)
    else:
        loss = model.eval(sess, num_steps=None)#, return_words=True)