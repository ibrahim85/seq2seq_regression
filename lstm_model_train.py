import tensorflow as tf
from tf_utils import start_interactive_session, set_gpu
from rnn_models import RNNModel
import numpy as np

set_gpu(0)

options = {
    'data_root_dir': "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtwN",
# "/home/michaeltrs/Projects/audio23d/data",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",

    'is_training' : True,
    'data_in': 'melf',  # mfcc, melf, melf_2d
    'split_name': 'train',
    'batch_size': 10,   # number of examples in queue either for training or inference
    'random_crop': True,
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|

    'has_encoder': True,
    'encoder_num_layers': 3,  # number of hidden layers in encoder lstm
    'residual_encoder': False,  # 
    'encoder_num_hidden': 256,  # number of hidden units in encoder lstm
    'encoder_dropout_keep_prob': 1.0,  # probability of keeping neuron, deprecated
    'encoder_layer_norm': True,
    'bidir_encoder': False,

    'loss_fun': "mse",  # "mse", "cos", "concordance_cc"
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
    'restore_model': "/data/mat10/Projects/audio23d/Models/lstm/lstm_all_melf_era1_epoch3_step301",

    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/Projects/audio23d/Models/lstm/lstm_all_melf_era1",
    'num_models_saved': 100,  # total number of models saved
    'save_steps': None,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/Projects/audio23d/Models/lstm/summaries",
    'save_summaries': False

          }


if __name__ == "__main__":

    model = RNNModel(options)

    sess = start_interactive_session()

    if options['save_graph']:
       model.save_graph(sess)

    if options['restore']:
        model.restore_model(sess)

    if options['is_training']:
        model.train(sess)
    else:
        loss = model.eval(sess, num_steps=None, return_words=False)
