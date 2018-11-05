import tensorflow as tf
# from data_provider2 import get_split
from tf_utils import start_interactive_session, set_gpu
from rnn_models import RNNModel
import numpy as np

set_gpu(-1)

options = {
    'data_root_dir': "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtwN",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",

    'is_training' : False,#True,
    'data_in': 'melf',  # mfcc, melf, melf_2d
    'random_crop': False,
    'split_name': 'devel',
    'use_rmse': False,
    'batch_size': 1,   # number of examples in queue either for training or inference
    #'reverse_time': False,
    #'shuffle': True,
    #'random_crop': False,
    #'standardize_inputs_and_labels': True,
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    #'max_out_len_multiplier': 1.0,  # max_out_len = max_out_len_multiplier * max_in_len

    #'mfcc_gaussian_noise_std': 0.0,  # 0.05,
    #'label_gaussian_noise_std':0.0,

    'has_encoder': True,
    'encoder_num_layers': 1,  # number of hidden layers in encoder lstm
    'residual_encoder': False,  # 
    'encoder_num_hidden': 256,  # number of hidden units in encoder lstm
    'encoder_dropout_keep_prob' : 1.0,  # probability of keeping neuron, deprecated
    'encoder_layer_norm': True,
    'bidir_encoder': False,

    'loss_fun': "concordance_cc",
    #'ccc_loss_per_batch': False,  # set True for PT loss (mean per component/batch), False (mean per component per sample)
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

    'restore': True, # boolean. restore model from disk
    'restore_model': "/data/mat10/Projects/audio23d/Models/lstm/lstm_all_melf_era1_epoch10_step301",

    'save': True,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/Projects/audio23d/Models/lstm/lstm_all_melf_era1",
    'num_models_saved': 100,  # total number of models saved
    'save_steps': None,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/Projects/audio23d/Models/lstm/summaries",
    'save_summaries': True

          }

if __name__ == "__main__":

    if False:
        model = RNNModel(options)
        sess = start_interactive_session()
        if options['save_graph']:
            model.save_graph(sess)
        if options['restore']:
            model.restore_model(sess)
        if options['is_training']:
            model.train(sess)
        else:
            loss = model.eval(sess, return_words=False)

    if True:
        losses = {}
        for ep in range(1, 4):
            options['restore_model'] = "/data/mat10/Projects/audio23d/Models/dtwN/lstm/lstm_all_melf_cc_era1_epoch%d_step3536" % ep
            model = RNNModel(options)
            sess = start_interactive_session()
            if options['restore']:
                model.restore_model(sess)
            loss = model.eval(sess, num_steps=None, return_words=False)
            losses[ep] = np.mean(loss)
            tf.reset_default_graph()
