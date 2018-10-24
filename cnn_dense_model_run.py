from cnn_models import DenseNet1D
from tf_utils import start_interactive_session, set_gpu
import numpy as np

set_gpu(1)

options = {
    'data_root_dir': "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
# "/home/michaeltrs/Projects/audio23d/data",  # "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",  # enhanced",

    'is_training' : True,
    'split_name': "train",  # 'devel',
    'data_in': 'mfcc',  # mfcc, melf, melf_2d
    # 'use_rmse': False,
    'batch_size': 32,   # number of examples in queue either for training or inference
    # 'reverse_time': False,
    # 'shuffle': True,
    # 'random_crop': False,
    # 'standardize_inputs_and_labels': False,
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    # 'max_out_len_multiplier': 1.0,  # max_out_len = max_out_len_multiplier * max_in_len
    'growth_rate': 20,
    'num_layers' : 7,
    'final_layer_dim': 128,

    'loss_fun': "concordance_cc",
    #'ccc_loss_per_batch': False,  # set True for PT loss (mean per component/batch), False (mean per component per sample)
    'reg_constant': 0.000,
    'max_grad_norm': 10.0,
    'num_epochs': 50,  # number of epochs over dataset for training
    'start_epoch': 1,  # epoch to start
    'reset_global_step': True,
    'train_era_step': 1,  # start train step during current era, value of 0 saves the current model

    'learn_rate': 0.0005,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'learn_rate_decay': 0.975,
    'staircase_decay': True,
    'decay_steps': 0.75,

    'ss_prob': 1.0,  # scheduled sampling probability for training. probability of passing decoder output as next

    'restore': False, # boolean. restore model from disk
    'restore_model': "/data/mat10/Projects/audio23d/Models/1dconv/conv1d_100words_ccloss_era1_epoch1_step1478",

    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/Projects/audio23d/Models/1dconv/conv1d_ccloss_era1",
    'num_models_saved': 100,  # total number of models saved
    'save_steps': None,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/Projects/audio23d/Models/1dconv/summaries",
    'save_summaries': False

          }

# if __name__ == "__main__":

model = DenseNet1D(options)

sess = start_interactive_session()

if options['save_graph']:
    model.save_graph(sess)

if options['restore']:
    model.restore_model(sess)

if options['is_training']:
    model.train(sess)
else:
    loss = model.eval(sess, return_words=False)



