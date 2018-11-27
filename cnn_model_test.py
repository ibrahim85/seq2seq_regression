from cnn_models import CNNModel
from tf_utils import start_interactive_session, set_gpu
import numpy as np
import tensorflow as tf

set_gpu(3)

options = {
    'data_root_dir': "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtw_antonio",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_dtwN",
# "/home/michaeltrs/Projects/audio23d/data",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
    'split_name': "devel",  # 'devel',
    'is_training' : False,
    'data_in': 'melf',  # mfcc, melf, melf_2d
    'batch_size': 1,   # number of examples in queue either for training or inference
    'random_crop': False,
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|

    'has_encoder': True,
    '1dcnn_features_dims': [256, 256, 256],

    'loss_fun': "concordance_cc",
    'reg_constant': 0.000,
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
<<<<<<< HEAD
    'restore_model':"/data/mat10/Projects/audio23d/Models/1dcnn/1dconv_res_melf_cc_era1",
=======
    'restore_model':"/data/mat10/Projects/audio23d/Models/dtwN/1dconv_res/1dconv_res_mfcc_all_era1_epoch1_step3536",
>>>>>>> f798981d5c303deabd8107e5086cbc23a1985d2f

    'save': True,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/Projects/audio23d/Models/1dcnn/1dconv_res_melf_cc_era1",
    'num_models_saved': 100,  # total number of models saved
    'save_steps': None,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/Projects/audio23d/Models/1dcnn/summaries",
    'save_summaries': True

          }

if __name__ == "__main__":
    ep =5 
    options['restore_model'] = options['save_model'] + "_epoch%d_step6504" % ep
    model = CNNModel(options)
    sess = start_interactive_session()
    model.restore_model(sess)
    loss = model.eval(sess, return_words=False)
    print("loss at epoch %d is %.4f" % (ep, np.mean(loss)))


<<<<<<< HEAD
if False:
    model = CNNModel(options)
    losses = {}
    for ep in range(1, 54):
        options['restore_model'] = options["save_model"] + "_epoch%d_step6504" % ep
        #model = CNNModel(options)
        sess = start_interactive_session()
=======
    if True:
        losses = {}
    for ep in range(1, 4):
        options['restore_model'] = "/data/mat10/Projects/audio23d/Models/dtwN/1dconv_res/1dconv_res_melf_all_era1_epoch%d_step3536" % ep
        model = CNNModel(options)
        sess = start_interactive_session()
        #if options['restore']:
>>>>>>> f798981d5c303deabd8107e5086cbc23a1985d2f
        model.restore_model(sess)
        loss = model.eval(sess, num_steps=None, return_words=False)
        losses[ep] = np.mean(loss)
        tf.reset_default_graph()
