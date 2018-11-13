from  mixed_seq_models import CNNRNNModel_dense_raw
from tf_utils import start_interactive_session, set_gpu
import numpy as np

set_gpu(1)

options = {
    'data_root_dir': "/home/michaeltrs/Projects/audio23d/data/antonio",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_antonio",
# "/home/michaeltrs/Projects/audio23d/data",
# "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_lrs",
# ,  # "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",  # enhanced",

    'is_training' : True,
    'split_name': "example",  # ""train",  # 'devel',
    'data_in': 'raw',  # mfcc, melf, melf_2d
    'batch_size': 20,   # number of examples in queue either for training or inference
    'random_crop': False,
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|

    'growth_rate': 20,
    'num_layers' : 6,
    'final_layer_dim': 64,
    'batch_norm': True,

    'loss_fun': "concordance_cc",
    'reg_constant': 0.00,
    'max_grad_norm': 100.0,
    'num_epochs': 30,  # number of epochs over dataset for training
    'start_epoch': 1,  # epoch to start
    'reset_global_step': True,
    'train_era_step': 1,  # start train step during current era, value of 0 saves the current model

    'learn_rate': 0.001,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'learn_rate_decay': 0.975,
    'staircase_decay': True,
    'decay_steps': 5.0,

    'ss_prob': 1.0,  # scheduled sampling probability for training. probability of passing decoder output as next

    'restore': False,  # boolean. restore model from disk
    'restore_model': "/data/mat10/Projects/audio23d/Models/seq_cnn2dres_lstm/seq2seq_cnn2dres_lstm_all_melf_cc_era1_epoch1_step8325",
    # "/data/mat10/Projects/audio23d/Models/seq2seq_cnn_lstm/seq2seq_cnn_lstm_seq10_era1_epoch10_step604",

    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/Projects/audio23d/Models/seq_cnn_raw2/seq_cnn_raw2_batchnorm_all_melf_cc_era",
    'num_models_saved': 100,  # total number of models saved
    'save_steps': None,  # every how many steps to save model

    'save_graph': False,
    'save_dir': "/data/mat10/Projects/audio23d/Models/seq_cnn_raw2/summaries",
    'save_summaries': False


          }

if __name__ == "__main__":

    model = CNNRNNModel_dense_raw(options)

    sess = start_interactive_session()

    if options['save_graph']:
       model.save_graph(sess)

    if options['restore']:
        model.restore_model(sess)

    if options['is_training']:
        model.train(sess)
    else:
        loss = model.eval(sess, num_steps=None, return_words=False)

