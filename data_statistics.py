import tensorflow as tf
from tf_utils import start_interactive_session, set_gpu
import numpy as np
from data_provider import get_split3

set_gpu(5)

def get_train_data_stats(options):
    encoder_inputs, target_labels, num_examples, words, decoder_inputs, \
    target_labels_lengths, encoder_inputs_lengths, decoder_inputs_lengths = get_split3(options)
    number_of_steps_per_epoch = 5#num_examples // options['batch_size'] + 1
    sess = start_interactive_session()
    eim = []
    eivar = []
    tlm = []
    tlvar = []
    eilm = []
    for i in range(number_of_steps_per_epoch):
        print("step %d of %d" % (i+1, number_of_steps_per_epoch))
        ei, tl, eil = sess.run([tf.nn.moments(encoder_inputs, [0, 1]),
                                tf.nn.moments(target_labels, [0, 1]),
                                tf.reduce_mean(encoder_inputs_lengths)])
        eim.append(ei[0])
        eivar.append(ei[1])
        tlm.append(tl[0])
        tlvar.append(tl[1])
        eilm.append(eil)

    eim = np.stack(eim, axis=0)#.mean(axis=0)
    eivar = np.stack(eivar, axis=0)#.mean(axis=0)
    tlm = np.stack(tlm, axis=0)#.mean(axis=0)
    tlvar = np.stack(tlvar, axis=0)#.mean(axis=0)
    eilm = np.max(eilm)
    return eim, eivar, tlm, tlvar, eilm

options = {
    'data_root_dir': "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",  # enhanced",
    'split_name': "train",
    'use_rmse': False,
    'batch_size': 10000,   # number of examples in queue either for training or inference
    'reverse_time': False,
    'shuffle': False,
    'random_crop': False,
    'mfcc_num_features': 20,  # 20,
    'raw_audio_num_features': 533,  # 256,
    'num_classes': 28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    'standardize_inputs and labels': True,
    'mfcc_gaussian_noise_std': 0.0,  # 0.05,
    'label_gaussian_noise_std': 0.0}

if __name__ == "__main__":
    eim, eivar, tlm, tlvar, eilm = get_train_data_stats(options)
