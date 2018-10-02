from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from transformer_model import SelfAttentionEncoder
from tf_utils import start_interactive_session, set_gpu

set_gpu(5)

options = defaultdict(
    lambda: None,  # Set default value to None.
    # Input params
    #default_batch_size=2,  # Maximum number of tokens per batch of examples.
    max_length=50,  # Maximum number of tokens per example.
    # Model params
    initializer_gain=1.0,  # Used in trainable variable initialization.
    vocab_size=28,  # Number of tokens defined in the vocabulary file.
    hidden_size=128,  # Model dimension in the hidden layers.
    num_hidden_layers=3,  # Number of layers in the encoder and decoder stacks.
    num_heads=4,  # Number of heads to use in multi-headed attention.
    filter_size=128,  # Inner layer dimension in the feedforward network.
    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    ##################################################

    data_root_dir="/home/mat10/Documents/Projects/audio23d/test_models/example_data",  # "/vol/atlas/homes/pt511/db/audio_to_3d/tf_records_clean",  # enhanced",

    is_training=True,
    split_name='example',
    data_split="split3",
    use_rmse=False,
    batch_size=512,  # number of examples in queue either for training or inference
    reverse_time=False,
    shuffle=True,
    random_crop=False,
    standardize_inputs_and_labels=True,
    mfcc_num_features=20,  # 20,
    raw_audio_num_features=533,  # 256,
    num_classes=28,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    max_out_len_multiplier=1.0,  # max_out_len = max_out_len_multiplier * max_in_len

    mfcc_gaussian_noise_std=0.0,  # 0.05,
    label_gaussian_noise_std=0.0,

    loss_fun="concordance_cc",  # "mse", "cos", "concordance_cc"
    ccc_loss_per_batch=False,  # set True for PT loss (mean per component/batch), False (mean per component per sample)
    reg_constant=0.00,
    max_grad_norm=5.0,
    num_epochs=5,  # number of epochs over dataset for training
    start_epoch=1,  # epoch to start
    reset_global_step=False,
    train_era_step=1,  # start train step during current era, value of 0 saves the current model

    learn_rate=0.001,  # initial learn rate corresponing top global step 0, or max lr for Adam
    learn_rate_decay=0.95,
    staircase_decay=True,
    decay_steps=0.5,

    restore=False,  # boolean. restore model from disk
    restore_model="/data/mat10/Projects/audio23d/Models/transformer/seq2seq_exccc_transformer_era1_final",

    save=False,  # boolean. save model to disk during current era
    save_model="/data/mat10/Projects/audio23d/Models/transformer/seq2seq_exccc_transformer_era2",
    num_models_saved=100,  # total number of models saved
    save_steps=None,  # every how many steps to save model

    save_graph=False,
    save_dir="/data/mat10/Projects/audio23d/Models/transformer/summaries",
    save_summaries=False
)


# inputs0 = tf.convert_to_tensor(np.random.randint(1, 10, (2, 300)), dtype=tf.int32)
# inputs = tf.convert_to_tensor(np.random.rand(2, 300, 128), dtype=tf.float32)
# targets = tf.convert_to_tensor(np.random.rand(2, 10, 128), dtype=tf.float32)

model = SelfAttentionEncoder(options)

# out = model.encode()

sess = start_interactive_session()

# o = sess.run(model.decoder_outputs)
# o.shape
# o[0,0,0]


if options['save_graph']:
    model.save_graph(sess)

if options['restore']:
    model.restore_model(sess)

if options['is_training']:
    model.train(sess)
else:
    loss = model.eval(sess, return_words=True)
