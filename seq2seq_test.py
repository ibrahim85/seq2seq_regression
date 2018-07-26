import tensorflow as tf
# from data_provider2 import get_split
from tf_utils import start_interactive_session
from models import Model1

options = {

    'is_training' : True,
    'split_name': 'devel',
    'batch_size': 256,   # number of examples in queue either for training or inference

    'num_classes': 7,  # number of output classes 29 = |a-z, " ", <sos>, <eos>|
    'max_out_len_multiplier': 0.50,  # max_out_len = max_out_len_multiplier * max_in_len

    'encoder_num_layers': 3,  # number of hidden layers in encoder lstm
    'encoder_num_residual_layers': 3,  # number of residual layers, starting from first layer
    'encoder_num_hidden': 128,  # number of hidden units in encoder lstm
    'encoder_dropout_keep_prob' : None,  # probability of keeping neuron, deprecated
    'encoder_layer_norm': True,

    'decoder_num_layers': 3,  # number of hidden layers in decoder lstm
    'decoder_num_residual_layers': 3,  # number of residual layers, starting from first layer
    'decoder_num_hidden': 128,  # number of hidden units in decoder lstm
    'encoder_state_as_decoder_init' : False,  # bool. encoder state is used for decoder init state, else zero state
    'decoder_layer_norm': True,

    'attention_layer_size': 128,  # number of hidden units in attention layer
    'attention_layer_norm': True,
    'num_hidden_out': 64,  # number of hidden units in output fcn

    # 'beam_width': 20,  # number of best solutions used in beam decoder
    'max_in_len': None,  # maximum number of frames in input videos
    'max_out_len': None,  # maximum number of characters in output text

    'loss_fun': "mse",  # "mse" or "cos"
    'reg_constant': 0.001,
    'max_grad_norm': 1.0, 
    'num_epochs': 1,  # number of epochs over dataset for training
    'start_epoch': 1,  # epoch to start
    'reset_global_step': True,
    'train_era_step': 1,  # start train step during current era, value of 0 saves the current model
    
    'learn_rate': 0.00,  # initial learn rate corresponing top global step 0, or max lr for Adam
    'learn_rate_decay': 0.,
    'staircase_decay': True,
    'decay_steps': 200,

    'ss_prob': 1.0,  # scheduled sampling probability for training. probability of passing decoder output as next

    'restore': True, # boolean. restore model from disk
    'restore_model': "/data/mat10/MSc_Project/audio_to_3dvideo/Models/model2/seq2seq_m3_train_era1_final",  # path to model to restore

    'save': False,  # boolean. save model to disk during current era
    'save_model': "/data/mat10/MSc_Project/audio_to_3dvideo/Models/model2/seq2seq_m3_train_era2",
    'num_models_saved': 50,  # total number of models saved
    'save_steps': 200  # every how many steps to save model

          }


model = Model1(options)

sess = start_interactive_session()

model.train(sess)



