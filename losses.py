import tensorflow as tf


def masked_loss(values_in, loss_fun):
    prediction, ground_truth, mask = values_in
    # apply mask to predictions and ground truth
    prediction = tf.boolean_mask(prediction, mask)
    ground_truth = tf.boolean_mask(ground_truth, mask)
    return loss_fun(prediction, ground_truth)


def batch_masked_loss(values_in, loss_fun, options, return_mean=True):
    predictions, ground_truths, mask = values_in
    # mask = tf.expand_dims(mask, -1)
    # multiply = tf.constant([1, 1, options['num_classes']])
    # mask = tf.tile(mask, multiply)
    train_losses = [loss_fun(
        (tf.reshape(predictions[:, :, i], (-1,)),
         tf.reshape(ground_truths[:, :, i], (-1,)),
         tf.reshape(mask[:, :, i], (-1,))))
        for i in range(options['num_classes'])]
    if return_mean:
        return tf.reduce_mean(train_losses)
    return train_losses


def concordance_cc(prediction, ground_truth):
    """Defines concordance loss for training the model.
    Args:
       prediction: prediction of the model.
       ground_truth: ground truth values.
       mask: True for non padded elements, False for padded elements
    Returns:
       The concordance value.
    """
    # compute CCC with tensors
    pred_mean, pred_var = tf.nn.moments(prediction, (0,))
    gt_mean, gt_var = tf.nn.moments(ground_truth, (0,))
    mean_cent_prod = tf.reduce_mean((prediction - pred_mean) * (ground_truth - gt_mean))
    return - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))


def masked_concordance_cc(values_in):
    return masked_loss(values_in, concordance_cc)


def batch_masked_concordance_cc(values_in, options, return_mean=True):
    return batch_masked_loss(values_in, masked_concordance_cc, options, return_mean)


def mse(prediction, ground_truth):
    return tf.reduce_mean(
        tf.pow(prediction - ground_truth, 2))


def masked_mse(values_in):
    return masked_loss(values_in, mse)


def batch_masked_mse(values_in, options, return_mean=True):
    return batch_masked_loss(values_in, masked_mse, options, return_mean)


def L2loss(reg_constant):
    return reg_constant * \
           tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

