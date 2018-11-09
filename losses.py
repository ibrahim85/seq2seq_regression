import tensorflow as tf


def masked_loss(values_in, loss_fun):
    prediction, ground_truth, mask = values_in
    # apply mask to predictions and ground truth
    prediction = tf.boolean_mask(prediction, mask)
    ground_truth = tf.boolean_mask(ground_truth, mask)
    return loss_fun(prediction, ground_truth)


# def batch_masked_loss(values_in, loss_fun, options, return_mean=True):
#     predictions, ground_truths, mask = values_in
#     max_label_len = tf.shape(ground_truths)[1]
#     label_dim = tf.shape(ground_truths)[-1]
#     # mask = tf.expand_dims(mask, -1)
#     # multiply = tf.constant([1, 1, options['num_classes']])
#     # mask = tf.tile(mask, multiply)
#     #if options['ccc_loss_per_batch']:
#     #    train_losses = [loss_fun(
#     #        (tf.reshape(predictions[:, :, i], (-1,)),
#     #         tf.reshape(ground_truths[:, :, i], (-1,)),
#     #         tf.reshape(mask[:, :, i], (-1,))))
#     #        for i in range(options['num_classes'])]
#     #else:  # CCC loss per example per component over time
#         #train_losses = [[loss_fun(
#         #    (tf.reshape(predictions[n, :, i], (-1,)),
#         #     tf.reshape(ground_truths[n, :, i], (-1,)),
#         #     tf.reshape(mask[n, :, i], (-1,))))
#         #    for i in range(options['num_classes'])]
#         #    for n in range(options['batch_size'])]
#     if options['loss_fun'] == "concordance_cc":
#         predictions = tf.transpose(predictions, (0, 2, 1))
#         ground_truths = tf.transpose(ground_truths, (0, 2, 1))
#         mask = tf.transpose(mask, (0, 2, 1))
#         train_losses = tf.map_fn(
#             fn=loss_fun,
#             elems=(tf.reshape(predictions, (-1, max_label_len)),
#                    tf.reshape(ground_truths, (-1, max_label_len)),
#                    tf.reshape(mask, (-1, max_label_len))),
#             dtype=tf.float32,
#             parallel_iterations=10)
#     elif options['loss_fun'] == "mse":
#         train_losses = tf.map_fn(
#             fn=loss_fun,
#             elems=(tf.reshape(predictions, (-1, label_dim)),
#                    tf.reshape(ground_truths, (-1, label_dim)),
#                    tf.reshape(mask, (-1, label_dim))),
#             dtype=tf.float32,
#             parallel_iterations=10)
#         train_losses = tf.boolean_mask(train_losses, tf.logical_not(tf.is_nan(train_losses)))
#     if return_mean:
#         return tf.reduce_mean(train_losses)
#     return train_losses


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


loss_weights = tf.convert_to_tensor(
    value=[
 0.48839835553540395,
 0.19681425259490665,
 0.12217412465555928,
 0.08584908914809869,
 0.026193009075865175,
 0.020437278568632953,
 0.01946587889940399,
 0.01317249981046566,
 0.009119010486653758,
 0.0045903790035307646,
 0.002720364420207774,
 0.0018374918370577466,
 0.0013809016118477538,
 0.0012247983722109938,
 0.00114557976840072,
 0.0008334686724394426,
 0.0007382192652328583,
 0.0005678340764310592,
 0.0005146449058031808,
 0.00047596709281886294,
 0.00044244558476774163,
 0.00035284317397555546,
 0.0003023795354451357,
 0.0002857594137422436,
 0.00026577835171407257,
 0.00025999602872380347,
 0.00022583396687155238,
 0.00021181614378860966],
    dtype=tf.float32)
# loss_weights = tf.convert_to_tensor(
#     value=[0.03571428571428571]*28,
#     dtype=tf.float32)


def batch_masked_concordance_cc(values_in, options, return_mean=True):
    def masked_concordance_cc(values_in):
        return masked_loss(values_in, concordance_cc)
    predictions, ground_truths, mask = values_in
    max_label_len = tf.shape(ground_truths)[1]
    label_dim = options['num_classes'] 
    predictions = tf.transpose(predictions, (2, 0, 1))  # (0, 2, 1))
    ground_truths = tf.transpose(ground_truths, (2, 0, 1))  # (0, 2, 1))
    mask = tf.transpose(mask, (2, 0, 1))  # (0, 2, 1))
    train_losses = tf.map_fn(
       fn=masked_concordance_cc,
       elems=(tf.reshape(predictions, (label_dim, -1)),  # (-1, max_label_len)),
              tf.reshape(ground_truths, (label_dim, -1)),  # (-1, max_label_len)),
              tf.reshape(mask, (label_dim, -1))),  # (-1, max_label_len))),
       dtype=tf.float32,
       parallel_iterations=label_dim)
    print("train losses:", train_losses)
    # train_losses = [masked_concordance_cc(
    #     (tf.reshape(predictions[:, :, i], (-1,)),
    #      tf.reshape(ground_truths[:, :, i], (-1,)),
    #      tf.reshape(mask[:, :, i], (-1,))))
    #     for i in range(options['num_classes'])]
    if return_mean:
        return tf.reduce_sum(tf.multiply(loss_weights, train_losses))
    # tf.losses.compute_weighted_loss(losses=train_losses, weights=loss_weights)  #tf.reduce_mean(train_losses)  #
    return train_losses


def mse(prediction, ground_truth):
    return tf.reduce_mean(
        tf.pow(prediction - ground_truth, 2))


def batch_masked_mse(values_in, options, return_mean=True):
    def masked_mse(values_in):
        return masked_loss(values_in, mse)
    predictions, ground_truths, mask = values_in
    batch_size =options['batch_size']
    # # option 1 - take average per label vector
    # train_losses = tf.map_fn(
    #         fn=masked_mse,
    #         elems=(tf.reshape(predictions, (-1, label_dim)),
    #                tf.reshape(ground_truths, (-1, label_dim)),
    #                tf.reshape(mask, (-1, label_dim))),
    #         dtype=tf.float32,
    #         parallel_iterations=10)
    # option 2 - take average per sample
    train_losses = tf.map_fn(
            fn=masked_mse,
            elems=(tf.reshape(predictions, (batch_size, -1)),
                   tf.reshape(ground_truths, (batch_size, -1)),
                   tf.reshape(mask, (batch_size, -1))),
            dtype=tf.float32,
            parallel_iterations=10)  # batch_size)
    train_losses = tf.boolean_mask(train_losses, tf.logical_not(tf.is_nan(train_losses)))
    if return_mean:
        return tf.reduce_mean(tf.multiply(loss_weights, train_losses))
    return train_losses  # batch_masked_loss(values_in, masked_mse, options, return_mean)


def L2loss(reg_constant):
    return reg_constant * \
           tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
