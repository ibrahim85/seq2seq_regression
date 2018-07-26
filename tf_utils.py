import tensorflow as tf

def start_interactive_session():
    sess = tf.InteractiveSession()
    # with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # what are local vars?
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    return sess
