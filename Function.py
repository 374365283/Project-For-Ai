import tensorflow as tf

#definition of weights
def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
    return initial

#definition of bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial

#convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#max pooling
def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


#compute loss
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


#optimize loss
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

#compute accuracy
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

