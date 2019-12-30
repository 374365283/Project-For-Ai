#definition of CNN

import tensorflow as tf
from Function import weight_variable, bias_variable, conv2d, max_pool_2x2


def net1(images, batch_size, n_classes):
    #1st convolution
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 32], 1.0), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)   # 32个偏置值
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1)+b_conv1, name='conv1')  # 得到128*128*32(假设原始图像是128*128)

    #1st max pooling
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pool_2x2(h_conv1, 'pooling1')   # 得到64*64*32
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    #1st full connected layer
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm1, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 128], 0.005),  name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)

    #2nd full connected layer
    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([128 ,128], 0.005),name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)

    # dropout
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.5)


    #softmax regression  
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')

    return softmax_linear



def net2(images, batch_size, n_classes):
    #1st convolution
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 64], 1.0), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)   # 64个偏置值
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1)+b_conv1, name='conv1')  # 得到128*128*64(假设原始图像是128*128)

    #1st max pooling
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pool_2x2(h_conv1, 'pooling1')   # 得到64*64*64
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    #2nd convolution
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 64, 32], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)   # 32个偏置值
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2)+b_conv2, name='conv2')  # 得到64*64*32

    #2nd max pooling
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pool_2x2(h_conv2, 'pooling2')  # 得到32*32*32
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    #1st full connected
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 128], 0.005),  name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)


    #2nd full connected
    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([128 ,128], 0.005),name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)


    #dropout
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.5)

    #softmax regression
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')

    return softmax_linear



def net3(images, batch_size, n_classes):
    #1st convolution
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 64], 1.0), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)   # 64个偏置值
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1)+b_conv1, name='conv1')  # 得到128*128*64(假设原始图像是128*128)

    #1st max pooling
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pool_2x2(h_conv1, 'pooling1')   # 得到64*64*64
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

	
    #2nd convolution
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 64, 32], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)   # 32个偏置值
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2)+b_conv2, name='conv2')  # 得到64*64*32

    #2nd max pooling
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pool_2x2(h_conv2, 'pooling2')  # 得到32*32*32
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    #3rd convolution
    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3, 3, 32, 16], 0.1), name='weights', dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([16]), name='biases', dtype=tf.float32)   # 16个偏置值
        h_conv3 = tf.nn.relu(conv2d(norm2, w_conv3)+b_conv3, name='conv3')  # 得到32*32*16

    #3rd max pooling
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = max_pool_2x2(h_conv3, 'pooling3')  # 得到16*16*16
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    #1st full connected
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 128], 0.005),  name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)

    #2nd full connected
    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([128 ,128], 0.005),name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)


    #dropout
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.5)


    # Softmax regression
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')

    return softmax_linear


