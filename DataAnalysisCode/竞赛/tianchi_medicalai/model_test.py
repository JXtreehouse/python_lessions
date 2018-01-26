import tensorflow as tf
import numpy as np

sess = tf.Session()
init = tf.global_variables_initializer()
img = tf.Variable(np.zeros([3, 32, 32, 32, 1]), dtype=tf.float32)
print(img)
pool0 = tf.nn.avg_pool3d(img, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 1, 1, 1], padding='SAME', name='pool0')
print(pool0)

with tf.variable_scope('conv1') as scope:
    dim1, dim2 = 1, 64
    kernel = tf.get_variable('weights', [3, 3, 3, dim1, dim2],
                             initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                             dtype=tf.float32)
    conv = tf.nn.conv3d(pool0, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [dim2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
print(conv1)
pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='pool1')
print(pool1)

with tf.variable_scope('conv2') as scope:
    dim1, dim2 = dim2, 128
    kernel = tf.get_variable('weights', [3, 3, 3, dim1, dim2],
                             initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                             dtype=tf.float32)
    conv = tf.nn.conv3d(pool1, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [dim2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
print(conv2)
pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2')
print(pool2)

with tf.variable_scope('conv3') as scope:
    dim1, dim2 = dim2, 256
    kernel1 = tf.get_variable('weights1', [3, 3, 3, dim1, dim2],
                              initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                              dtype=tf.float32)
    conv = tf.nn.conv3d(pool2, kernel1, strides=[1, 1, 1, 1, 1], padding='SAME')
    kernel2 = tf.get_variable('weights2', [3, 3, 3, dim2, dim2],
                              initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                              dtype=tf.float32)
    conv = tf.nn.conv3d(conv, kernel2, strides=[1, 1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [dim2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
print(conv3)
pool3 = tf.nn.max_pool3d(conv3, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool3')
print(pool3)

with tf.variable_scope('conv4') as scope:
    dim1, dim2 = dim2, 512
    kernel1 = tf.get_variable('weights1', [3, 3, 3, dim1, dim2],
                              initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                              dtype=tf.float32)
    conv = tf.nn.conv3d(pool3, kernel1, strides=[1, 1, 1, 1, 1], padding='SAME')
    kernel2 = tf.get_variable('weights2', [3, 3, 3, dim2, dim2],
                              initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                              dtype=tf.float32)
    conv = tf.nn.conv3d(conv, kernel2, strides=[1, 1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [dim2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
print(conv4)
pool4 = tf.nn.max_pool3d(conv4, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool4')
print(pool4)

with tf.variable_scope('conv5') as scope:
    dim1, dim2 = dim2, 64
    kernel = tf.get_variable('weights', [2, 2, 2, dim1, dim2],
                             initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                             dtype=tf.float32)
    conv = tf.nn.conv3d(pool4, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
    biases = tf.get_variable('biases', [dim2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
print(conv5)

with tf.variable_scope('conv6') as scope:
    dim1, dim2 = dim2, 1
    kernel = tf.get_variable('weights', [1, 1, 1, dim1, dim2],
                             initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                             dtype=tf.float32)
    conv = tf.nn.conv3d(conv5, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
    biases = tf.get_variable('biases', [dim2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    bias = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.sigmoid(bias, name=scope.name)
print(conv6)

with tf.variable_scope('conv7') as scope:
    dim1, dim2 = dim2, 1
    kernel = tf.get_variable('weights', [1, 1, 1, dim1, dim2],
                             initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                             dtype=tf.float32)
    conv7 = tf.nn.conv3d(conv6, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
print(conv7)

    # biases = tf.get_variable('biases', [dim2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    # bias = tf.nn.bias_add(conv, biases)
    # conv7 = tf.nn.sigmoid(bias, name=scope.name)


# # Lab 6 Softmax Classifier
# import tensorflow as tf
# import numpy as np
# tf.set_random_seed(777)  # for reproducibility
#
# # Predicting animal type based on various features
# xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]
#
# print(x_data.shape, y_data.shape)
#
# nb_classes = 7  # 0 ~ 6
#
# X = tf.placeholder(tf.float32, [None, 16])
# Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
# Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
# print("one_hot", Y_one_hot)
# Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# print("reshape", Y_one_hot)
#
# W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
# b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
#
# # tf.nn.softmax computes softmax activations
# # softmax = exp(logits) / reduce_sum(exp(logits), dim)
# logits = tf.matmul(X, W) + b
# print(logits)
