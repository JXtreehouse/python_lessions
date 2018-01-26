import os
import re
import tensorflow as tf
import numpy as np

from src.dlmod.m3d import nc2_input

TOWER_NAME = 'tower'
IMG_SHAPE = nc2_input.IMG_SHAPE
NUM_CLASSES = nc2_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = nc2_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = nc2_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = nc2_input.NUM_EXAMPLES_PER_EPOCH_FOR_TEST

MOVING_AVERAGE_DECAY = 0.9999
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001

NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTORY = 0.1
INITIAL_LEARNING_RATE = 0.1

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 16, '')
tf.app.flags.DEFINE_string('data_dir', nc2_input.DATA_DIR, '')
tf.app.flags.DEFINE_boolean('use_fp16', False, '')


class Config(object):
    def __init__(self):
        self.train_phase = False
        self.normalization = False
        self.biase = False


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs(dataset):
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('Please supply a valid data_dir')
    data_tfr_dir = os.path.join(FLAGS.data_dir, dataset)
    if not os.path.isdir(data_tfr_dir):
        raise ValueError('Please supply a valid dataset')
    name_b, coord_diameter_b, frames_b, label_b = nc2_input.distorted_inputs(data_dir=data_tfr_dir,
                                                                             batch_size=FLAGS.batch_size)
    return name_b, coord_diameter_b, frames_b, label_b


def inputs(dataset):
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('Please supply a valid data_dir')
    data_tfr_dir = os.path.join(FLAGS.data_dir, dataset)
    if not os.path.isdir(data_tfr_dir):
        raise ValueError('Please supply a valid dataset')
    name_b, coord_diameter_b, frames_b, label_b = nc2_input.inputs(data_dir=data_tfr_dir,
                                                                   batch_size=FLAGS.batch_size)
    return name_b, coord_diameter_b, frames_b, label_b


def batch_norm_layer(x, scope_name_bn, train_phase):
    with tf.variable_scope(scope_name_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=BN_DECAY)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def inference(frames_batch, train_phase):
    pool0 = tf.nn.avg_pool3d(frames_batch, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 1, 1, 1], padding='SAME', name='pool0')
    with tf.variable_scope('conv1') as scope:
        dim1, dim2 = 1, 64
        kernel = tf.get_variable('weights', [3, 3, 3, dim1, dim2],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                 dtype=tf.float32)
        conv = tf.nn.conv3d(pool0, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [dim2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name='pool1')
    # print(pool1)
    norm1 = batch_norm_layer(pool1, 'norm1', train_phase)

    with tf.variable_scope('conv2') as scope:
        dim1, dim2 = dim2, 128
        kernel = _variable_with_weight_decay(name='weights',
                                             shape=[3, 3, 3, dim1, dim2],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv3d(norm1, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
    pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2')
    norm2 = batch_norm_layer(pool2, 'norm2', train_phase)

    with tf.variable_scope('conv3') as scope:
        dim1, dim2 = dim2, 256
        kernel1 = _variable_with_weight_decay(name='weights1',
                                              shape=[3, 3, 3, dim1, dim2],
                                              stddev=5e-2,
                                              wd=0.0)
        conv = tf.nn.conv3d(norm2, kernel1, strides=[1, 1, 1, 1, 1], padding='SAME')
        kernel2 = _variable_with_weight_decay(name='weights2',
                                              shape=[3, 3, 3, dim2, dim2],
                                              stddev=5e-2,
                                              wd=0.0)
        conv = tf.nn.conv3d(conv, kernel2, strides=[1, 1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)
    pool3 = tf.nn.max_pool3d(conv3, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool3')
    norm3 = batch_norm_layer(pool3, 'norm3', train_phase)

    with tf.variable_scope('conv4') as scope:
        dim1, dim2 = dim2, 512
        kernel1 = _variable_with_weight_decay(name='weights1',
                                              shape=[3, 3, 3, dim1, dim2],
                                              stddev=5e-2,
                                              wd=0.0)
        conv = tf.nn.conv3d(norm3, kernel1, strides=[1, 1, 1, 1, 1], padding='SAME')
        kernel2 = _variable_with_weight_decay(name='weights2',
                                              shape=[3, 3, 3, dim2, dim2],
                                              stddev=5e-2,
                                              wd=0.0)
        conv = tf.nn.conv3d(conv, kernel2, strides=[1, 1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)
    pool4 = tf.nn.max_pool3d(conv4, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool4')
    norm4 = batch_norm_layer(pool4, 'norm4', train_phase)
    # print(norm4)

    with tf.variable_scope('conv5') as scope:
        dim1, dim2 = dim2, 64
        kernel = _variable_with_weight_decay(name='weights',
                                             shape=[2, 2, 2, dim1, dim2],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv3d(norm4, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv5)
    norm5 = batch_norm_layer(conv5, 'norm5', train_phase)

    with tf.variable_scope('out_class') as scope:
        dim1, dim2 = dim2, 1
        kernel = _variable_with_weight_decay(name='weights',
                                             shape=[1, 1, 1, dim1, dim2],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv3d(norm5, kernel, strides=[1, 1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv6)
    norm6 = batch_norm_layer(conv6, 'norm_out_class', train_phase)

    with tf.variable_scope('out_malignancy') as scope:
        # dim1, dim2 = dim2, 1
        kernel = _variable_with_weight_decay(name='weights',
                                             shape=[1, 1, 1, dim1, dim2],
                                             stddev=5e-2,
                                             wd=0.0)
        conv7 = tf.nn.conv3d(norm5, kernel, strides=[1, 1, 1, 1, 1], padding='VALID', name=scope.name)
        _activation_summary(conv7)
    norm7 = batch_norm_layer(conv7, 'norm_out_class', train_phase)

    softmax_linear = tf.reshape(tf.concat([norm6, norm7], 1), [FLAGS.batch_size, 2])

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + 'raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTORY,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
