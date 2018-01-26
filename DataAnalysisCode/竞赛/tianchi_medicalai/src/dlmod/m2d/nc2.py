import os
import re

import numpy as np
import tensorflow as tf

from src.dlmod.m2d import nc2_input

TOWER_NAME = 'tower'
IMG_SIZE = nc2_input.IMG_SIZE
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

PROP_COEF = [1, 1, 1]

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 16, '')
tf.app.flags.DEFINE_string('data_dir', nc2_input.DATA_DIR, '')
tf.app.flags.DEFINE_boolean('use_fp16', False, '')


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


def distorted_inputs(dataset, slice=-1):
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('Please supply a valid data_dir')
    data_tfr_dir = os.path.join(FLAGS.data_dir, dataset)
    if not os.path.isdir(data_tfr_dir):
        raise ValueError('Please supply a valid dataset')
    name_b, coord_diameter_b, frames_b, label_b = nc2_input.distorted_inputs(data_dir=data_tfr_dir,
                                                                             batch_size=FLAGS.batch_size,
                                                                             slice=slice)
    return name_b, coord_diameter_b, frames_b, label_b


def inputs(dataset, slice=-1):
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('Please supply a valid data_dir')
    data_tfr_dir = os.path.join(FLAGS.data_dir, dataset)
    if not os.path.isdir(data_tfr_dir):
        raise ValueError('Please supply a valid dataset')
    name_b, coord_diameter_b, frames_b, label_b = nc2_input.inputs(data_dir=data_tfr_dir,
                                                                   batch_size=FLAGS.batch_size,
                                                                   slice=slice)
    return name_b, coord_diameter_b, frames_b, label_b


def batch_norm_layer(x, scope_name_bn, train_phase):
    with tf.variable_scope(scope_name_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def inference_one_slice(frame_batch, name_, train_phase):
    # frame_shape = frame_batch.get_shape()
    with tf.variable_scope('%sconv1' % name_) as scope:
        dim = 1
        dim2 = 64
        kernel = _variable_with_weight_decay(name='weights',
                                             shape=[5, 5, dim, dim2],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(frame_batch, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        # conv1 = tf.nn.sigmoid(bias, name=scope.name)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='%spool1' % name_)
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    norm1 = batch_norm_layer(pool1, '%snorm1' % name_, train_phase)

    with tf.variable_scope('%sconv2' % name_) as scope:
        dim = dim2
        dim2 = 64
        kernel = _variable_with_weight_decay(name='weights',
                                             shape=[5, 5, dim, dim2],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        # conv2 = tf.nn.sigmoid(bias, name=scope.name)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    norm2 = batch_norm_layer(conv2, '%snorm2' % name_, train_phase)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='%spool2' % name_)

    with tf.variable_scope('%slocal3' % name_) as scope:
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        dim2 = 384
        weights = _variable_with_weight_decay('weights', shape=[dim, dim2], stddev=0.04, wd=0.004)

        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.1))

        # local3 = tf.nn.sigmoid(tf.matmul(reshape, weights) + biases, name=scope.name)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    with tf.variable_scope('%slocal4' % name_) as scope:
        dim = dim2
        dim2 = 192
        weights = _variable_with_weight_decay('weights', shape=[dim, dim2], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [dim2], tf.constant_initializer(0.1))

        # local4 = tf.nn.sigmoid(tf.matmul(local3, weights) + biases, name=scope.name)
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    with tf.variable_scope('%ssoftmax_linear' % name_) as scope:
        weights = _variable_with_weight_decay('weights', [dim2, NUM_CLASSES], stddev=1 / float(dim2), wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def inference(frames_batch, prop_coef, train_phase):
    total_softmax_linear = 0

    with tf.variable_scope('total_softmax_linear') as scope:
        weights = _variable_with_weight_decay('total_weights', [3, 1], stddev=1 / float(3), wd=0.0)
        for i, coef in enumerate(prop_coef):
            name_ = 'i%d_' % i
            if coef == 0:
                continue
            frame_b = tf.reshape(frames_batch[:, :, :, i], [FLAGS.batch_size, IMG_SIZE, IMG_SIZE, 1])
            sl = inference_one_slice(frame_b, name_, train_phase)
            # print(sl)
            total_softmax_linear += (sl * weights[i])
        _activation_summary(total_softmax_linear)
    return total_softmax_linear


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
