import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from bak.dl import nc2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/cangzhu/data/output/train_log', '')
tf.app.flags.DEFINE_integer('max_steps', 10000, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, '')


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        name_b, coord_diameter_b, frames_b, label_b = nc2.inputs('trf_train')
        with tf.device('/cpu:0'):
            is_training = tf.placeholder('bool', [], name='is_training')
        print(is_training)
        #
        # images, labels = tf.cond(is_training,
        #                          lambda: (frames_b, label_b),
        #                          lambda: (frames_b, frames_b))
        # print(images, labels)

        logits = nc2.inference(frames_b, is_training)

        loss = nc2.loss(logits, label_b)

        train_op = nc2.train(loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summart_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            # print(sess.run(frames_b).reshape(4, -1).mean(1), sess.run(label_b))
            # print(step)
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            print('haha')
            if step % 1 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = '%s： step %d, loss = %.2f (%.1f excamples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summart_writer.add_summary(summary_str, step)

            if step % 1000 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir,
                                               '%s-%d-model.ckpt' % (datetime.now().date(), datetime.now().hour))
                saver.save(sess, checkpoint_path, global_step=step)
                meta_graph_def = tf.train.export_meta_graph(filename=checkpoint_path + '.meta2')


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
