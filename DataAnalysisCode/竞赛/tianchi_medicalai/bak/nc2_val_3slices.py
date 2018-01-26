import math
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.dlmod import nc2_3slices as nc2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('val_dir', '/home/cangzhu/data/output/val_log', '')
tf.app.flags.DEFINE_string('ckpt_dir', '/home/cangzhu/data/output/train_log', '')
tf.app.flags.DEFINE_integer('val_interval_secs', 60 * 5, '')
tf.app.flags.DEFINE_integer('num_examples', nc2.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, '')
tf.app.flags.DEFINE_boolean('run_once', False, '')


def eval_once(saver, summary_writer, top_k_op, label_b, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
        print('Evaluation results are based on the model:',
              os.path.basename(ckpt.model_checkpoint_path))
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            lbs_positive_num, lbs_negative_num = 0, 0
            while step < num_iter and not coord.should_stop():
                predictions, labels = sess.run([top_k_op, label_b])
                true_count += np.sum(predictions)
                predictions = np.asarray(predictions).flatten()
                labels = np.asarray(labels).flatten()
                idx_true = np.where(predictions)
                idx_false = np.where(~predictions)
                lbs_positive_num += labels.sum()
                lbs_negative_num += (labels.size - lbs_positive_num)
                print(labels[idx_true])
                print(labels[idx_false])
                # idx_true = (predictions == True)

                # idx_true = (np.where(predictions, True))
                # idx_false = (predictions == False)
                # print(idx_true)
                # print(idx_false)
                print('\n\n')
                step += 1
            print(lbs_positive_num, lbs_negative_num)

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    with tf.Graph().as_default() as g:
        name_b, coord_diameter_b, frames_b, label_b = nc2.inputs('tfr_val2', slice=-1)
        logits = nc2.inference(frames_b)
        top_k_op = tf.nn.in_top_k(logits, label_b, 1)
        variable_averages = tf.train.ExponentialMovingAverage(nc2.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.val_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, label_b, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.val_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    tf.gfile.MakeDirs(FLAGS.val_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
