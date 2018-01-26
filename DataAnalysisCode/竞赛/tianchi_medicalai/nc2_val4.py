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
tf.app.flags.DEFINE_boolean('run_once', True, '')


def eval_once(saver, summary_writer, softmax, label_b, summary_op, min_probability=0.5):
    x = (min_probability - 0.5) * 2
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
            false_count = 0
            true_positive_count = 0
            true_negative_count = 0
            false_positive_count = 0
            false_negative_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            # lbs_positive_num, lbs_negative_num = 0, 0
            while step < num_iter and not coord.should_stop():
                softmax_result, labels = sess.run([softmax, label_b])
                softmax_result[:, 1] = softmax_result[:, 1] - x
                predictions = np.argmax(softmax_result, 1)
                predictions = np.asarray(predictions).flatten()
                labels = np.asarray(labels).flatten()
                mask_true = (predictions == labels)
                mask_false = ~mask_true
                t_c = mask_true.sum()
                f_c = mask_false.sum()
                tp_c = predictions[mask_true].sum()
                tn_c = t_c - tp_c
                fp_c = predictions[mask_false].sum()
                fn_c = f_c - fp_c
                # print(tp_c, tn_c, fp_c, fn_c)

                true_count += t_c
                false_count += f_c
                true_positive_count += tp_c
                true_negative_count += tn_c
                false_positive_count += fp_c
                false_negative_count += fn_c
                step += 1
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: min_probability @ %.3f, precision @ 1 = %.3f' % (datetime.now(), min_probability, precision))
            print('RP:%d, RN:%d' % (
                true_positive_count + false_negative_count, false_positive_count + true_negative_count,))
            print('TP:%d, FP:%d, TN:%d, FN:%d' % (
                true_positive_count, false_positive_count, true_negative_count, false_negative_count))
            tpr = true_positive_count / (true_positive_count + false_positive_count)  # 命中率
            fpr = false_positive_count / (false_positive_count + true_negative_count)  # 假报警率
            rcr = true_positive_count / (true_positive_count + false_negative_count)
            print('命中率TPR:%.3f, 假报警率FPR:%.3f, 召回率RecallRate:%.3f' % (tpr, fpr, rcr))
            # print('%s')

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
        logits = nc2.inference(frames_b, nc2.PROP_COEF, tf.constant(False, tf.bool))
        softmax = tf.nn.softmax(logits)
        # top_k = tf.nn.top_k(possibility)
        variable_averages = tf.train.ExponentialMovingAverage(nc2.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.val_dir, g)

        while True:
            eval_once(saver, summary_writer, softmax, label_b, summary_op, min_probability=0.8)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.val_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    tf.gfile.MakeDirs(FLAGS.val_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
