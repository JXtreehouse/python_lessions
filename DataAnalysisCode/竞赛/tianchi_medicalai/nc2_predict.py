import csv
import math
import os

import numpy as np
import tensorflow as tf

from src.dlmod.m2d import nc2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('predict_dir', '/media/cangzhu/data/output_2d/test_log', '')
tf.app.flags.DEFINE_string('ckpt_dir', '/media/cangzhu/data/output_2d/train_log', '')
tf.app.flags.DEFINE_integer('predict_interval_secs', 60 * 5, '')
tf.app.flags.DEFINE_integer('num_examples', -1, '')
tf.app.flags.DEFINE_boolean('run_once', True, '')


def prediction(min_probability=0.5):
    x = (min_probability - 0.5) * 2
    if FLAGS.num_examples == -1:
        FLAGS.num_examples = nc2.NUM_EXAMPLES_PER_EPOCH_FOR_TEST
    print(FLAGS.num_examples)

    o_head = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
    csv_file = open(os.path.join(FLAGS.predict_dir, 'result_mask.csv'), 'w')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(o_head)
    with tf.Graph().as_default() as g:
        name_b, coord_diameter_b, frames_b, label_b = nc2.inputs('tfr_test2', slice=-1)
        logits = nc2.inference(frames_b, nc2.PROP_COEF, tf.constant(False, tf.bool))
        softmax = tf.nn.softmax(logits)
        # predict = tf.nn.top_k(possibility)

        variable_averages = tf.train.ExponentialMovingAverage(nc2.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
            print('Prediction results are based on the model:',
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
                cnt = 0
                while step < num_iter and not coord.should_stop():
                    # v, idk = sess.run([predict.values, predict.indices])
                    # values, indices, n_b, c_d_b = sess.run([predict.values, predict.indices, name_b, coord_diameter_b])
                    softmax_result, n_b, c_d_b = sess.run([softmax, name_b, coord_diameter_b])
                    values = softmax_result[:, 1].copy()
                    softmax_result[:, 1] = softmax_result[:, 1] - x
                    indices = np.argmax(softmax_result, 1)
                    indices = np.asarray(indices).flatten()
                    # break
                    for idx, ind in enumerate(indices):
                        if ind == 1:
                            name = str(n_b[idx], encoding='utf-8')
                            print(name, c_d_b[idx], values[idx])
                            csv_writer.writerow([name, c_d_b[idx][0], c_d_b[idx][1], c_d_b[idx][2], values[idx]])
                            cnt += 1
                    # true_count += np.sum(predictions)
                    step += 1
                print(cnt)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
    csv_file.close()


def main(argv=None):  # pylint: disable=unused-argument
    tf.gfile.MakeDirs(FLAGS.predict_dir)
    prediction(min_probability=0.5)


if __name__ == '__main__':
    tf.app.run()
