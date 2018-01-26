"""Node classification 2 (NC-2) input module
This module include:
  Method:
    - read_nc2: read data from TFRecords (suffix is <.trf>) file
    - distorted_inputs: distort the image and output a batch of the data

"""
import csv
import os

import tensorflow as tf

from src.dlmod.m3d import tfr_io

IMG_SHAPE = [48, 48, 48]  # z, y, x
IMG_SHAPE_4D = [48, 48, 48, 1]  # z, y, x, depth
# IMG_SHAPE = tfr_io.IMG_SHAPE  # z, y, x
# IMG_SHAPE_4D = tfr_io.IMG_SHAPE_4D  # z, y, x, depth
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 86000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 30000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 140000

DATA_DIR = '/media/cangzhu/data/tfr_3d_float'


def get_epoch_num(s_path):
    """Get epoch number from tfr summary file

    """
    if not os.path.isfile(s_path):
        print('tfr summary did not exist, using default epoch number')
        return None
    with open(s_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='|')
        s_head = csv_reader.__next__()
        try:
            sn_idx = s_head.index('sample_num')
        except:
            print('tfr summary is not formatted, using default epoch number')
            return None
        try:
            srn_idx = s_head.index('sample_real_num')
        except:
            print('tfr summary is not formatted, using default epoch number')
            return None
        total = None
        for row in csv_reader:
            if row[0] == 'total':
                total = row
        if total is None:
            print('tfr summary has no <total> row, using default epoch number')
            return None
        epoch_num = int(total[sn_idx])
        if epoch_num < 0:
            epoch_num = int(total[srn_idx])
    return epoch_num


e_num = get_epoch_num(os.path.join(DATA_DIR, 'tfr_train/summary.csv'))
if e_num is not None:
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = e_num

e_num = get_epoch_num(os.path.join(DATA_DIR, 'tfr_val2/summary.csv'))
if e_num is not None:
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = e_num

e_num = get_epoch_num(os.path.join(DATA_DIR, 'tfr_test/summary.csv'))
if e_num is not None:
    NUM_EXAMPLES_PER_EPOCH_FOR_TEST = e_num


def read_nc2(filename_queue):
    """Reads and parses examples from NC-2 files
    
    Args:
    filename_queue: A queue of strings with the file(s) to read from.
    
    Returns:
    result: Class <NC2Recorder>.
    """

    class NC2Recorder(object):
        """
        name: 1-D tensor, bytes, name of the image
        coord_diameter: 1-D tensor, float32, [coordX, coordY, coordZ, diameter]
        frames: 4-D tensor, float32, [z, y, x, 1]
        label: 1-D tensor, int32
        """

        def __init__(self):
            self.name = None
            self.coord_diameter = None
            self.frames = None
            self.label = None

    result = NC2Recorder()
    result.name, result.coord_diameter, result.frames, result.label = tfr_io.read_and_decode(filename_queue)
    return result


def _generate_image_and_info_batch(nc2r, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of image and information
    
    Args
    nc2r: Class <NC2Recorder>
    min_queue_examples: Minimum number of samples to retain
    batch_size: Number of images of batch example
    shuffle: True/False
    
    Returns:
    name_batch: 1-D tensor 
    c_d_batch: 1-D tensor [coordX, coordY, coordZ, diameter_mm]
    frames_batch: 5-D tensor [batch_num, z, y, x, 1]
    label_batch: 1-D tensor
    """
    num_preprocess_threads = 8
    name = nc2r.name
    coord_diameter = nc2r.coord_diameter
    frames = nc2r.frames
    label = nc2r.label
    if shuffle:
        name_batch, c_d_batch, frames_batch, label_batch = tf.train.shuffle_batch(
            [name, coord_diameter, frames, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 2 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        name_batch, c_d_batch, frames_batch, label_batch = tf.train.batch(
            [name, coord_diameter, frames, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    # print(frames_batch[:, int(IMG_SHAPE[0] / 2), :, :])

    tf.summary.image('frame slice', frames_batch[:, int(IMG_SHAPE_4D[0] / 2), :, :])
    name_batch = tf.reshape(name_batch, [batch_size])
    label_batch = tf.reshape(label_batch, [batch_size])
    return name_batch, c_d_batch, frames_batch, label_batch


def _generate_image_batch(image, queue_pending, batch_size):
    num_preprocess_threads = 1
    images = tf.train.batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=queue_pending)

    tf.summary.image('images', images)
    return images


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for NC-2 Training using the TFRecordReader
    
    Args:
    data_dir: Path to the NC-2 data directory. 
    batch_size: Number of images per batch.
    
    Returns: 
    The result of <_generate_image_and_info_batch>.
    """
    file_names = []
    for name in os.listdir(data_dir):
        if name.split('.')[-1] == 'tfr':
            file_names.append(os.path.join(data_dir, name))
    filename_queue = tf.train.string_input_producer(file_names)
    nc2r = read_nc2(filename_queue)
    distorted_frames = tf.reshape(nc2r.frames, tfr_io.IMG_SHAPE)
    distorted_frames = tf.random_crop(distorted_frames, IMG_SHAPE)
    distorted_frames = tf.image.random_flip_left_right(distorted_frames)
    distorted_frames = tf.image.random_flip_up_down(distorted_frames)
    distorted_frames = tf.image.random_brightness(distorted_frames, max_delta=63)
    distorted_frames = tf.image.random_contrast(distorted_frames, lower=0.2, upper=1.8)
    distorted_frames = tf.image.per_image_standardization(distorted_frames)

    nc2r.frames = tf.reshape(distorted_frames, IMG_SHAPE_4D)

    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    return _generate_image_and_info_batch(nc2r, min_queue_examples, batch_size, True)


def inputs(data_dir, batch_size):
    """Construct input for NC-2 evaluation using TFRecordReader
    
    Args:
    data_dir: Path to the NC-2 data directory. 
    batch_size: Number of images per batch.
    
    Returns:
    The result of <_generate_image_and_info_batch>.
    """
    file_names = []
    for name in os.listdir(data_dir):
        if name.split('.')[-1] == 'tfr':
            file_names.append(os.path.join(data_dir, name))
    filename_queue = tf.train.string_input_producer(file_names)
    nc2r = read_nc2(filename_queue)
    frames = tf.reshape(nc2r.frames, tfr_io.IMG_SHAPE)
    # d = tf.subtract(tfr_io.IMG_SHAPE, IMG_SHAPE)
    dz, dy, dx = tfr_io.IMG_SHAPE[0] - IMG_SHAPE[0], \
                 tfr_io.IMG_SHAPE[1] - IMG_SHAPE[1], \
                 tfr_io.IMG_SHAPE[2] - IMG_SHAPE[2]

    frames = frames[dz:IMG_SHAPE[0] + dz, dy:IMG_SHAPE[1] + dy, dx:IMG_SHAPE[2] + dx]
    frames = tf.image.per_image_standardization(frames)
    nc2r.frames = tf.reshape(frames, IMG_SHAPE_4D)
    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    return _generate_image_and_info_batch(nc2r, min_queue_examples, batch_size, False)
