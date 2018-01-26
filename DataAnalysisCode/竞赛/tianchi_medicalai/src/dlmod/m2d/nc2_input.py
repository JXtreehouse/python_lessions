"""Node classification 2 (NC-2) input module
This module include:
  Method:
    - read_nc2: read data from TFRecords (suffix is <.tfr>) file
    - distorted_inputs: distort the image and output a batch of the data

"""
import csv
import os

import tensorflow as tf

# from src.dlmod.m2d import tfr_io_uint8 as tfr_io
from src.dlmod.m2d import tfr_io

IMG_SIZE = tfr_io.IMG_SIZE
IMG_SHAPE = tfr_io.IMG_SHAPE  # z, y, x

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 86000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 30000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 80000

DATA_DIR = '/media/cangzhu/data/tfr_2d_float'


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


def read_nc2(filename_queue, slice=-1):
    """Reads and parses examples from NC-2 files
    
    Args:
    filename_queue: A queue of strings with the file(s) to read from.
    slice: Selected slice for image
    
    Returns:
    result: Class <NC2Recorder>.
    """

    class NC2Recorder(object):
        """
        name: 1-D tensor, bytes, name of the image
        coord_diameter: 1-D tensor, float32, [coordX, coordY, coordZ, diameter]
        frames: 3-D tensor, float32, [1 or 3, y, x]
        label: 1-D tensor, int32
        """

        def __init__(self):
            self.name = None
            self.coord_diameter = None
            self.frames = None
            self.label = None

    result = NC2Recorder()
    result.name, result.coord_diameter, result.frames, result.label = tfr_io.read_and_decode(filename_queue, slice)
    result.frames = tf.transpose(result.frames, [1, 2, 0])
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
    frames_batch: 4-D tensor [batch_num, y, x, 1 or 3]
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

    tf.summary.image('frame slice', frames_batch)
    name_batch = tf.reshape(name_batch, [batch_size])
    label_batch = tf.reshape(label_batch, [batch_size])
    return name_batch, c_d_batch, frames_batch, label_batch


def distorted_inputs(data_dir, batch_size, slice=-1):
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
    nc2r = read_nc2(filename_queue, slice)
    distorted_frames = nc2r.frames
    # nc2r.frames = tf.random_crop(nc2r.frames, [38, 38, 38])
    distorted_frames = tf.image.random_flip_left_right(distorted_frames)
    distorted_frames = tf.image.random_flip_up_down(distorted_frames)
    distorted_frames = tf.image.random_brightness(distorted_frames, max_delta=63)
    distorted_frames = tf.image.random_contrast(distorted_frames, lower=0.2, upper=1.8)
    distorted_frames = tf.image.per_image_standardization(distorted_frames)

    nc2r.frames = distorted_frames

    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    return _generate_image_and_info_batch(nc2r, min_queue_examples, batch_size, True)


def inputs(data_dir, batch_size, slice=-1):
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
    nc2r = read_nc2(filename_queue, slice)
    nc2r.frames = tf.image.per_image_standardization(nc2r.frames)

    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    return _generate_image_and_info_batch(nc2r, min_queue_examples, batch_size, False)
