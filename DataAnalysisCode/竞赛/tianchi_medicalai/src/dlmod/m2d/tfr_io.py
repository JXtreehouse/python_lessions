"""The module for TFRecords dataset create and decode
This module include:
  Method:
    - create_train_dataset:
    - read_and_decode:

Author: Jns Ridge--##--ridgejns@gmail.com
"""

import os
import numpy as np
import random
import warnings
import uuid
import cv2
import csv
import tensorflow as tf
from src.imrp import mhd_io
from tqdm import tqdm

IMG_SIZE = 64
IMG_SHAPE = (3, IMG_SIZE, IMG_SIZE)  # z, y, x
RESAMPLE_SPACING = (0.5, 0.5, 0.5)

s_head = ['dataset', 'sample_num', 'class0_num', 'class1_num', 'sample_real_num', 'class0_real_num', 'class1_real_num']


class Summary(object):
    def __init__(self, path, head):
        self.__f = open(path, 'w')
        self.__f_writer = csv.writer(self.__f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.__f_writer.writerow(head)

    def write(self, data):
        self.__f_writer.writerow(data)

    def close(self):
        self.__f.close()


def create_train_dataset(subsets_root_folder, in_img_folder, out_img_folder=None, tfr_folder=None, ns_rate=0.1):
    """Create train/val dataset by exported candidates' csv file(s)
    
    Args:
    subsets_root_folder: A folder path which keep the csv file.
    in_img_folder: A folder path which store the mhd image.
    out_img_folder: A folder path for saving the cropped nodes image.
    tfr_folder: A folder path for saving the TFRecords file.
    ns_rate: Negative sampling rate for negative sample.
     
    """
    if (out_img_folder is None) & (tfr_folder is None):
        raise ValueError('it has no out folder')
    if out_img_folder is not None:
        if os.path.isdir(out_img_folder):
            print('<out_img_folder> exists, please delete that folder before running <create_train_dataset>')
            return
        try:
            os.makedirs(out_img_folder)
        except:
            raise IOError('Can not make that folder')
    if tfr_folder is not None:
        if os.path.isdir(tfr_folder):
            print('<tfr_folder> exists, please delete that folder before running <create_train_dataset>')
            return
        try:
            os.makedirs(tfr_folder)
        except:
            raise IOError('Can not make that folder')

    subsets_name = []
    std_head = None
    for sn in os.listdir(subsets_root_folder):
        sn_path = os.path.join(subsets_root_folder, sn)
        if os.path.isdir(sn_path):
            subsets_name.append(os.path.join(sn))
        elif sn == 'std_head.csv':
            with open(sn_path) as head_file:
                head_csv = csv.reader(head_file, delimiter=',', quotechar='|')
                std_head = head_csv.__next__()
    if std_head is None:
        warnings.warn('<std_head.csv> didn\'t exist in the root folder, using normal std_head', RuntimeWarning)
        std_head = ['seriesuid', 'uuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'pixelX', 'pixelY', 'pixelZ',
                    'diameter_pixel', 'label']

    summary = Summary(os.path.join(tfr_folder, 'summary.csv'), s_head)
    total_class0_num, total_class1_num = 0, 0
    total_class0_real_num, total_class1_real_num = 0, 0

    for sn in subsets_name:
        class0_num, class1_num = 0, 0
        subset_folder = os.path.join(subsets_root_folder, sn)
        subset_ii_folder = os.path.join(in_img_folder, sn)

        if out_img_folder is not None:
            subset_oi_folder = os.path.join(out_img_folder, sn)
            if os.path.isdir(subset_oi_folder):
                return
            try:
                os.makedirs(subset_oi_folder)
            except:
                raise IOError('Can not make that folder')

        tfr_writer = None
        if tfr_folder is not None:
            tfr_writer = tf.python_io.TFRecordWriter(os.path.join(tfr_folder, '%s.tfr' % sn))

        csvs_name = []
        data0_csv_path, data1_csv_path = None, None
        for cn in os.listdir(subset_folder):
            if cn.split('.')[-1].lower() == 'csv':
                csvs_name.append(cn)
                label = int(cn.split('.')[0].split('-')[-1])
                if label == 0:
                    data0_csv_path = os.path.join(subset_folder, cn)
                elif label == 1:
                    data1_csv_path = os.path.join(subset_folder, cn)
                else:
                    raise ValueError('subset_folder exists unexpected file')
        if (data0_csv_path is None) | (data1_csv_path is None):
            raise IOError('<%s> dataset lost' % sn)

        data0 = []
        with open(data0_csv_path, 'r') as data0_csv:
            data0_csv_reader = csv.reader(data0_csv, delimiter=',', quotechar='|')
            i_head = data0_csv_reader.__next__()
            if i_head != std_head:
                for i, head in enumerate(std_head):
                    if head != i_head(i):
                        raise ValueError('Can not find <%s> in the <%s> table' % (head, cn))
            for row in data0_csv_reader:
                data0.append(row)

        data1 = []
        with open(data1_csv_path, 'r') as data1_csv:
            data1_csv_reader = csv.reader(data1_csv, delimiter=',', quotechar='|')
            i_head = data1_csv_reader.__next__()
            if i_head != std_head:
                for i, head in enumerate(std_head):
                    if head != i_head(i):
                        raise ValueError('Can not find <%s> in the <%s> table' % (head, cn))
            for row in data1_csv_reader:
                data1.append(row)

        data0 = np.asarray(data0)
        data1 = np.asarray(data1)
        img_name_set = set(data0[:, 0])
        num_data0 = len(data0)
        num_data1 = len(data1)
        s_num_data0 = ns_rate * num_data0
        s_num_data1 = int((random.random() * 0.2 + 0.9) * s_num_data0)
        ps_rate = s_num_data1 / num_data1
        pbar = tqdm(desc='%s' % sn, total=len(img_name_set))
        for img_name in img_name_set:
            i_img_path = os.path.join(subset_ii_folder, '%s.mhd' % img_name)
            idxs_d0 = [i for i, v in enumerate(data0[:, 0]) if v.lower() == img_name.lower()]
            idxs_d1 = [i for i, v in enumerate(data1[:, 0]) if v.lower() == img_name.lower()]
            sub_data0 = data0[idxs_d0]
            sub_data1 = data1[idxs_d1]
            num_sub_data0 = len(idxs_d0)
            num_sub_data1 = len(idxs_d1)
            s_num_sub_data0 = int(ns_rate * num_sub_data0)
            s_num_sub_data1 = int(ps_rate * num_sub_data1)
            lb_list = np.append(np.zeros(s_num_sub_data0, 'int'), np.ones(s_num_sub_data1, 'int'))
            random.shuffle(lb_list)
            s_idxs_d0 = random.sample(range(num_sub_data0), s_num_sub_data0)
            # print(i_img_path)
            mhd_img = mhd_io.read(i_img_path)
            for lb in lb_list:
                out_img_path = None
                ni = mhd_io.NODE()
                ni.ctype = 'pixel'
                if lb == 0:
                    label = 0
                    idx = s_idxs_d0.pop()
                    d = sub_data0[idx]
                    # ni.coord_x, ni.coord_y, ni.coord_z, ni.diameter = np.asarray(d[6:-1], 'float')
                    p_x, p_y, p_z, d_pix = d[std_head.index('pixelX')], d[std_head.index('pixelY')], d[
                        std_head.index('pixelZ')], d[std_head.index('diameter_pixel')]
                    p_x, p_y, p_z, d_pix = np.array([p_x, p_y, p_z, d_pix], 'float')
                    ni.coord_x, ni.coord_y, ni.coord_z, ni.diameter = p_x, p_y, p_z, d_pix
                    node_mhd_img = mhd_io.node_crop(mhd_img, ni, resample_spacing=RESAMPLE_SPACING,
                                                    crop_shape=[IMG_SIZE, IMG_SIZE, IMG_SIZE])
                    if node_mhd_img is not None:
                        class0_num += 1
                        mask = np.zeros([IMG_SIZE, IMG_SIZE], 'uint8')
                        mask = cv2.circle(mask, (int(IMG_SIZE / 2), int(IMG_SIZE / 2)), int(d_pix / 2) + 5, 255,
                                          cv2.FILLED)
                        mask = mask.astype('bool')
                        yx_img = node_mhd_img.frames[int(IMG_SIZE / 2), :, :]
                        # yx_img[~mask] = 0
                        zx_img = node_mhd_img.frames[:, int(IMG_SIZE / 2), :]
                        # zx_img[~mask] = 0
                        zy_img = node_mhd_img.frames[:, :, int(IMG_SIZE / 2)]
                        # zy_img[~mask] = 0
                        merge_img = np.asarray([zx_img, yx_img, zy_img])
                        if out_img_folder is not None:
                            out_img_path = os.path.join(subset_oi_folder, '%s-%s-0.png' % (img_name, d[1]))
                            # out_path = os.path.join(subset_oi_folder, '%s-%s-0.mhd' % (img_name, d[1]))

                elif lb == 1:
                    label = 1
                    idx = random.choice(range(num_sub_data1))
                    d = sub_data1[idx]
                    # ni.coord_x, ni.coord_y, ni.coord_z, ni.diameter = np.asarray(d[6:-1], 'float')
                    p_x, p_y, p_z, d_pix = d[std_head.index('pixelX')], d[std_head.index('pixelY')], d[
                        std_head.index('pixelZ')], d[std_head.index('diameter_pixel')]
                    p_x, p_y, p_z, d_pix = np.array([p_x, p_y, p_z, d_pix], 'float')
                    ni.coord_x, ni.coord_y, ni.coord_z, ni.diameter = p_x, p_y, p_z, d_pix
                    # max_shift = min((ni.diameter / 2), 5)
                    # x_offset = (random.random() * ni.diameter) - max_shift * 0.8
                    # y_offset = (random.random() * ni.diameter) - max_shift * 0.8
                    # z_offset = (random.random() * ni.diameter) - max_shift * 0.8
                    # ni.coord_x, ni.coord_y, ni.coord_z = np.subtract([ni.coord_x, ni.coord_y, ni.coord_z],
                    #                                                  [x_offset, y_offset, z_offset])
                    node_mhd_img = mhd_io.node_crop(mhd_img, ni, resample_spacing=RESAMPLE_SPACING,
                                                    crop_shape=[IMG_SIZE, IMG_SIZE, IMG_SIZE])
                    if node_mhd_img is not None:
                        class1_num += 1
                        mask = np.zeros([IMG_SIZE, IMG_SIZE], 'uint8')
                        mask = cv2.circle(mask, (int(IMG_SIZE / 2), int(IMG_SIZE / 2)), int(d_pix / 2) + 5, 255,
                                          cv2.FILLED)
                        mask = mask.astype('bool')
                        yx_img = node_mhd_img.frames[int(IMG_SIZE / 2), :, :]
                        # yx_img[~mask] = 0
                        zx_img = node_mhd_img.frames[:, int(IMG_SIZE / 2), :]
                        # zx_img[~mask] = 0
                        zy_img = node_mhd_img.frames[:, :, int(IMG_SIZE / 2)]
                        # zy_img[~mask] = 0
                        merge_img = np.asarray([zx_img, yx_img, zy_img])
                        if out_img_folder is not None:
                            out_img_path = os.path.join(subset_oi_folder,
                                                        '%s-%s-%s-1.png' % (img_name.split('.')[0], d[1], uuid.uuid1()))
                else:
                    warnings.warn('unexpected label <%s> in image <%s>' % (str(lb), img_name), RuntimeWarning)
                    continue

                if out_img_path is not None:
                    merge_img_uint8 = merge_img.transpose([1, 2, 0])
                    merge_img_uint8 = merge_img_uint8 * 255
                    merge_img_uint8 = merge_img_uint8.astype('uint8')
                    cv2.imwrite(out_img_path, merge_img_uint8)
                    # mhd_io.write(out_img_path, node_mhd_img, compress=True)

                if (node_mhd_img is not None) & (tfr_writer is not None):
                    # c_x, c_y, c_z, d_mm = np.asarray(d[2:6], 'float')
                    c_x, c_y, c_z, d_mm = d[std_head.index('coordX')], d[std_head.index('coordY')], d[
                        std_head.index('coordZ')], d[std_head.index('diameter_mm')]
                    c_d = np.array([c_x, c_y, c_z, d_mm], 'float32')
                    c_d_raw = c_d.tobytes()
                    frames = merge_img.astype('float32')
                    frames_raw = frames.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name.encode('utf-8')])),
                        'coord_diameter': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_d_raw])),
                        'frames_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[frames_raw])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }))
                    tfr_writer.write(example.SerializeToString())
            pbar.update()
        total_class0_num += class0_num
        total_class1_num += class1_num
        total_class0_real_num += num_data0
        total_class1_real_num += num_data1
        summary.write([sn, class0_num + class1_num, class0_num, class1_num,
                       num_data0 + num_data1, num_data0, num_data1])
        tfr_writer.close()
        pbar.close()
    summary.write(['total', total_class0_num + total_class1_num, total_class0_num, total_class1_num,
                   total_class0_real_num + total_class1_real_num, total_class0_real_num, total_class1_real_num])
    summary.close()


def create_test_dataset(subsets_root_folder, in_img_folder, out_img_folder=None, tfr_folder=None):
    """Create test dataset by exported candidates' csv file(s)

    Args:
    subsets_root_folder: A folder path which keep the csv file.
    in_img_folder: A folder path which store the mhd image.
    out_img_folder: A folder path for saving the cropped nodes image.
    tfr_folder: A folder path for saving the TFRecords file.
    ns_rate: Negative sampling rate for negative sample.

    """
    if (out_img_folder is None) & (tfr_folder is None):
        raise ValueError('it has no out folder')
    if out_img_folder is not None:
        if os.path.isdir(out_img_folder):
            print('<out_img_folder> exists, please delete that folder before running <create_train_dataset>')
            return
        try:
            os.makedirs(out_img_folder)
        except:
            raise IOError('Can not make that folder')
    if tfr_folder is not None:
        if os.path.isdir(tfr_folder):
            print('<tfr_folder> exists, please delete that folder before running <create_train_dataset>')
            return
        try:
            os.makedirs(tfr_folder)
        except:
            raise IOError('Can not make that folder')

    subsets_name = []
    std_head = None
    for sn in os.listdir(subsets_root_folder):
        sn_path = os.path.join(subsets_root_folder, sn)
        if os.path.isdir(sn_path):
            subsets_name.append(os.path.join(sn))
        elif sn == 'std_head.csv':
            with open(sn_path) as head_file:
                head_csv = csv.reader(head_file, delimiter=',', quotechar='|')
                std_head = head_csv.__next__()
    if std_head is None:
        warnings.warn('<std_head.csv> didn\'t exist in the root folder, using normal std_head', RuntimeWarning)
        std_head = ['seriesuid', 'uuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'pixelX', 'pixelY', 'pixelZ',
                    'diameter_pixel', 'label']

    summary = Summary(os.path.join(tfr_folder, 'summary.csv'), s_head)
    total_sample_real_num = 0
    total_class0_real_num, total_class1_real_num = 0, 0

    for sn in subsets_name:
        sample_real_num = 0
        class0_real_num, class1_real_num = -1, -1

        subset_folder = os.path.join(subsets_root_folder, sn)
        subset_ii_folder = os.path.join(in_img_folder, sn)

        if out_img_folder is not None:
            subset_oi_folder = os.path.join(out_img_folder, sn)
            if os.path.isdir(subset_oi_folder):
                return
            try:
                os.makedirs(subset_oi_folder)
            except:
                raise IOError('Can not make that folder')

        tfr_writer = None
        if tfr_folder is not None:
            tfr_writer = tf.python_io.TFRecordWriter(os.path.join(tfr_folder, '%s.tfr' % sn))

        for cn in os.listdir(subset_folder):
            lb = cn.split('.')[0].split('-')[-1]
            if lb == '0':
                lb = int(lb)
                class0_real_num = 0
            elif lb == '1':
                lb = int(lb)
                class1_real_num = 0
            else:
                lb = -1
            print(cn, lb)
            data_csv_path = os.path.join(subset_folder, cn)

            data = []
            with open(data_csv_path, 'r') as data_csv:
                data_csv_reader = csv.reader(data_csv, delimiter=',', quotechar='|')
                i_head = data_csv_reader.__next__()
                if i_head != std_head:
                    for i, head in enumerate(std_head):
                        if head != i_head(i):
                            raise ValueError('Can not find <%s> in the <%s> table' % (head, cn))
                for row in data_csv_reader:
                    data.append(row)

            data = np.asarray(data)
            img_name_set = set(data[:, 0])

            pbar = tqdm(desc='%s' % sn, total=len(img_name_set))
            for img_name in img_name_set:
                i_img_path = os.path.join(subset_ii_folder, '%s.mhd' % img_name)
                idxs_d = [i for i, v in enumerate(data[:, 0]) if v.lower() == img_name.lower()]
                mhd_img = mhd_io.read(i_img_path)
                for idx in idxs_d:
                    out_img_path = None
                    ni = mhd_io.NODE()
                    ni.ctype = 'pixel'
                    d = data[idx]
                    label = int(d[std_head.index(('label'))])

                    # ni.coord_x, ni.coord_y, ni.coord_z, ni.diameter = np.asarray(d[6:-1], 'float')
                    p_x, p_y, p_z, d_pix = d[std_head.index('pixelX')], d[std_head.index('pixelY')], d[
                        std_head.index('pixelZ')], d[std_head.index('diameter_pixel')]
                    p_x, p_y, p_z, d_pix = np.array([p_x, p_y, p_z, d_pix], 'float')
                    ni.coord_x, ni.coord_y, ni.coord_z, ni.diameter = p_x, p_y, p_z, d_pix
                    node_mhd_img = mhd_io.node_crop(mhd_img, ni, resample_spacing=RESAMPLE_SPACING,
                                                    crop_shape=[IMG_SIZE, IMG_SIZE, IMG_SIZE])
                    if node_mhd_img is not None:
                        sample_real_num += 1
                        if label == 0:
                            class0_real_num += 1
                        elif label == 1:
                            class1_real_num += 1
                        else:
                            pass
                        mask = np.zeros([IMG_SIZE, IMG_SIZE], 'uint8')
                        mask = cv2.circle(mask, (int(IMG_SIZE / 2), int(IMG_SIZE / 2)), int(d_pix / 2) + 5, 255,
                                          cv2.FILLED)
                        mask = mask.astype('bool')
                        yx_img = node_mhd_img.frames[int(IMG_SIZE / 2), :, :]
                        # yx_img[~mask] = 0
                        zx_img = node_mhd_img.frames[:, int(IMG_SIZE / 2), :]
                        # zx_img[~mask] = 0
                        zy_img = node_mhd_img.frames[:, :, int(IMG_SIZE / 2)]
                        # zy_img[~mask] = 0
                        merge_img = np.asarray([zx_img, yx_img, zy_img])

                    if (node_mhd_img is not None) & (out_img_folder is not None):
                        out_img_path = os.path.join(subset_oi_folder, '%s-%s.png' % (img_name, d[1]))
                        merge_img_uint8 = merge_img.transpose([1, 2, 0])
                        merge_img_uint8 = merge_img_uint8 * 255
                        merge_img_uint8 = merge_img_uint8.astype('uint8')
                        cv2.imwrite(out_img_path, merge_img_uint8)

                    if (node_mhd_img is not None) & (tfr_writer is not None):
                        # c_x, c_y, c_z, d_mm = np.asarray(d[2:6], 'float')
                        c_x, c_y, c_z, d_mm = d[std_head.index('coordX')], d[std_head.index('coordY')], d[
                            std_head.index('coordZ')], d[std_head.index('diameter_mm')]
                        c_d = np.array([c_x, c_y, c_z, d_mm], 'float32')
                        c_d_raw = c_d.tobytes()
                        frames = merge_img.astype('float32')
                        frames_raw = frames.tobytes()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name.encode('utf-8')])),
                            'coord_diameter': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_d_raw])),
                            'frames_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[frames_raw])),
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                        }))
                        tfr_writer.write(example.SerializeToString())
                pbar.update(1)
            pbar.close()
        # sample_real_num = class0_real_num + class1_real_num
        # total_sample_real_num += sample_real_num
        # total_class0_real_num += class0_real_num
        # total_class1_real_num += class1_real_num
        if sample_real_num >= 0:
            total_sample_real_num += sample_real_num
        if class0_real_num >= 0:
            total_class0_real_num += class0_real_num
        if class1_real_num >= 0:
            total_class1_real_num += class1_real_num
        print(sample_real_num, class0_real_num, class1_real_num)
        summary.write(
            [sn, sample_real_num, class0_real_num, class1_real_num, sample_real_num, class0_real_num, class1_real_num])
        tfr_writer.close()
    if total_sample_real_num == 0:
        total_sample_real_num = -1
    if total_class0_real_num == 0:
        total_class0_real_num = -1
    if total_class1_real_num == 0:
        total_class1_real_num = -1
    summary.write(['total', total_sample_real_num, total_class0_real_num, total_class1_real_num, total_sample_real_num,
                   total_class0_real_num, total_class1_real_num])
    summary.close()


def read_and_decode(filename_queue, slice=-1):
    """Read TFRecords data and decode the data to normal form
    
    Args:
    filename_queue: A queue of strings with the file(s) to read from.
    
    Returns:
    name, coord_diameter, frames, label
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_examples, features={
        'name': tf.FixedLenFeature([], tf.string),
        'coord_diameter': tf.FixedLenFeature([], tf.string),
        'frames_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    name = tf.cast(features['name'], tf.string)
    coord_diameter = tf.decode_raw(features['coord_diameter'], tf.float32)
    coord_diameter = tf.reshape(coord_diameter, (4,))
    frames = tf.decode_raw(features['frames_raw'], tf.float32) - 0.5
    # frames = frames
    # frames = frames.resample([3, IMG_SIZE, IMG_SIZE])
    frames = tf.reshape(frames, [3, IMG_SIZE, IMG_SIZE])  # z(zx, yx, zy), y, x

    if slice == -1:
        pass
    elif (isinstance(slice, list)) | (isinstance(slice, tuple)):
        s0, s1 = slice
        f0 = tf.reshape(frames[s0], [1, IMG_SIZE, IMG_SIZE])
        f1 = tf.reshape(frames[s1], [1, IMG_SIZE, IMG_SIZE])
        frames = tf.concat([f0, f1], 0)
        frames = tf.reshape(frames, [2, IMG_SIZE, IMG_SIZE])
    else:
        frames = tf.reshape(frames[slice], [1, IMG_SIZE, IMG_SIZE])
    # frames = tf.cast(frames, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return name, coord_diameter, frames, label
