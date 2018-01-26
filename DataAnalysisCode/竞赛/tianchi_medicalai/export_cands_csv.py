"""This is a main module for candidate export


Author: Jns Ridge--##--ridgejns@gmail.com
"""

import numpy as np
import os
import csv
import argparse
import uuid
import time
from multiprocessing import Pool
from src.imrp import mhd_io
from src.imrp import pre_process
from src.utilities import FileSet


def run_process(args, name, rg):
    """Multiprocess entrance

    Args:
    args: Default arguments
    name: name of the progress
    rg: file set range

    """
    fs = FileSet(['mhd'])
    fs.search_file_set(args.i_path, recursive=True)
    task = fs.file_set[rg[0]:rg[1]]
    # print(task)
    # time.sleep(2)
    print('task <%s> is running, pid: %s' % (name, os.getpid()))
    if os.path.isfile(args.annotations):
        with open(args.annotations, 'r') as csv_file:
            spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
            i_head = spamreader.__next__()
            annotations = {i_head[0]: [], i_head[1]: [], i_head[2]: [], i_head[3]: [], i_head[4]: []}
            for row in spamreader:
                for i in range(len(annotations)):
                    annotations[i_head[i]].append(row[i])
    else:
        annotations = None

    o_csv_folder = args.o_path
    if not os.path.isdir(o_csv_folder):
        try:
            os.makedirs(o_csv_folder)
        except:
            raise IOError('Can not make that folder')

    o_head = ['seriesuid', 'uuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
              'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel', 'label']

    nodes_num, d_nodes_num, d_candidates_num = 0, 0, 0
    for img_set in task:
        o_csv_folder = os.path.join(args.o_path, img_set.description)
        if not os.path.isdir(o_csv_folder):
            try:
                os.makedirs(o_csv_folder)
            except:
                raise ValueError('Can not make the folder <%s>' % o_csv_folder)
        cnt = 0

        if annotations is not None:
            o_csv_path0 = os.path.join(o_csv_folder, '%s-0.csv' % img_set.description)
            o_csv_path1 = os.path.join(o_csv_folder, '%s-1.csv' % img_set.description)
            o_csv_file0 = open(o_csv_path0, 'w')
            o_csv_file1 = open(o_csv_path1, 'w')

            csv_writer0 = csv.writer(o_csv_file0, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer1 = csv.writer(o_csv_file1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer0.writerow(o_head)
            csv_writer1.writerow(o_head)
            nodes_detected = []
        else:
            o_csv_path = os.path.join(o_csv_folder, '%s.csv' % img_set.description)
            o_csv_file = open(o_csv_path, 'w')
            csv_writer = csv.writer(o_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(o_head)
            nodes_detected = None

        for i, img_path in enumerate(img_set.location):
            img_name = os.path.split(img_path)[-1].split('.')[0]
            cnt += 1
            print(img_set.description, img_name, cnt, img_set.count, "progress:%.2f" % (cnt / img_set.count))
            mhd_img = mhd_io.read(img_path)
            frames = mhd_img.frames

            # annotations exist, get the real node info.
            if annotations is not None:
                nodes_idx = [i for i, v in enumerate(annotations[i_head[0]]) if v.lower() == img_name.lower()]
                if len(nodes_idx) == 0:
                    print('image %s has NO node' % img_name)
                    continue

                nodes_num += len(nodes_idx)
                nodes_coord = []
                nodes_pixel = []
                nodes_min_dist = []
                nodes_detected = []
                for j, node_idx in enumerate(nodes_idx):
                    # coordX, coordY, coordZ, diameter_mm
                    c_x, c_y, c_z, d_mm = float(annotations[i_head[1]][node_idx]), \
                                          float(annotations[i_head[2]][node_idx]), \
                                          float(annotations[i_head[3]][node_idx]), \
                                          float(annotations[i_head[4]][node_idx])
                    p_x, p_y, p_z = np.subtract([c_x, c_y, c_z], mhd_img.offset)

                    p_x, p_y, p_z = np.divide([p_x, p_y, p_z], mhd_img.spacing)
                    d_pix = d_mm / mhd_img.spacing[0]
                    # r_pix = d_pix / 2
                    nodes_coord.append([c_x, c_y, c_z, d_mm])
                    nodes_pixel.append([p_x, p_y, p_z, d_pix])
                    nodes_min_dist.append(np.inf)
                    nodes_detected.append(None)
                    csv_writer1.writerow([img_name, uuid.uuid1().hex, c_x, c_y, c_z, d_mm, p_x, p_y, p_z,
                                          d_pix, 1])

            masks_lung, masks_convex_lung, frames_contours_lung, frames_contours_lung_convex = pre_process.lung_mask(
                frames, disp_progress=True)
            frames_blobs = pre_process.blob_detection(mhd_img, masks_convex_lung, frames_contours_lung,
                                                      kernel_size=0, disp_progress=True)
            cands, rinsed_cands = pre_process.search_candidates(mhd_img.spacing, frames_blobs)

            d_candidates_num += len(cands)

            """detect no rinsed result"""
            for j, cand in enumerate(cands):
                cand_uuid = uuid.uuid1()
                cand = np.asarray(cand)
                # detected pixel z, detected pixel y, detected pixel x, detected radius pixel
                d_p_z, d_p_y, d_p_x, d_r_pix = cand[:, 0:4].mean(0)
                # detected diameter pixel
                d_d_pix = d_r_pix * 2

                # detected coordX, detected coordY, detected coordZ
                d_c_x, d_c_y, d_c_z = np.multiply([d_p_x, d_p_y, d_p_z], mhd_img.spacing)
                d_c_x, d_c_y, d_c_z = np.add([d_c_x, d_c_y, d_c_z], mhd_img.offset)
                # detected diameter_mm
                d_d_mm = d_d_pix * mhd_img.spacing[0]

                # annotations exist, search the nodes from detected nodes
                if annotations is not None:
                    label_flag = 0
                    for k, node_coord in enumerate(nodes_coord):
                        c_x, c_y, c_z, d_mm = node_coord
                        r_mm = d_mm / 2
                        dist = np.linalg.norm([d_c_x - c_x, d_c_y - c_y, d_c_z - c_z])
                        if dist < r_mm:
                            label_flag = 1
                            if dist < nodes_min_dist[k]:
                                nodes_min_dist[k] = dist
                                nodes_detected[k] = [img_name, cand_uuid.hex, c_x, c_y, c_z, d_mm, p_x, p_y, p_z,
                                                     d_pix, label_flag]
                            break
                    if label_flag == 0:
                        # ['seriesuid', 'uuid', 'coordX', ',coordY', 'coordZ', 'diameter_mm',
                        # 'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel', 'label]'
                        csv_writer0.writerow(
                            [img_name, cand_uuid.hex, d_c_x, d_c_y, d_c_z, d_d_mm, d_p_x,
                             d_p_y, d_p_z, d_d_pix, label_flag])
                # annotations not exit, write the detected candidates directly
                else:
                    csv_writer.writerow(
                        [img_name, cand_uuid.hex, d_c_x, d_c_y, d_c_z, d_d_mm, d_p_x, d_p_y,
                         d_p_z, d_d_pix, -1])
            if nodes_detected is None:
                d_nodes_num = -1
            else:
                d_nodes_num += (len(nodes_detected) - nodes_detected.count(None))
            print('real nodes:%d, detected_nodes:%s, detected_candidates:%d, \n' % (
                nodes_num, d_nodes_num, d_candidates_num))
        if annotations is not None:
            o_csv_file0.close()
            o_csv_file1.close()
        else:
            o_csv_file.close()


def main():
    parser = argparse.ArgumentParser(description='Export candidates initialization')
    parser.add_argument('-i', '--i_path', type=str, default='/media/cangzhu/data/img/uint8_img/test2',
                        help='import root folder')
    parser.add_argument('-o', '--o_path', type=str, default='/media/cangzhu/data/candidates_csv/csv_test2',
                        help='output root folder')
    parser.add_argument('-np', '--num_process', type=int, default=5, help="mask for folder")
    parser.add_argument('--rg', type=list, default=[], help='file set')
    parser.add_argument('-an', '--annotations', type=str,
                        default='',
                        help='annotations\' path')
    parser.add_argument('--version', type=str, default='0.1', help='version of the project')
    args = parser.parse_args()

    fs = FileSet(['mhd'])
    fs.search_file_set(args.i_path, recursive=True)
    folder_count = fs.valid_folder_count
    num_process = args.num_process
    each_batch_folder_count = np.ceil(folder_count / num_process).astype('int')
    fss = []
    rgs = []
    for i in range(num_process):
        fss.append(fs.file_set[i * each_batch_folder_count:(i + 1) * each_batch_folder_count])
        rgs.append([i * each_batch_folder_count, (i + 1) * each_batch_folder_count])

    fss_tmp = []
    rgs_tmp = []
    for i, fs_ in enumerate(fss):
        if len(fs_) >= 0:
            fss_tmp.append(fs_)
            rgs_tmp.append(rgs[i])
        else:
            break
    fss = fss_tmp
    num_process = len(fss)
    print('%d processes will be run...' % num_process)

    p = Pool(num_process)
    for i, fs_ in enumerate(fss):
        p.apply_async(run_process, args=(args, i, rgs[i]))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    main()
