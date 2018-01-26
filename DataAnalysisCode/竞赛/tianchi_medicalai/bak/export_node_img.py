"""This is a main module for candidate export


Author: Jns Ridge--##--ridgejns@gmail.com
"""

from src.imrp import mhd_io
from src.imrp import pre_process
from src.utilities import FileSet
import numpy as np
import os
import csv
import argparse
import cv2
import uuid


def main():
    parser = argparse.ArgumentParser(description='Export candidates initialization')
    parser.add_argument('-i', '--i_folder', type=str, default='/home/cangzhu/data/img/uint8_img/train',
                        help='input (image) folder')
    parser.add_argument('-o', '--o_folder', type=str, default='/home/cangzhu/data/out_img/train',
                        help='output folder')
    parser.add_argument('-bn', '--box_num', type=str, default='0', help="mask for folder")
    parser.add_argument('-an', '--annotations', type=str,
                        default='/home/cangzhu/data/csv/train/annotations.csv',
                        help='annotations path')
    parser.add_argument('--version', type=str, default='0.1', help='version of the project')
    args = parser.parse_args()

    if args.box_num == '00':
        folder_mask = ['00', '01', '02']
    elif args.box_num == '01':
        folder_mask = ['03', '04', '05']
    elif args.box_num == '02':
        folder_mask = ['06', '07', '08']
    elif args.box_num == '03':
        folder_mask = ['09', '10', '11']
    elif args.box_num == '04':
        folder_mask = ['12', '13', '14']
    else:
        args.box_num = '0'
        folder_mask = ['-1']

    # if args.box_num == '00':
    #     folder_mask = ['00']
    # elif args.box_num == '01':
    #     folder_mask = ['01']
    # elif args.box_num == '02':
    #     folder_mask = ['02']
    # elif args.box_num == '03':
    #     folder_mask = ['03']
    # elif args.box_num == '04':
    #     folder_mask = ['04']
    # else:
    #     args.box_num = '0'
    #     folder_mask = ['-1']

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

    if not os.path.isdir(args.o_folder):
        try:
            os.mkdir(args.o_folder)
        except:
            raise IOError('Can not make that folder')

    mhd_set = FileSet(('.mhd',))
    mhd_set.search_file_set(args.i_folder, recursive=True)
    print(mhd_set.file_set)

    nodes_num, d_nodes_num, d_candidates_num = 0, 0, 0

    for img_set in mhd_set.file_set:
        f = 0
        if folder_mask[0] == '-1':
            f = 1
        else:
            for fm in folder_mask:
                if fm in img_set.description:
                    f = 1
                    break
        if f == 0:
            continue

        cnt = 0

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
                    cv2.imwrite(os.path.join(args.o_folder, '%s-%d_%d_%d.png' % (img_name, p_x, p_y, p_z)),
                                frames[int(p_z), :, :])


if __name__ == '__main__':
    main()
