"""This is a main module for candidate export


Author: Jns Ridge--##--ridgejns@gmail.com
"""

from src.imrp import mhd_io
from src.imrp import pre_process
from src.utilities import MHDSet
import numpy as np
import os
import csv
import argparse
import uuid


def main():
    parser = argparse.ArgumentParser(description='Export candidates initialization')
    parser.add_argument('-i', '--i_folder', type=str, default='/home/cangzhu/data/uint8_img/train',
                        help='input (image) folder')
    parser.add_argument('-o', '--o_folder', type=str, default='/home/cangzhu/data/csv_train',
                        help='output folder')
    parser.add_argument('-bn', '--box_num', type=str, default='0', help="mask for folder")
    parser.add_argument('-an', '--annotations', type=str,
                        default='/home/cangzhu/data/csv/train/annotations.csv',
                        help='annotations\' path')
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

    o_csv_folder = args.o_folder
    if not os.path.isdir(o_csv_folder):
        try:
            os.mkdir(o_csv_folder)
        except:
            raise IOError('Can not make that folder')

    o_csv_box_path = os.path.join(o_csv_folder, args.box_num)
    if not os.path.isdir(o_csv_box_path):
        try:
            os.mkdir(o_csv_box_path)
        except:
            raise IOError('Can not make that folder')

    o_head = ['dataset', 'seriesuid', 'uuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
              'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel', 'label']

    undetected_csv_file = open(os.path.join(o_csv_box_path, 'undetected_coord.csv'), 'w')
    undetected_csv_writer = csv.writer(undetected_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    undetected_csv_writer.writerow(o_head[0:-1])
    with open(os.path.join(o_csv_box_path, 'train_coord.csv'), 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(o_head)

        mhd_set = MHDSet()
        mhd_set.search_mhd_set(args.i_folder, recursive=True)

        for img_set in mhd_set.image_set:
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
            for i, img_path in enumerate(img_set.img_location):
                img_name = os.path.split(img_path)[-1].split('.')[0]
                cnt += 1
                print(img_name, img_set.description, cnt, img_set.count, "progress:%.2f" % (cnt / img_set.count))
                mhd_img = mhd_io.read(img_path)
                frames = mhd_img.frames

                # annotations exist, get the real node info.
                if annotations is not None:
                    nodes_idx = [i for i, v in enumerate(annotations[i_head[0]]) if v.lower() == img_name.lower()]
                    if len(nodes_idx) == 0:
                        print('image %s has NO node' % img_name)
                        continue

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

                masks_lung, masks_convex_lung, frames_contours_lung, frames_contours_lung_convex = pre_process.lung_mask(
                    frames, disp_progress=True)
                frames_blobs = pre_process.blob_detection(mhd_img, masks_convex_lung, frames_contours_lung,
                                                          kernel_size=0, disp_progress=True)
                cands, rinsed_cands = pre_process.search_candidates(mhd_img.spacing, frames_blobs)

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
                                    nodes_detected[k] = [img_set.description, img_name, cand_uuid.hex, c_x, c_y,
                                                         c_z, d_mm, d_p_x, d_p_y, d_p_z, d_d_pix, label_flag]
                                break
                        if label_flag == 0:
                            # ['dataset', 'seriesuid', 'uuid', 'coordX', ',coordY', 'coordZ', 'diameter_mm',
                            # 'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel', 'label]'
                            csv_writer.writerow(
                                [img_set.description, img_name, cand_uuid.hex, d_c_x, d_c_y, d_c_z, d_d_mm, d_p_x,
                                 d_p_y, d_p_z, d_d_pix, label_flag])
                    # annotations not exit, write the detected candidates directly
                    else:
                        csv_writer.writerow(
                            [img_set.description, img_name, cand_uuid.hex, d_c_x, d_c_y, d_c_z, d_d_mm, d_p_x, d_p_y,
                             d_p_z, d_d_pix])
                # annotations exit, write the real detected nodes in dataset of candidates
                if annotations is not None:
                    for k, node_detected in enumerate(nodes_detected):
                        if node_detected is not None:
                            csv_writer.writerow(node_detected)
                        else:
                            undetected_csv_writer.writerow(
                                [img_set.description] + [img_name] + [cand_uuid.hex] + nodes_coord[k] + nodes_pixel[k])
    undetected_csv_file.close()


if __name__ == '__main__':
    main()
