"""This is a main module for candidate export


Author: Jns Ridge--##--ridgejns@gmail.com
"""

from src.imrp import mhd_io
from src.imrp import pre_process
from src.utilities import MHDSet
import numpy as np
import os
import csv
import cv2
import argparse
import uuid


def main():
    parser = argparse.ArgumentParser(description='Export candidates initialization')
    parser.add_argument('--i_path', type=str, default='/home/cangzhu/data/uint8_img/train',
                        help='import root folder')
    parser.add_argument('--box_num', type=str, default='-1', help="mask for folder")
    parser.add_argument('--o_path', type=str, default='/home/cangzhu/data/output_train',
                        help='output root folder')
    parser.add_argument('--annotations', type=str,
                        default='/home/cangzhu/data/csv/train/annotations.csv',
                        help='output root folder')
    parser.add_argument('--version', type=str, default='0.1', help='version of the project')
    args = parser.parse_args()

    nodes_num = 0
    r_d_nodes_num = 0
    nr_d_nodes_num = 0

    # datasets = args.datasets.split()
    # folder_mask = datasets
    if args.box_num == '00':
        # folder_mask = ['00']
        folder_mask = ['00', '01', '02']
    elif args.box_num == '01':
        folder_mask = ['03', '04', '05']
    elif args.box_num == '02':
        folder_mask = ['06', '07', '08']
    elif args.box_num == '03':
        folder_mask = ['09', '10', '11']
    elif args.box_num == '04':
        folder_mask = ['12', '13', '14']
    elif int(args.box_num) < 0:
        if len(args.box_num) == 2:
            folder_mask = ['0' + str(abs(int(args.box_num)))]
        else:
            folder_mask = [str(abs(int(args.box_num)))]

    else:
        args.box_num = '0'
        folder_mask = ['-1']

    if not os.path.isdir(args.i_path):
        raise FileExistsError('invalid i_path')
    if not os.path.isdir(args.o_path):
        try:
            os.mkdir(args.o_path)
        except:
            raise IOError('Can not make that folder')

    o_path_no_rinse_csv = os.path.join(args.o_path, 'no_rinse')
    o_path_rinse_csv = os.path.join(args.o_path, 'rinse')
    o_path_undetected_slice = os.path.join(args.o_path, 'undetected_slice')

    if not os.path.isdir(o_path_no_rinse_csv):
        try:
            os.mkdir(o_path_no_rinse_csv)
        except:
            raise IOError('Can not make that folder')

    if not os.path.isdir(o_path_rinse_csv):
        try:
            os.mkdir(o_path_rinse_csv)
        except:
            raise IOError('Can not make that folder')

    if not os.path.isdir(o_path_undetected_slice):
        try:
            os.mkdir(o_path_undetected_slice)
        except:
            raise IOError('Can not make that folder')

    with open(args.annotations, 'r') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i_head = spamreader.__next__()
        annotations = {i_head[0]: [], i_head[1]: [], i_head[2]: [], i_head[3]: [], i_head[4]: []}
        for row in spamreader:
            for i in range(len(annotations)):
                annotations[i_head[i]].append(row[i])

    o_csv_root_path = os.path.join(args.o_path, 'csv')
    if not os.path.isdir(o_csv_root_path):
        try:
            os.mkdir(o_csv_root_path)
        except:
            raise IOError('Can not make that folder')

    o_csv_box_path = os.path.join(o_csv_root_path, args.box_num)
    if not os.path.isdir(o_csv_box_path):
        try:
            os.mkdir(o_csv_box_path)
        except:
            raise IOError('Can not make that folder')

    o_head = ['seriesuid', 'uuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
              'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel', 'label']
    nr_detected_csv_file = open(os.path.join(o_csv_box_path, 'nr_detected.csv'), 'w')
    nr_d_writer = csv.writer(nr_detected_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    nr_d_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
                          'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel'])

    nr_undetected_csv_file = open(os.path.join(o_csv_box_path, 'nr_undetected.csv'), 'w')
    nr_ud_writer = csv.writer(nr_undetected_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    nr_ud_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
                           'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel'])

    r_detected_csv_file = open(os.path.join(o_csv_box_path, 'r_detected.csv'), 'w')
    r_d_writer = csv.writer(r_detected_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    r_d_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
                         'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel'])

    r_undetected_csv_file = open(os.path.join(o_csv_box_path, 'r_undetected.csv'), 'w')
    r_ud_writer = csv.writer(r_undetected_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    r_ud_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
                          'pixelX', 'pixelY', 'pixelZ', 'diameter_pixel'])

    mhd_set = MHDSet()
    mhd_set.search_mhd_set(args.i_path, recursive=True)
    r_d_num, r_nd_num, nr_d_num, nr_nd_num = 0, 0, 0, 0

    for img_set in mhd_set.image_set:
        box_flag = 0
        if folder_mask[0] == '-1':
            box_flag = 1
        else:
            for fm in folder_mask:
                if fm in img_set.description:
                    box_flag = 1
                    break
        if box_flag == 0:
            continue
        print(img_set.description)
        cnt = 0
        for i, img_path in enumerate(img_set.img_location):
            img_name = os.path.split(img_path)[-1].split('.')[0]
            cnt += 1
            print(img_name, img_set.description, cnt, img_set.count, "progress:%.2f" % (cnt / img_set.count))
            mhd_img = mhd_io.read(img_path)
            frames = mhd_img.frames.copy()
            frames1 = frames.copy()
            frames2 = frames.copy()

            nodes_idxs = [i for i, v in enumerate(annotations[i_head[0]]) if v.lower() == img_name.lower()]
            if len(nodes_idxs) == 0:
                print('image %s has NO node' % img_name)
                continue
            nodes_num += len(nodes_idxs)
            nodes_coord = []
            nodes_pixel = []
            nodes_min_dist = []
            nodes_detected = []
            for j, node_idx in enumerate(nodes_idxs):
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

            """detect rinsed result"""
            csv_path = os.path.join(o_path_rinse_csv, '%s.csv' % img_name)
            with open(csv_path, 'w') as csv_file:
                spamwriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(o_head)
                r_nd_num += len(rinsed_cands)
                for k, cand in enumerate(rinsed_cands):
                    cand_uuid = uuid.uuid1()
                    for blob in cand:
                        cv2.circle(frames1[int(blob[0])], (int(blob[2]), int(blob[1])), int(blob[3]), 255, 2)
                    cand = np.asarray(cand)
                    d_p_z, d_p_y, d_p_x, d_r_pix = cand[:, 0:4].mean(0)
                    d_c_x, d_c_y, d_c_z = np.multiply([d_p_x, d_p_y, d_p_z], mhd_img.spacing)
                    d_c_x, d_c_y, d_c_z = np.add([d_c_x, d_c_y, d_c_z], mhd_img.offset)
                    d_d_mm = d_r_pix * mhd_img.spacing[0] * 2

                    label_flag = 0
                    for l, node_coord in enumerate(nodes_coord):
                        c_x, c_y, c_z, d_mm = node_coord
                        r_mm = d_mm / 2
                        dist = np.linalg.norm([d_c_x - c_x, d_c_y - c_y, d_c_z - c_z])
                        if dist < r_mm:
                            label_flag = 1
                            if dist < nodes_min_dist[l]:
                                nodes_min_dist[l] = dist
                                nodes_detected[l] = [img_name, cand_uuid, d_c_x, d_c_y, d_c_z, d_d_mm, d_c_x, d_c_y,
                                                     d_c_z, d_d_mm, label_flag]
                                break
                        if label_flag == 0:
                            spamwriter.writerow(
                                [img_name, cand_uuid, d_c_x, d_c_y, d_c_z, d_d_mm, d_c_x, d_c_y, d_c_z, d_d_mm,
                                 label_flag])

                for k, node_detected in enumerate(nodes_detected):
                    if node_detected is not None:
                        r_d_nodes_num += 1
                        spamwriter.writerow(node_detected)
                        r_d_writer.writerow(node_detected + nodes_coord[k] + nodes_pixel[k])
                    else:
                        r_ud_writer.writerow(nodes_coord[k] + nodes_pixel[k])

                        # mhd_io.write(os.path.join(o_path_rinsed_img, '%s.mhd' % img_name), frames1, offset=mhd_img.offset,
                        #              spacing=mhd_img.spacing, compress=True)

            """detect no rinsed result"""
            # detect_result = np.zeros(len(real_pixs), 'bool')
            csv_path = os.path.join(o_path_no_rinse_csv, '%s.csv' % img_name)
            with open(csv_path, 'w') as csv_file:
                spamwriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(o_head)
                nr_nd_num += len(cands)
                for k, cand in enumerate(cands):
                    cand_uuid = uuid.uuid1()
                    for blob in cand:
                        cv2.circle(frames1[int(blob[0])], (int(blob[2]), int(blob[1])), int(blob[3]), 255, 2)
                    cand = np.asarray(cand)
                    d_p_z, d_p_y, d_p_x, d_r_pix = cand[:, 0:4].mean(0)
                    d_c_x, d_c_y, d_c_z = np.multiply([d_p_x, d_p_y, d_p_z], mhd_img.spacing)
                    d_c_x, d_c_y, d_c_z = np.add([d_c_x, d_c_y, d_c_z], mhd_img.offset)
                    d_d_mm = d_r_pix * mhd_img.spacing[0] * 2

                    label_flag = 0
                    for l, node_coord in enumerate(nodes_coord):
                        c_x, c_y, c_z, d_mm = node_coord
                        r_mm = d_mm / 2
                        dist = np.linalg.norm([d_c_x - c_x, d_c_y - c_y, d_c_z - c_z])
                        if dist < r_mm:
                            label_flag = 1
                            if dist < nodes_min_dist[l]:
                                nodes_min_dist[l] = dist
                                nodes_detected[l] = [img_name, cand_uuid, d_c_x, d_c_y, d_c_z, d_d_mm, d_c_x, d_c_y,
                                                     d_c_z, d_d_mm, label_flag]
                                break
                        if label_flag == 0:
                            spamwriter.writerow(
                                [img_name, cand_uuid, d_c_x, d_c_y, d_c_z, d_d_mm, d_c_x, d_c_y, d_c_z, d_d_mm,
                                 label_flag])
                for k, node_detected in enumerate(nodes_detected):
                    if node_detected is not None:
                        nr_d_nodes_num += 1
                        spamwriter.writerow(node_detected)
                        nr_d_writer.writerow(node_detected + nodes_coord[k] + nodes_pixel[k])
                    else:
                        nr_ud_writer.writerow([img_name] + nodes_coord[k] + nodes_pixel[k])
                        p_x, p_y, p_z, d_pix = nodes_pixel[k]
                        img = frames[int(p_z)]
                        img_2 = img.copy()
                        cv2.circle(img_2, (int(p_x), int(p_y)), int(d_pix), 255, 2)
                        cv2.imwrite(os.path.join(o_path_undetected_slice,
                                                 '%s-%d-%d-%d_0.png' % (img_name, int(p_x), int(p_y), int(p_z))), img)
                        cv2.imwrite(os.path.join(o_path_undetected_slice,
                                                 '%s-%d-%d-%d_1.png' % (img_name, int(p_x), int(p_y), int(p_z))), img_2)
            print(
                'real nodes:%d, nr_detected_nodes:%s, r_detected_nodes:%d' % (nodes_num, nr_d_nodes_num, r_d_nodes_num))
            print('nr_detected_candidates:%d, n_detected_candidates:%d\n' % (nr_nd_num, r_nd_num))

            # print('real nodes:%d, nr_detected_nodes:%s, r_detected_nodes:%d' % (nodes_num, nr_d_nodes_num, r_d_nodes_num))
            # print('nr_detected_candidates:%d, n_detected_candidates:%d' % (nr_nd_num, r_nd_num))


if __name__ == '__main__':
    main()
