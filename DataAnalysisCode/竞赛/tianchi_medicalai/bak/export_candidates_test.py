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


def main():
    parser = argparse.ArgumentParser(description='Export candidates initialization')
    parser.add_argument('--i_path', type=str, default='/home/cangzhu/data/uint8_img/test',
                        help='import root folder')
    parser.add_argument('--box_num', type=str, default='0', help="mask for folder")
    parser.add_argument('--o_path', type=str, default='/home/cangzhu/data/first_test_data',
                        help='output root folder')
    parser.add_argument('--version', type=str, default='0.1', help='version of the project')
    args = parser.parse_args()

    num_nodes = 0
    num_detected_nodes = 0
    num_rinsed_detected_nodes = 0

    if args.box_num == '06':
        folder_mask = ['00', '01', '02']
    elif args.box_num == '09':
        folder_mask = ['03', '04', '05']
    elif args.box_num == '13':
        folder_mask = ['06', '07', '08']
    elif args.box_num == '16':
        folder_mask = ['09', '10', '11']
    elif args.box_num == '18':
        folder_mask = ['12', '13', '14']
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

    o_path_no_rinse = os.path.join(args.o_path, 'no_rinse')
    o_path_rinse = os.path.join(args.o_path, 'rinse')

    o_path_no_rinsed_img = os.path.join(args.o_path, 'o_no_rinsed_img')
    o_path_rinsed_img = os.path.join(args.o_path, 'o_rinsed_img')

    o_path_undetected_slice = os.path.join(args.o_path, 'undetected_slice')

    if not os.path.isdir(o_path_no_rinse):
        try:
            os.mkdir(o_path_no_rinse)
        except:
            raise IOError('Can not make that folder')

    if not os.path.isdir(o_path_rinse):
        try:
            os.mkdir(o_path_rinse)
        except:
            raise IOError('Can not make that folder')

    if not os.path.isdir(o_path_no_rinsed_img):
        try:
            os.mkdir(o_path_no_rinsed_img)
        except:
            raise IOError('Can not make that folder')

    if not os.path.isdir(o_path_rinsed_img):
        try:
            os.mkdir(o_path_rinsed_img)
        except:
            raise IOError('Can not make that folder')

    if not os.path.isdir(o_path_undetected_slice):
        try:
            os.mkdir(o_path_undetected_slice)
        except:
            raise IOError('Can not make that folder')

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

    mhd_set = MHDSet()
    mhd_set.search_mhd_set(args.i_path, recursive=True)
    d_num = 0
    nd_num = 0

    csv_out_file = open('/home/cangzhu/data/first_test_data/csv/c_out.csv', 'w')
    csv_out_writer = csv.writer(csv_out_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

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
        print(img_set.description)
        cnt = 0
        for img_path in img_set.img_location:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            cnt += 1
            print(img_name, img_set.description, cnt, img_set.count, "progress:%.2f" % (cnt / img_set.count))
            mhd_img = mhd_io.read(img_path)
            frames = mhd_img.frames.copy()

            masks_lung, masks_convex_lung, frames_contours_lung, frames_contours_lung_convex = pre_process.lung_mask(
                frames, disp_progress=True)
            frames_blobs = pre_process.blob_detection(frames, mhd_img.spacing, masks_convex_lung, frames_contours_lung,
                                                      kernel_size=0,
                                                      disp_progress=True)
            cands, rinsed_cands = pre_process.search_candidates(mhd_img.spacing, frames_blobs, True)

            """detect no rinsed result"""
            csv_path = os.path.join(o_path_no_rinse, '%s.csv' % img_name)
            with open(csv_path, 'w') as csv_file:
                spamwriter = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(['seriesuid', 'pix_x', 'pix_y', 'pix_z', 'pix_diameter',
                                     'coordX', 'coordY', 'coordZ', 'diameter_mm'])
                for i, cand in enumerate(cands):
                    cand = np.asarray(cand)
                    z, y, x, r = cand[:, 0:4].mean(0)
                    c_z, c_y, c_x = np.multiply([z, y, x], mhd_img.spacing)
                    c_d = r * mhd_img.spacing[0] * 2

                    # node_img = mhd_io.node_crop(mhd_img, node_info)
                    # if node_img is not None:
                    #     mhd_io.write(os.path.join(o_path_rinsed_img, '%s-%d.mhd' % (img_name, i)), node_img)
                    #
                    est_slices = int(np.max((np.round(r * 2 / mhd_img.spacing[2]), 1)))
                    cbox = np.array([[x - r * 1.2, x + r * 1.2],
                                     [y - r * 1.2, y + r * 1.2],
                                     [z - est_slices / 1.8, z + est_slices / 1.8]]).astype('int')
                    bound = np.array([[0, mhd_img.width], [0, mhd_img.height], [0, mhd_img.depth]])
                    if (cbox[:, 0] < bound[:, 0]).sum() | (cbox[:, 1] > bound[:, 1]).sum():
                        continue
                    cropped_frames = mhd_img.frames[cbox[2][0]:cbox[2][1], cbox[1][0]:cbox[1][1],
                                     cbox[0][0]:cbox[0][1]].copy()
                    overlap = cropped_frames.sum(0)
                    overlap = (overlap - overlap.min()) / (overlap.max() - overlap.min()) * 255
                    overlap = overlap.astype('uint8')
                    cv2.imwrite(os.path.join(args.o_path, '%s-%d.png' % (img_name, i)), overlap)
                    cord_x = x * mhd_img.spacing[0] + mhd_img.offset[0]
                    cord_y = y * mhd_img.spacing[1] + mhd_img.offset[1]
                    cord_z = z * mhd_img.spacing[2] + mhd_img.offset[2]
                    cord_d = r * mhd_img.spacing[0] * 2
                    csv_out_writer.writerow(['%s-%d.png' % (img_name, i), cord_x, cord_y, cord_z, cord_d])
    csv_out_file.close()


if __name__ == '__main__':
    main()
