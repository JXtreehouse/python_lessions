"""This is the main port for image type converting.

Author: Jns Ridge--##--ridgejns@gmail.com
"""

import numpy as np
import argparse
import os
import time
from src.imrp import mhd_io
from src.utilities import FileSet
from multiprocessing import Pool


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
    print('task <%s> is running, pid: %s' % (name, os.getpid()))
    time.sleep(2)
    for img_set in task:
        o_path = os.path.join(args.o_path, img_set.description)
        if not os.path.isdir(o_path):
            try:
                os.makedirs(o_path)
            except:
                raise IOError('Can not make that folder')
        for img_path in img_set.location:
            t1 = time.time()
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mhd_img = mhd_io.read(img_path, args.c_type)
            o_f_path = os.path.join(o_path, '%s.mhd' % img_name)
            mhd_io.write(o_f_path, mhd_img, compress=True)
            print('p%s: %s, t:%.3f s' % (name, o_f_path, time.time() - t1))


def main():
    parser = argparse.ArgumentParser(description='Export candidates initialization')
    parser.add_argument('-i', '--i_path', type=str, default='/media/cangzhu/data/img/original_img/test2',
                        help='import root folder')
    parser.add_argument('-o', '--o_path', type=str, default='/media/cangzhu/data/img/uint8_img/test2',
                        help='output root folder')
    parser.add_argument('-ct', '--c_type', type=str, default='uint8', help='converting type')
    parser.add_argument('-np', '--num_process', type=int, default=5, help='number of the multiprocessing')
    parser.add_argument('--rg', type=list, default=[], help='file set')
    parser.add_argument('-v', '--version', type=str, default='0.1', help='version of the project')
    args = parser.parse_args()

    if not os.path.isdir(args.i_path):
        raise FileExistsError('invalid i_path')

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
