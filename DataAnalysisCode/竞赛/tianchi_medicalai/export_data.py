"""This is the main port for export tfr data from candidates csv file

Author: Jns Ridge--##--ridgejns@gmail.com
"""

import os
import argparse
import time
from multiprocessing import Pool

# select the correct <tfr_io> you wanted.

# from src.dlmod.m2d import tfr_io_uint8 as tfr_io
# from src.dlmod.m2d import tfr_io
from src.dlmod.m3d import tfr_io_uint8 as tfr_io
# from src.dlmod.m3d import tfr_io


def run_process(args):
    """Multiprocess entrance, it is 4 processes normally

    Args:
    args: [train_phase, name, csv_path, i_path, o_img_path, o_tfr_path]

    """
    print('task <%s> is running, pid: %s' % (args[1], os.getpid()))
    time.sleep(2)
    if args[0] is True:
        tfr_io.create_train_dataset(args[2], args[3], args[4], args[5])
    else:
        tfr_io.create_test_dataset(args[2], args[3], args[4], args[5])


def main():
    parser = argparse.ArgumentParser(description='Export candidates initialization')
    parser.add_argument('-i', '--i_path', type=str, default='/media/cangzhu/data/img/uint8_img',
                        help='import root folder')
    parser.add_argument('-c', '--c_path', type=str, default='/media/cangzhu/data/candidates_csv',
                        help='candidates csv path')
    parser.add_argument('-o', '--o_path', type=str, default='/media/cangzhu/data/tfr_3d_uint8',
                        help='output root folder')
    parser.add_argument('-v', '--version', type=str, default='0.1', help='version of the project')
    args = parser.parse_args()

    p = Pool(4)
    a = (True, 0, os.path.join(args.c_path, 'csv_train'), os.path.join(args.i_path, 'train'),
         None, os.path.join(args.o_path, 'tfr_train'))
    p.apply_async(run_process, args=(a,))
    a = (True, 1, os.path.join(args.c_path, 'csv_val'), os.path.join(args.i_path, 'val'),
         None, os.path.join(args.o_path, 'tfr_val'))
    p.apply_async(run_process, args=(a,))
    a = (False, 2, os.path.join(args.c_path, 'csv_test2'), os.path.join(args.i_path, 'test2'),
         None, os.path.join(args.o_path, 'tfr_test2'))
    p.apply_async(run_process, args=(a,))
    a = (False, 3, os.path.join(args.c_path, 'csv_val'), os.path.join(args.i_path, 'val'),
         None, os.path.join(args.o_path, 'tfr_val2'))
    p.apply_async(run_process, args=(a,))

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    main()
