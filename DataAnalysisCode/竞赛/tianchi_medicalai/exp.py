from src.dlmod.m2d import tfr_io_uint8 as tfr_io

tfr_io.create_test_dataset('/media/cangzhu/data/candidates_csv/csv_test2',
                           '/media/cangzhu/data/img/uint8_img/test2',
                           None, '/media/cangzhu/data/tfr_2d_uint8/tfr_test2')
