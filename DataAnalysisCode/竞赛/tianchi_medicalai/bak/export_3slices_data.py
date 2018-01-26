from src.dlmod import tfr_3slices_io

# p = Pool(4)
#
#
# def task(name):
#     if name == 0:
#         tfr_3slices_io.create_train_dataset('/home/cangzhu/data/candidates_csv/csv_train',
#                                             '/home/cangzhu/data/img/float_img/train',
#                                             '/home/cangzhu/data/candidates_3slices/candidates_train',
#                                             '/home/cangzhu/data/tfr_3slices/tfr_train')
#     elif name == 1:
#         tfr_3slices_io.create_train_dataset('/home/cangzhu/data/candidates_csv/csv_val',
#                                             '/home/cangzhu/data/img/float_img/val',
#                                             '/home/cangzhu/data/candidates_3slices/candidates_val',
#                                             '/home/cangzhu/data/tfr_3slices/tfr_val')
#     elif name == 2:
#         tfr_3slices_io.create_test_dataset('/home/cangzhu/data/candidates_csv/csv_test',
#                                            '/home/cangzhu/data/img/float_img/test',
#                                            '/home/cangzhu/data/candidates_3slices/candidates_test',
#                                            '/home/cangzhu/data/tfr_3slices/tfr_test')
#     elif name == 3:
#         tfr_3slices_io.create_test_dataset('/home/cangzhu/data/candidates_csv/csv_val',
#                                            '/home/cangzhu/data/img/float_img/val',
#                                            '/home/cangzhu/data/candidates_3slices2/candidates_val2',
#                                            '/home/cangzhu/data/tfr_3slices2/tfr_val2')
#     else:
#         pass
#
#
# if __name__ == '__main__':
#     print('Parent process %s.' % os.getpid())
#     for i in range(4):
#         p.apply_async(task, args=(i,))
#
#     print('Waiting for all subprocesses done...')
#     p.close()
#     p.join()
#     print('All subprocesses done.')

tfr_3slices_io.create_train_dataset('/home/cangzhu/data/candidates_csv/csv_train',
                                    '/home/cangzhu/data/img/float_img/train',
                                    '/home/cangzhu/data/candidates_3slices2/candidates_train',
                                    '/home/cangzhu/data/tfr_3slices2/tfr_train')

tfr_3slices_io.create_train_dataset('/home/cangzhu/data/candidates_csv/csv_val', '/home/cangzhu/data/img/float_img/val',
                                    '/home/cangzhu/data/candidates_3slices2/candidates_val',
                                    '/home/cangzhu/data/tfr_3slices2/tfr_val')

tfr_3slices_io.create_test_dataset('/home/cangzhu/data/candidates_csv/csv_test',
                                   '/home/cangzhu/data/img/float_img/test',
                                   '/home/cangzhu/data/candidates_3slices2/candidates_test',
                                   '/home/cangzhu/data/tfr_3slices2/tfr_test')

tfr_3slices_io.create_test_dataset('/home/cangzhu/data/candidates_csv/csv_val',
                                   '/home/cangzhu/data/img/float_img/val',
                                   '/home/cangzhu/data/candidates_3slices2/candidates_val2',
                                   '/home/cangzhu/data/tfr_3slices2/tfr_val2')
