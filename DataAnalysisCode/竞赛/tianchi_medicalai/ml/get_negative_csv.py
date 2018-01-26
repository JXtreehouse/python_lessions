import os
import cv2
import csv
import random
import shutil
from tqdm import tqdm

in_path_0 = '/home/cangzhu/data/first_train_data_0'
in_path_1 = '/home/cangzhu/data/first_train_data_1'
out_path = '/home/cangzhu/data/first_train'

dir_list0 = os.listdir(in_path_0)

in_path_0 = '/home/cangzhu/data/first_train_data_0'
data_path = '/home/cangzhu/data/first_train_csv/negative_data.csv'
label_path = '/home/cangzhu/data/first_train_csv/negative_label.csv'

data_writer = csv.writer(open(data_path, 'w'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
label_writer = csv.writer(open(label_path, 'w'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

idxs = random.sample(range(len(dir_list0)), 20000)
pbar = tqdm(desc='extract', total=len(idxs))
i = 0

for idx in idxs:
    img = cv2.imread(os.path.join(in_path_0, dir_list0[idx]), 0)
    img = cv2.resize(img, (20, 20))
    name = dir_list0[idx]
    label = name.split('-')[-1].split('.')[0]
    img_line = img.flatten()
    data_writer.writerow(img_line)
    label_writer.writerow(label)
    pbar.update(1)

pbar.close()
    # print(name)
