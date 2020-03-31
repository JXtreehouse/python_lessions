
# 给图片添加马赛克
import os
import numpy as np
# opencv
# 安装注意　https://blog.csdn.net/qq_38660394/article/details/80581383
# pip install opencv-python
import cv2
import glob
import math

comparison_data_path = ''

types = (comparison_data_path + '/*.png', comparison_data_path + '/*.png') # the tuple of file types
grabbed_files = []

for ext_type in types:
    grabbed_files.extend(glob.glob(ext_type))


