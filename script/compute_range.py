import os
import numpy as np
import cv2
from PIL import Image
import json

# DEP_DIR="/home/lilium/myDataset/NYU_v2/"
# TRAIN_DEP_TXT="../datasets/nyu_path/train_depth.txt"

DEP_DIR="/home/lilium/myDataset/KITTI/datasets_KITTI/"
TRAIN_DEP_TXT="../datasets/kitti_path/eigen_test_depth_files.txt"

with open(TRAIN_DEP_TXT, 'r') as f:
    image_path_list = f.readlines()

d_min, d_max = 1000, 0
for i in range(len(image_path_list)):
    path = os.path.join(DEP_DIR, image_path_list[i].strip().split(' ')[0])
    img = np.array(Image.open(path))
    img = np.float32(img) / 256
    img = img[img>0]
    print('Processing [{}]/[{}]'.format(i, len(image_path_list)))
    if img.min() < d_min:
        d_min = img.min()
    if img.max() > d_max:
        d_max = img.max()

print(d_min, d_max)