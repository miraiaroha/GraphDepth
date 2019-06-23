import os
import numpy as np
import cv2
from PIL import Image
import json

def continuous2discrete(depth, d_min, d_max, num_classes):
    #continuous_depth = np.clip(continuous_depth, d_min, d_max)
    mask = 1 - (depth > d_min) * (depth < d_max)
    depth = np.round(np.log(depth / d_min) / np.log(d_max / d_min) * (num_classes - 1))
    depth[mask] = 0
    return depth

def frequence():
    # DEP_DIR="/home/lilium/myDataset/NYU_v2/"
    # TRAIN_DEP_TXT="../datasets/nyu_path/train_depth_12k.txt"
    DEP_DIR="/home/lilium/myDataset/KITTI/datasets_KITTI/"
    TRAIN_DEP_TXT="../datasets/kitti_path/eigen_train_depth_files.txt"

    min_depth = 1.98
    max_depth = 80.0
    num_classes = 80

    with open(TRAIN_DEP_TXT, 'r') as f:
        image_path_list = f.readlines()
    
    pixs_class = dict(zip(np.arange(num_classes), np.zeros(num_classes)))
    pics_class = dict(zip(np.arange(num_classes), np.zeros(num_classes)))
    ratios = dict(zip(np.arange(num_classes), np.zeros(num_classes)))
    weights = dict(zip(np.arange(num_classes), np.zeros(num_classes)))

    for i in range(len(image_path_list)):
        path = os.path.join(DEP_DIR, image_path_list[i].strip().split(' ')[0])
        img = np.array(Image.open(path))
        img = np.float32(img) / 256
        H, W = img.shape[:2]
        disc_img = continuous2discrete(img, min_depth, max_depth, num_classes)

        print('Processing [{}]/[{}] image, path {}'.format(i, len(image_path_list), path))
        for j in range(num_classes):
            mask = disc_img == j
            num = np.sum(disc_img == j)
            if num:
                pixs_class[j] += num
                pics_class[j] += 1

    print('---------- pixels of class --------')
    print(pixs_class.values())
    print('---------- pictures of class -----------')
    print(pics_class.values())
    
    for j in range(num_classes):
        ratios[j] = pixs_class[j] / (pics_class[j] * H * W)
    print('------------ ratios of class -------------')
    print(ratios)

    ratio_median = np.median(list(ratios.values()))
    print('median of ratio: {}'.format(ratio_median))
    
    for j in range(num_classes):
        weights[j] = ratio_median / ratios[j]
    
    print('------------- weights of class -------------')
    print(list(weights.values()))

    save_dict = {'pixels': list(pixs_class.values()), 'pictures': list(pics_class.values()), 
                 'ratio': list(ratios.values()), 'median': ratio_median, 'weights': list(weights.values())}

    with open('./kitti_weights_22k_80.json', 'w') as f:
        json.dump(save_dict, f)


if __name__ == '__main__':
    frequence()
    
    # with open('./kitti_weights_22k_80.json', 'r') as f:
    #     weight = f.read()
    # print(weight)