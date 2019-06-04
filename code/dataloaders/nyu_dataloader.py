import os
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import transforms as transforms
from dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size

def make_dataset(root, txt):
    with open(txt, 'r') as f:
        List = []
        for line in f:
            List.append(line.strip('\n'))
    return List

class NYUDataset(MyDataloader):
    def __init__(self, root_image, root_depth, 
                 image_txt, depth_txt, 
                 min_depth, max_depth,
                 mode='train', make=make_dataset):
        super(NYUDataset, self).__init__(root_image, root_depth, image_txt, depth_txt, min_depth, max_depth, mode, make)
        self.input_size = (224, 304)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.RandomCrop(self.input_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        rgb_np = color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.input_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np
