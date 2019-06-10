import os
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import transforms as transforms
from dataloader import MyDataloader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

iheight, iwidth = 375, 1242 # raw image size

def make_dataset(root, txt):
    with open(txt, 'r') as f:
        List = []
        for line in f:
            left, right = line.strip('\n').split(' ')
            List.append(os.path.join(root, left))
            #List.append(right)
    return List

class KITTIDataset(MyDataloader):
    def __init__(self, root_image, root_depth, 
                 image_txt, depth_txt, 
                 min_depth, max_depth,
                 mode='train', make=make_dataset):
        super(KITTIDataset, self).__init__(root_image, root_depth, image_txt, depth_txt, min_depth, max_depth, mode, make)
        self.input_size = (160, 640)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        slide = np.random.uniform(0.0, 1.0)
        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.Resize(180 / 240), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.RandomCrop(self.input_size, slide),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        rgb_np = color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        # Scipy affine_transform produced RuntimeError when the depth map was
        # given as a 'numpy.ndarray'
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.Resize(180 / 240),
            transforms.CenterCrop(self.input_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        return rgb_np, depth_np


if __name__ == '__main__':
    HOME = os.environ['HOME']
    rgbdir = HOME + '/myDataset/KITTI/raw_data_KITTI/'
    depdir = HOME + '/myDataset/KITTI/datasets_KITTI/'
    trainrgb = '../datasets/kitti_path/eigen_train_files.txt'
    traindep = '../datasets/kitti_path/eigen_train_depth_files.txt'
    valrgb = '../datasets/kitti_path/eigen_test_files.txt'
    valdep = '../datasets/kitti_path/eigen_test_depth_files.txt'
    train_dataset = KITTIDataset(rgbdir, depdir, trainrgb, traindep, 1.8, 80, mode='train')
    val_dataset = KITTIDataset(rgbdir, depdir, valrgb, valdep, 1.8, 80, mode='val')
    trainloader = DataLoader(train_dataset, 20,
                                shuffle=True, num_workers=4, 
                                pin_memory=True, drop_last=False)
    valloader = DataLoader(val_dataset, 20,
                                shuffle=True, num_workers=4, 
                                pin_memory=True, drop_last=False)
    image, label = train_dataset[400]
    image_npy = image.numpy().transpose(1, 2, 0)
    label_npy = label.numpy().squeeze()

    #trainloader = iter(trainloader)
    #image, label = next(trainloader)
    print(image.shape, label.shape)
    print(label.max())
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_npy)
    plt.subplot(1, 2, 2)
    plt.imshow(label_npy, cmap='plasma')
    plt.show()