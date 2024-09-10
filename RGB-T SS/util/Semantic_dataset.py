# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from ipdb import set_trace as st

# from train import augmentation_methods
# from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=640, input_w=640 ,transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test','test_day','test_night'], 'split must be "train"|"val"|"test"|"test_day"|"test_night"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.names)


    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.jpg' % (folder, name))
        image     = np.array(Image.open(file_path)) # (w,h,c)
        # image = np.array(image)
        image.flags.writeable = True
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')

        for func in self.transform:
            image, label = func(image, label)

        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h)), dtype=np.int64)

        return image ,label, name

    def get_test_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255

        return image, name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data



if __name__ == '__main__':
    # augmentation_methods = [
    #     RandomFlip(prob=0.5),
    #     RandomCrop(crop_rate=0.1, prob=1.0),
    #     # RandomCropOut(crop_rate=0.2, prob=1.0),
    #     # RandomBrightness(bright_range=0.15, prob=0.9),
    #     # RandomNoise(noise_range=5, prob=0.9),
    # ]
    data_dir = 'datatset/SemanticRT_dataset/SemanticRT_dataset/'
    MF_dataset(data_dir, 'train', have_label=True, transform=" ")
