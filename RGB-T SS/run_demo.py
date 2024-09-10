# coding:utf-8
import os
import argparse
import time

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from util.util import visualize
from model import MFNet
from train import n_class, model_dir


def main():

    model = eval(args.model_name)(n_class=n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    if os.path.exists(final_model_file):
        model.load_state_dict(torch.load(final_model_file))
    elif os.path.exists(checkpoint_model_file):
        model.load_state_dict(torch.load(checkpoint_model_file))
    else:
        raise Exception('| model file do not exists in %s' % model_dir)
    print('| model loaded!')
    data_dir = '...../test.txt'
    img_dir='....../image/'
    gt_dir = '......./visual_color_GT/'
    with open(os.path.join(data_dir), 'r') as f:
        files = [img_dir+name.strip()+'.png' for name in f.readlines()]
    # files = os.listdir('image')
    images = []
    fpath  = []
    gt_images=[]
    for file in files[47:80]:
        if file[-3:] != 'png': continue
        # gt_file = gt_dir+file.split('/')[-1].split('.')[0]+'.jpg'
        fpath.append("......"+file.split('/')[-1])
        images.append( np.asarray(Image.open(file)) )
        # gt_images=np.asarray(Image.open(gt_file))
        # cv2.imwrite("......"+file.split('/')[-1],gt_images)
    images = np.asarray(images, dtype=np.float32).transpose((0,3,1,2))/255.
    images = Variable(torch.tensor(images))
    if args.gpu >= 0: images = images.cuda(args.gpu)

    model.eval()
    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(1)
        visualize(fpath, predictions)

    print('| prediction files have been saved in image/')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run MFNet demo with pytorch')
    parser.add_argument('--model_name', '-M',  type=str, default='MFNet')
    parser.add_argument('--gpu',        '-G',  type=int, default=0)
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)

    checkpoint_model_file = os.path.join(model_dir, 'tmp.pth')
    final_model_file      = ' ....'

    print('| running %s demo on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()
