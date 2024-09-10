# coding:utf-8
import logging
import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.MF_dataset import MF_dataset
from util.util import calculate_accuracy, calculate_result
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from model import MFNet, SegNet
from tqdm import tqdm
from util.loss import eeemodelLoss
logger = logging.getLogger(__name__)
# config
n_class   = "num_class"
data_dir  = '....'
model_dir = '....'
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0), 
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]
lr_start  = 0.01
lr_decay  = 0.95


def train(epo, model, train_loader, optimizer):

    lr_this_epo = lr_start * lr_decay**(epo-1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo

    loss_avg = 0.
    acc_avg  = 0.
    start_t = t = time.time()
    model.train()
    pbar = enumerate(train_loader)
    pbar=tqdm(pbar,total=len(train_loader), position = 0, leave= True)
    logger.info(('\n' + '%10s' *3)%("epo", "loss" , "acc" ))
    # SemanticRT loss
    # train_criterion = eeemodelLoss().to(args.device)


    for it, (images, labels, names) in pbar:
        images = Variable(images).to(args.device)
        labels = Variable(labels).to(args.device)
        # if args.gpu >= 0:
        #     images = images.cuda(args.gpu)
        #     labels = labels.cuda(args.gpu)
        # print(torch.min(labels))
        # print(torch.max(labels))
        optimizer.zero_grad()
        logits = model(images)
        # logits =torch.sigmoid(logits)
        # logits = torch.randn(4,13,640,640).to(args.device)
        # labels = torch.randint(0,13,(4,640,640)).to(args.device)
        loss = F.cross_entropy(logits, labels)
        # print(loss)
        # loss = train_criterion(logits, labels)

        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(logits, labels)
        loss_avg += float(loss)
        acc_avg  += float(acc)

        cur_t = time.time()
        if cur_t-t > 5:
            # print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
            #     % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            s = ('%10s'*2+'%10.4g'*2+'%10s')%(epo, args.epoch_max,float(loss), float(acc), images.shape)
            pbar.set_description(s)
            # t += 5

    content = '\n | epo:%s/%s lr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
            % (epo, args.epoch_max, lr_this_epo, loss_avg/train_loader.n_iter, acc_avg/train_loader.n_iter)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)


def validation(epo, model, val_loader):

    loss_avg = 0.
    acc_avg  = 0.
    start_t = time.time()
    model.eval()
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader),position = 0, leave= True)
    logger.info(('\n' + '%10s' * 3) % ("epo", "loss", "acc"))
    with torch.no_grad():
        for it, (images, labels, names) in pbar:
            images = Variable(images).to(args.device)
            labels = Variable(labels).to(args.device)
            # if args.gpu >= 0:
            #     images = images.cuda(args.gpu)
            #     labels = labels.cuda(args.gpu)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            acc = calculate_accuracy(logits, labels)
            loss_avg += float(loss)
            acc_avg  += float(acc)

            cur_t = time.time()
            # print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
            #         % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss), float(acc)))
            s = ('%10s' * 2 + '%10.4g' * 2) % (epo, args.epoch_max, float(loss), float(acc))
            pbar.set_description(s)
    content = '\n | val_loss_avg:%.4f val_acc_avg:%.4f\n' \
            % (loss_avg/val_loader.n_iter, acc_avg/val_loader.n_iter)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)


def test_fuse(model, test_loader):
    cf = np.zeros((n_class, n_class))

    test_loader.n_iter = len(test_loader)

    loss_avg = 0.
    acc_avg = 0.
    model.eval()
    pbar = enumerate(test_loader)
    pbar = tqdm(pbar, total=len(test_loader),position = 0, leave= True)
    logger.info(('\n' + '%10s' * 3) % ("epo", "loss", "acc"))
    with torch.no_grad():
        for it, (images, labels, names) in pbar:
            images = Variable(images)
            labels = Variable(labels)
            # if args.gpu >= 0:
            images = images.to(args.device)
            labels = labels.to(args.device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            acc = calculate_accuracy(logits, labels)
            loss_avg += float(loss)
            acc_avg += float(acc)

            s = ( '%10.4g' * 2) % (  float(loss), float(acc))
            pbar.set_description(s)

            predictions = logits.argmax(1)
            for gtcid in range(n_class):
                for pcid in range(n_class):
                    gt_mask = labels == gtcid
                    pred_mask = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())

    overall_acc, acc, IoU = calculate_result(cf)

    print('| overall accuracy:', overall_acc)
    print('| accuracy of each class:', acc)
    print('| class accuracy avg:', acc.mean())
    print('| IoU:', IoU)
    print('| class IoU avg:', IoU.mean())

    return IoU.mean()


def main():

    model = eval(args.model_name)(n_class=n_class).to(args.device)
    # if args.device >= 0: model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.9, weight_decay=0.0005) 
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file, map_location={'cuda:0'}))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))
        print('done!')

    train_dataset = MF_dataset(data_dir, 'train', have_label=True)
    val_dataset  = MF_dataset(data_dir, 'val', have_label=True)
    test_dataset = MF_dataset(data_dir, 'test', have_label=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = True
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter   = len(val_loader)
    best_IoU =0
    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        print('\n| epo #%s begin...' % epo)

        train(epo, model, train_loader, optimizer)
        validation(epo, model, val_loader)
        IoU = test_fuse(model, test_loader)
        if best_IoU < IoU:
            best_IoU = IoU
            torch.save(model.state_dict(), '......pth')
            torch.save(optimizer.state_dict(), '.......optim')
        # save check point model
        print('| saving check point model file... ', end='')
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)
        print('done!')

    os.rename(checkpoint_model_file, final_model_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MFNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='MFNet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=4)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=100)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--device',      '-G',  default='cuda')
    parser.add_argument('--num_workers', '-j',  type=int, default=24)
    args = parser.parse_args()

    model_dir = os.path.join(model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_model_file = os.path.join(model_dir, 'tmp.pth')
    checkpoint_optim_file = os.path.join(model_dir, 'tmp.optim')
    final_model_file      = os.path.join(model_dir, 'final.pth')
    log_file              = os.path.join(model_dir, 'log.txt')

    # print('| training %s on GPU #%d with pytorch' % (args.model_name, args.device))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir)

    main()
