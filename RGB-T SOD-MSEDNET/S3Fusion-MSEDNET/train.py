

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from config import opt
from net import Net, S3F_Net
from dataset import get_loader, test_dataset
from loss import MulScaleBoundLoss
from utils import adjust_lr

writer = SummaryWriter(os.path.join(opt.save_path, 'summary/finally'), flush_secs=30)

# logging config
if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)
logging.basicConfig(filename=os.path.join(opt.save_path, "log", datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".log"),
                    format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
                    level=logging.INFO, filemode="a", datefmt="%Y-%m-%d %I:%M:%S %p")

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
logging.info("use gpu: {}".format(opt.gpu_id))

# build the model
logging.info("\n<=============build model=============>")
model = S3F_Net()
if opt.load is not None:
    model.load_state_dict(torch.load(opt.load),strict=False)
    print('load model from ', opt.load)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
logging.info(model)

# load dataset
logging.info("\n<=============load dataset=============>")
image_root = os.path.join(opt.train_root, "RGB/")
ti_root = os.path.join(opt.train_root, "T/")
gt_root = os.path.join(opt.train_root, "GT/")
val_image_root = os.path.join(opt.val_root, "RGB/")
val_ti_root = os.path.join(opt.val_root, "T/")
val_gt_root = os.path.join(opt.val_root, "GT/")

train_loader = get_loader(image_root, gt_root, ti_root, batchsize=opt.batch_size, trainsize=opt.train_size)
test_loader = test_dataset(val_image_root, val_gt_root, val_ti_root, opt.train_size)
total_step = len(train_loader)

logging.info(
    "\nepoch:{};lr:{};batch_size:{};train_size:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}".format(
        opt.epoch, opt.lr, opt.batch_size, opt.train_size, opt.clip, opt.decay_rate, opt.load, opt.save_path,
        opt.decay_epoch))

step = 0
best_mae = 1
best_epoch = 0
last_loss = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step, last_loss
    msb_loss = MulScaleBoundLoss()
    model.train()
    """
    x5_decoder   ->  torch.Size([1, 256, 40, 30])      
    x4_decoder   ->  torch.Size([1, 256, 80, 60])
    x3_decoder   ->  torch.Size([1, 128, 160, 120])
    x2_decoder   ->  torch.Size([1, 64, 320, 240])
    out          ->  torch.Size([1, 9, 640, 480])
    """
    loss_all = 0
    epoch_step = 0
    last_loss = loss_all
    try:
        for i, (images, gts, tis) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            rgb = images.cuda()
            ir = tis.cuda()
            gt = gts.cuda()
            # print("gt shape:", gt.shape)

            out = model(rgb, ir)
            loss, bound_i, pre_bound_i = msb_loss(gt, out)
            loss.backward()
            optimizer.step()
            step = step + 1
            epoch_step = epoch_step + 1
            loss_all = loss.item() + loss_all

            if i % 10 == 0 or i == total_step or i == 1:
                print("{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_all : {:.4f}".
                      format(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), epoch, opt.epoch, i, total_step,
                             loss.item(), loss_all))
                logging.info("{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_all : {:.4f}".
                             format(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), epoch, opt.epoch, i, total_step, loss.item(), loss_all))

                writer.add_scalar('Loss', loss, global_step=step)

                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/Ground_truth', grid_image, step)
                grid_image = make_grid(bound_i.clone().cpu().data, 1, normalize=True)
                writer.add_image('train/bound', grid_image, step)

                res = out[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/out', torch.tensor(res), step, dataformats='HW')

                res = pre_bound_i.clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/bound', torch.tensor(res), step, dataformats='HW')

            loss_all /= epoch_step
            writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        torch.save(model.state_dict(), os.path.join(save_path, "models/finally", 'weight_Last.pth'))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, "models/finally", 'weight_Inter.pth'))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in tqdm(range(test_loader.size)):
            image, gt, ti, name = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            ti = ti.cuda()

            res = model(image, ti)
            res = torch.sigmoid(res)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = torch.sum(torch.abs(res - gt)) * 1.0 / (torch.numel(gt))
            # print(mae_train)
            mae_sum = mae_train.item() + mae_sum

        # print(test_loader.size)
        mae = mae_sum / test_loader.size
        # print(test_loader.size)
        writer.add_scalar('MAE', torch.as_tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, "models/finally", 'weight.pth'))
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, opt.save_path)
        test(test_loader, model, epoch, opt.save_path)
