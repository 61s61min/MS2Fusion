
import os
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from tqdm import tqdm

from net import Net, S3F_Net


class ValDatasets:
    def __init__(self, val_path, test_size):
        self.test_size = test_size
        image_root = os.path.join(val_path, "RGB")
        gt_root = os.path.join(val_path, "GT")
        ti_root = os.path.join(val_path, "T")

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.tis = [os.path.join(ti_root, f) for f in os.listdir(ti_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.tis = sorted(self.tis)
        self.transform = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor()])
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt, h, w = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt).unsqueeze(0)
        ti = self.rgb_loader(self.tis[self.index])
        ti = self.tis_transform(ti).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, ti, name, h, w

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            h, w =  np.shape(img)[:2]
            return img.convert('L'), h, w

    def __len__(self):
        return self.size


def predict_val(model, val_data, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(val_data.size)):
            image, gt, ti, name, h, w = val_data.load_data()
            # gt = gt.cuda()
            image = image.cuda()
            ti = ti.cuda()

            res = model(image, ti)

            # 将预测结果转换为二值图像
            predict = torch.sigmoid(res)
            predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)

            # 将预测结果保存为图像
            predict = predict.data.cpu().numpy().squeeze()
            predict = cv2.resize(predict, dsize=(w, h))

            save_path = os.path.join(save_dir, name)
            # predict.save(save_path)
            cv2.imwrite(save_path, predict*255)        


if __name__ == "__main__":
    model_path = "your/path/to/weight.pth"
    test_data_path = "your/path/to/RGBT_dataset"
    result_save_path = "your/path/to/save/RGBT_dataset"

    val_loader = ValDatasets(test_data_path, 224)

    model = S3F_Net()
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    predict_val(model, val_loader, result_save_path)
