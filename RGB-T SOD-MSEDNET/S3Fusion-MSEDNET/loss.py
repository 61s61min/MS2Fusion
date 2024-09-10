
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_bound(img, k_size):
    B, C, H, W = img.shape
    pad = int((k_size - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant', value=0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, k_size, 1).unfold(3, k_size, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion


class IOUBCELoss(nn.Module):
    def __init__(self):
        super(IOUBCELoss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, target_scale):
        b, _, _, _ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, target_scale):
            bce = self.nll_lose(inputs, targets)
            pred = torch.sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1 - IOU + bce)
        total_loss = sum(loss)
        return total_loss / b


class IOUBCEWithoutLogitLoss(nn.Module):
    def __init__(self):
        super(IOUBCEWithoutLogitLoss, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, target_scale):
        b, c, h, w = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, target_scale):
            bce = self.nll_lose(inputs, targets)

            inter = (inputs * targets).sum(dim=(1, 2))
            union = (inputs + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1 - IOU + bce)
        total_loss = sum(loss)
        return total_loss / b


class MulScaleBoundLoss(nn.Module):
    """
    x5_decoder   ->  torch.Size([1, 256, 40, 30])
    x4_decoder   ->  torch.Size([1, 256, 80, 60])
    x3_decoder   ->  torch.Size([1, 128, 160, 120])
    x2_decoder   ->  torch.Size([1, 64, 320, 240])
    out          ->  torch.Size([1, 9, 640, 480])
    """

    def __init__(self):
        super(MulScaleBoundLoss, self).__init__()
        self.IOUBCE = IOUBCELoss().cuda()
        self.IOUBCE_without_logit = IOUBCEWithoutLogitLoss().cuda()

    def forward(self, gt: torch.Tensor, out: [torch.Tensor]):
        # gts0 = F.interpolate(gt, size=(out[0].shape[2:]))
        # gts1 = F.interpolate(gt, size=(out[1].shape[2:]))
        # gts2 = F.interpolate(gt, size=(out[2].shape[2:]))

        # ground truth bound
        bound0 = tensor_bound(gt, 3).cuda()
        bound1 = tensor_bound(gt, 3).cuda()
        bound2 = tensor_bound(gt, 3).cuda()

        loss1 = self.IOUBCE(out[0], gt).cuda()
        loss2 = self.IOUBCE(out[1], gt).cuda()
        loss3 = self.IOUBCE(out[2], gt).cuda()

        # predict bound
        # out = torch.sigmoid(out)
        predict_bound0 = tensor_bound(torch.sigmoid(out[0]), 3).cuda()
        predict_bound1 = tensor_bound(torch.sigmoid(out[1]), 3).cuda()
        predict_bound2 = tensor_bound(torch.sigmoid(out[2]), 3).cuda()

        loss4 = self.IOUBCE_without_logit(predict_bound0, bound0).cuda()
        loss5 = self.IOUBCE_without_logit(predict_bound1, bound1).cuda()
        loss6 = self.IOUBCE_without_logit(predict_bound2, bound2).cuda()

        loss_sod = loss1 + loss2 + loss3
        loss_bound = loss4 + loss5 + loss6
        return loss_sod + loss_bound, bound0[0], predict_bound0[0]


if __name__ == '__main__':
    x5_decoder = torch.randn([256, 1, 40, 30])
    x4_decoder = torch.randn([256, 1, 80, 60])
    x3_decoder = torch.randn([128, 1, 160, 120])
    x2_decoder = torch.randn([64, 1, 320, 240])
    out = torch.randn([16, 1, 640, 480])

    pre = [x5_decoder, x4_decoder, x3_decoder, x2_decoder, out]
    gt = torch.randn([16, 1, 640, 480])
    test_loss = MulScaleBoundLoss()
    loss = test_loss(gt, pre)
    print(loss)
