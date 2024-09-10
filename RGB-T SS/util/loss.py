from util.lovasz_losses import lovasz_softmax
import numpy as np
import torch
import torch.nn as nn


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


class eeemodelLoss(nn.Module):

    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [1.681, 43.623, 41.695, 42.325, 38.371, 42.011, 6.873, 43.406, 40.634, 37.884, 37.325, 31.001,
             30.114])).float()

        self.class_weight = class_weight
        self.LovaszSoftmax = lovasz_softmax
        self.cross_entropy = nn.CrossEntropyLoss()

        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.BCEWithLogitsLoss()
        # self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.MSE_loss = nn.MSELoss()

    def CCM_Loss(self, sideout, targets):
        semantic_gt, binary_gt = targets
        aux_pred_rgb, aux_pred_t, Med_rgb, Med_thermal, umap_rgb, umap_t = sideout

        loss1 = self.semantic_loss(aux_pred_rgb, semantic_gt)
        loss2 = self.semantic_loss(aux_pred_t, semantic_gt)
        loss1_1 = self.semantic_loss(Med_rgb, semantic_gt)
        loss2_1 = self.semantic_loss(Med_thermal, semantic_gt)

        n, h, w = semantic_gt.shape
        labels_onehot = torch.zeros(n, 13, h, w)
        labels_onehot = labels_onehot.cuda()
        labels_onehot.scatter_(1, semantic_gt.view(n, 1, h, w), 1)
        labels_onehot = labels_onehot[:, 1:, :, :]  # Ignore unlabeled class-0, focusing on known classes.
        # print(labels_onehot.shape)

        kl_distance = nn.KLDivLoss(reduction='none')
        aux_pred_rgb_fore = aux_pred_rgb[:, 1:, :, :]  # Ignore unlabeled class-0, focusing on foreground classes.
        discrepancy_rgb = torch.sum(kl_distance(self.log_sm(aux_pred_rgb_fore), self.sm(labels_onehot)), dim=1)
        Diff_rgb = min_max_norm(discrepancy_rgb.unsqueeze(1))
        aux_pred_t_fore = aux_pred_t[:, 1:, :, :]  # Ignore unlabeled class-0, focusing on foreground classes.
        discrepancy_thermal = torch.sum(kl_distance(self.log_sm(aux_pred_t_fore), self.sm(labels_onehot)), dim=1)
        Diff_thermal = min_max_norm(discrepancy_thermal.unsqueeze(1))

        loss3 = self.MSE_loss(umap_rgb, Diff_rgb)
        loss4 = self.MSE_loss(umap_t, Diff_thermal)

        loss_CCM = (loss1+loss2+loss1_1+loss2_1+loss3+loss4) / 6.0
        return loss_CCM


    def forward(self, predict, targets):
        [sideouts1, sideouts2, sideouts3, sideouts4, sideouts5, output] = predict
        semantic_gt, binary_gt = targets
        [logits, supp] = output

        loss_CCM1 = self.CCM_Loss(sideouts1, targets)
        loss_CCM2 = self.CCM_Loss(sideouts2, targets)
        loss_CCM3 = self.CCM_Loss(sideouts3, targets)
        loss_CCM4 = self.CCM_Loss(sideouts4, targets)
        loss_CCM5 = self.CCM_Loss(sideouts5, targets)

        loss_CCM = (loss_CCM1+loss_CCM2+loss_CCM3+loss_CCM4+loss_CCM5) / 5.0

        loss_1 = self.semantic_loss(supp, semantic_gt)

        loss_2 = self.semantic_loss(logits, semantic_gt)
        
        loss_3 = self.LovaszSoftmax(logits, semantic_gt, weights=self.class_weight_semantic)
        loss_SEG = (loss_1 + loss_2 + loss_3) / 3.0

        loss = (loss_SEG + loss_CCM)/2.0

        return loss
