# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/14 14:36
import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrastive_loss_batch(nn.Module):
    '''简单计算一个批次内的NCELoss'''
    def __init__(self,t):
        super(Contrastive_loss_batch, self).__init__()
        self.t = t

    def forward(self,input,label):
        size = int(len(label) / 2)
        logits = torch.matmul(input,input[:size].t())

        sim = torch.exp(torch.diag(logits[size:])/self.t)
        con = torch.sum(torch.exp(logits/self.t),dim=0)-1

        out = -torch.sum(torch.log(sim/con))/len(label)
        return out

# 有监督损失的Loss
class sup_con_loss(nn.Module):
    def __init__(self,):
        super(sup_con_loss, self).__init__()

    def forward(self,input,label,anchor):
        # 1.计算相同标签的Loss

        # 2.计算该数据和对应anchor之间的距离

        return 0

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # 将feature 变为 [batch_size, n_views, ...]的形状
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 初始化mask 为 batch_size X batch_size大小
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # 将features的n个视图拆分并合并为二维
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits 锚和对比特征计算
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability 找每一行最大的值，是自己和自己相乘
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask 复制 m * n 个mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logit_mask 是对角线为0，其余为1
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()

    def forward(self,features,labels,anchor_map):
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.T)
        # 计算每个类中的表示和对应anchor之间的余弦相似 1-cos
        labels_anchor = torch.cat([labels, labels])
        keys = [labels_anchor[i].item() for i in range(len(labels_anchor))]
        anchor = torch.stack([anchor_map[i] for i in keys]).detach()
        # pd_func = nn.PairwiseDistance(p=2)
        # pd_loss = pd_func(contrast_feature,anchor)
        cos_sim_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim_loss = -torch.log(0.5 *  cos_sim_func(contrast_feature, anchor) + 0.5)
        # cos_sim_loss = 1 - cos_sim_func(contrast_feature, anchor)
        cos_sim_loss = cos_sim_loss.sum()
        return cos_sim_loss
