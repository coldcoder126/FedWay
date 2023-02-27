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
        fenzi = torch.diag(logits[size:])
        sim = torch.exp(fenzi/self.t)
        con = torch.sum(torch.exp(logits/self.t),dim=0) - torch.e
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

def MySupConLoss(representations,label,T):
    #温度参数T
    n = label.shape[0]  # batch

    #这步得到它的相似度矩阵
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    #这步得到它的label矩阵，相同label的位置为1
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

    #这步得到它的不同类的矩阵，不同类的位置为1
    mask_no_sim = torch.ones_like(mask) - mask

    #这步产生一个对角线全为0的，其他位置为1的矩阵
    mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n )

    #这步给相似度矩阵求exp,并且除以温度参数T
    similarity_matrix = torch.exp(similarity_matrix/T)

    #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
    similarity_matrix = similarity_matrix*mask_dui_jiao_0


    #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
    sim = mask*similarity_matrix


    #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
    no_sim = similarity_matrix - sim


    #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
    no_sim_sum = torch.sum(no_sim , dim=1)

    '''
    将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
    至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
    每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
    '''
    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum  = sim + no_sim_sum_expend
    loss = torch.div(sim , sim_sum)


    '''
    由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
    全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
    '''
    loss = mask_no_sim + loss + torch.eye(n, n )


    #接下来就是算一个批次中的loss了
    loss = -torch.log(loss)  #求-log
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))
    # loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n

    return loss

class SupConAnchorLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConAnchorLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, features, labels, anchors):
        # 每个经过encoder的128维feature和其他anchor进行对比学习
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0).to(device)
        labels = torch.cat([labels,labels]).detach().to(device)
        anchor_t = torch.stack(list(anchors.values())).detach() # 10 x 128
        logits = torch.matmul(anchor_t,contrast_feature.t()).t().to(device) # 10 x 128
        loss = self.loss_func(logits,labels)
        # 每个Label的值就是分子的
        return loss


class SupConNceLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConNceLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, features, labels, anchors):
        # 每个经过encoder的128维feature和其他anchor进行对比学习
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0).to(device)
        contrast_feature = features
        labels = torch.cat([labels,labels]).detach().to(device)
        anchor_list = [anchors[i] for i in range(len(anchors))]
        anchor_t = torch.stack(anchor_list).detach() # 10x10
        logits = torch.matmul(contrast_feature, anchor_t.t()).to(device) # 128 x 10
        row_idx = torch.range(0,len(labels)-1).long()
        logits_up = logits[row_idx,labels]
        fenzi_2 = torch.exp(logits_up/self.temperature)
        fenmu = torch.exp(logits/self.temperature).sum(dim=1)
        loss = -torch.log(fenzi_2/fenmu)
        loss = loss.mean()
        # 每个Label的值就是分子的
        return loss

class SupConLoss2(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss2, self).__init__()
        self.temperature = temperature
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        # 每个经过encoder的128维feature和其他anchor进行对比学习
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0).to(device)
        size = len(labels)
        contrast_feature = features
        labels = torch.cat([labels,labels]).detach().to(device)
        idx1 = torch.arange(size * 2)
        idx_col = idx1.view(1,-1)
        idx_row = (labels==labels[idx_col])


        anchor_list = [anchors[i] for i in range(len(anchors))]
        anchor_t = torch.stack(anchor_list).detach() # 10x10
        logits = torch.matmul(contrast_feature, anchor_t.t()).to(device) # 128 x 10
        row_idx = torch.range(0,len(labels)-1).long()
        logits_up = logits[row_idx,labels]
        fenzi_2 = torch.exp(logits_up/self.temperature)
        fenmu = torch.exp(logits/self.temperature).sum(dim=1)
        loss = -torch.log(fenzi_2/fenmu)
        loss = loss.mean()
        # 每个Label的值就是分子的
        return loss





Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
KL_Loss = nn.KLDivLoss(reduction='batchmean')
class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()

    def forward(self,features,labels,anchor_map):
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature =features
        # anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.T)
        # 计算每个类中的表示和对应anchor之间的余弦相似 1-cos
        anchor_idx = torch.cat([labels, labels])
        anchor_list = [ anchor_map[i] for i in range(len(anchor_map))]
        anchors = torch.stack(anchor_list)
        anchors_target = anchors[anchor_idx].detach()

        cos_sim_func = nn.CosineSimilarity(dim=1, eps=1e-6)

        cos_sim_loss = 1 - cos_sim_func(contrast_feature, anchors_target)
        # cos_sim_loss = KL_Loss(LogSoftmax(features)/0.1, Softmax(anchors_target.detach()/0.1))

        s = torch.square(cos_sim_loss)
        loss = torch.exp(s) - 1
        return loss.sum()
