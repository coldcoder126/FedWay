# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/12 22:08

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# pytorch中state_dict()和named_parameters()的差别 https://blog.csdn.net/weixin_41712499/article/details/110198423

class LocalTrain(object):
    def __init__(self, args, train_loader):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.epoch = args.epoch

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []
        for epoch in range(self.epoch):
            per_epoch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                net.zero_grad()
                prediction = net(inputs)
                loss = self.loss_func(prediction, labels)
                per_epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            per_loss = sum(per_epoch_loss) / len(per_epoch_loss)
            epoch_loss.append(per_loss)
            print(f"Update Epoch : {epoch} Loss : {per_loss}")
        return net.parameters(), sum(epoch_loss) / len(epoch_loss)

    # 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.args.lr, momentum=0.5)

        model.train();
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.5
        beta = 0.5

        for e in range(self.epoch):
            # Training
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                model.zero_grad()

                output_local = model(inputs)
                output_meme = meme(inputs)
                # 蒸馏，互学习
                ce_local = CE_Loss(output_local, labels)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
                ce_meme = CE_Loss(output_meme, labels)
                kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))

                loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * ce_meme + (1 - beta) * kl_meme
                loss_local.backward()
                loss_meme.backward()

                optimizer.step()
                meme_optimizer.step()

        return model.parameters(), meme.parameters()
