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
            per_loss = sum(per_epoch_loss)/len(per_epoch_loss)
            epoch_loss.append(per_loss)
            print(f"Update Epoch : {epoch} Loss : {per_loss}")
        return net.parameters(), sum(epoch_loss)/len(epoch_loss)