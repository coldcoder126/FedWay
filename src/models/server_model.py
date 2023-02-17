# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/4 12:13
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightNet(nn.Module):
    '''
    client_num: 客户端数量
    param_shape: 一维参数的长度

    hidden: 融合过后的网络层参数
    '''
    def __init__(self,all_client_num,param_shape,selected_client_num):
        super(WeightNet,self).__init__()
        self.w1 = nn.Parameter(torch.randn(selected_client_num+1, param_shape))
        # self.conv1 = nn.Conv1d(selected_client_num+1, 2*(selected_client_num+1), 1)
        # self.conv2 = nn.Conv1d(2*(selected_client_num+1), 1, 1)
        self.fc = nn.Linear(param_shape,all_client_num)
        # self.fc.weight.data.fill_(1)
        torch.nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        self.fc.bias.data.fill_(0.01)
        # freeze(self.fc)

    def forward(self,Models, Clients):
        out = Models * self.w1
        hidden = out.sum(dim=0).unsqueeze(dim=1).t()

        # out = self.conv1(Models)
        # hidden = self.conv2(out)
        out = self.fc(hidden)
        out = torch.mm(out, Clients.float().t())
        return (hidden,out)

class RNN(nn.Module):
    def __init__(self,input_shape,hidden_shape,all_client_num):
        super(RNN,self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_shape))
        self.w2 = nn.Parameter(torch.randn(hidden_shape))
        self.ln = nn.LayerNorm(hidden_shape)
        self.fc = nn.Linear(hidden_shape, all_client_num)
        freeze(self.fc)


    def forward(self,input,hidden,client_onehot):
        hx = input * self.w1 + hidden*self.w2
        hx = F.relu(hx)
        hx = self.ln(hx)
        out = self.fc(hx)
        out = torch.mm(out.unsqueeze(dim=1).t(), client_onehot.float().t())
        return (hx,out)

def freeze(layer):
    for param in layer.parameters():
        param.requires_grad = False