# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/13 22:50
# 对参数的公共操作

import torch
import numpy as np


# 获取每个模型的参数(一维)
def get_flat_params_from(parameters):
    params = []
    for param in parameters:
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


# 更新模型中的参数
def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def global_test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    return acc
