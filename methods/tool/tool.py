# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/13 22:50
# 对参数的公共操作
import math
import os
import sys

import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    model = model.to(device)
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    return acc


# 对模型进行聚合
def aggregate_avg(flat_params):
    """Aggregate local solutions and output new global parameter

    Args:
        flat_params: a generator or (list) with element (num_sample, local_solution)

    Returns:
        flat global model parameter
    """

    averaged_solution = torch.zeros_like(flat_params[0])
    # averaged_solution = np.zeros(self.latest_model.shape)

    # 简单平均
    num = 0
    for local_solution in flat_params:
        num += 1
        averaged_solution += local_solution
    averaged_solution /= num

    # for num_sample, local_solution in flat_params:
    #     averaged_solution += num_sample * local_solution
    # averaged_solution /= self.all_train_data_num

    # averaged_solution = from_numpy(averaged_solution, self.gpu)
    return averaged_solution.detach()

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def mk_path(args):
    path = f"{sys.path[0]}/{args.data_path}/run_result/{args.begin_time}/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path
