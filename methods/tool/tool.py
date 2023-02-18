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

def set_flat_params_custom(model, flat_params,percent):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()) * percent + param.data * (1-percent) )
        prev_ind += flat_size

# 更新模型的指定层参数
def set_layer_to_model(args,layer_tensor, model):
    layers = list(model.parameters())[-args.ln:]
    prev_ind = 0
    for layer in layers:
        if len(layer.shape)==2:
            layer_size = layer.shape[0]*layer.shape[1]
            layer.data.copy_(layer_tensor[prev_ind:layer_size + prev_ind].view(layer.shape[0],-1))
        else:
            layer_size = layer.size()[0]
            layer.data.copy_(layer_tensor[prev_ind:layer_size+prev_ind])
        prev_ind += layer_size


# 将net1的最高p层替换为net2中的最高p层
def replace_layers(args, net1, net2):

    layers1 = list(net1.parameters())[-args.ln:]
    layers2 = list(net2.parameters())[-args.ln:]
    for i in range(len(layers1)):
        layers1[i].data = layers2[i].data
    print("done")





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


def getLr(args, cur_round):
    factor = 2 ** math.floor(cur_round / 10)
    lr = args.lr / factor
    if lr < 0.001:
        lr = 0.001
    print(f"---Round:{cur_round},lr={lr} ---")
    return lr


# 对模型进行聚合
def aggregate_avg(flat_params,selected_data_num):
    """Aggregate local solutions and output new global parameter

    Args:
        flat_params: a generator or (list) with element (num_sample, local_solution)

    Returns:
        flat global model parameter
    """

    averaged_solution = torch.zeros_like(flat_params[0])
    # averaged_solution = np.zeros(self.latest_model.shape)

    # 简单平均
    # num = 0
    # for local_solution in flat_params:
    #     num += 1
    #     averaged_solution += local_solution
    # averaged_solution /= num

    # 按照模型中的数据量平均
    num = 0
    for i in selected_data_num:
        num += i
    for j in range(len(flat_params)):
        averaged_solution += flat_params[j] * (selected_data_num[j]/num)

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
    path = f"{sys.path[0]}/{args.data_path}/run_result/{args.begin_time}-{args.desc}/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path
