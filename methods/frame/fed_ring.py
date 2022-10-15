# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/14 11:38
import copy
import sys
import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
from methods.tool import tool


# 实现fed_ring 首尾相连+蒸馏
def fed_ring(args, trainset, testset, part_data):
    # 1. 将所有要训练的客户端首尾相连
    path = tool.mk_path(args)
    writer_file = f"fed_ring-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=True)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model = md.choose_model(options)

    # 需要记录每个客户端中的私有模型
    client_private_model = {}
    client_private_params_temp = {}
    for item in range(args.round_num):
        print(f"---Round:{item}---")

        # 每轮随机选择所有客户端
        first_round_params = {}
        for k in range(args.client_num):
            if k % 5 == 0:
                print(f"training {k}th - {k + 5}th clients")

            # 如果是第一轮，所有客户端只训练一个自己的模型
            if item == 0:
                local = LocalTrain(args, train_loaders[k])
                param, loss = local.train(model)
                first_round_params[k] = tool.get_flat_params_from(param)
            # 如果不是第一轮，每次取相邻两个的模型互相学习
            else:
                local = LocalTrain(args, train_loaders[k])
                m1 = copy.deepcopy(client_private_model[k][0])
                m2 = copy.deepcopy(client_private_model[k][1])
                pre, nex = local.train_mul(m1, m2)
                client_private_params_temp[k] = (
                copy.deepcopy(tool.get_flat_params_from(pre)), copy.deepcopy(tool.get_flat_params_from(nex)))
        if item == 0:
            client_private_model = get_assigned_model_dic(args, first_round_params, first_epoch=True)
        else:
            client_private_model = get_assigned_model_dic(args, client_private_params_temp, first_epoch=False)

        # 每一轮训练完，测试全局模型在全局测试集上的表现(所有模型精度的平均值)
        all_acc = []
        for i in range(args.client_num):
            pre_idx = (i + args.client_num - 1) % args.client_num
            model_i = client_private_model[pre_idx][1]
            acc_i = tool.global_test(model_i, test_loader)
            all_acc.append(acc_i)
            print(f"model{i} acc : {acc_i}")
        avg_acc = sum(all_acc) / len(all_acc)

        print(f'Fed_ring Round {item} Accuracy on global test set: {avg_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data': avg_acc}, item)


# 根据第一轮模型列表，组装出一个分配好的字典
def get_assigned_model_dic(args, param_dic, first_epoch):
    dic = {}
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model1 = md.choose_model(options)
    model2 = md.choose_model(options)
    for i in range(args.client_num):
        pre_idx = (i + args.client_num - 1) % args.client_num
        nex_idx = (i + 1) % args.client_num
        if first_epoch:
            tool.set_flat_params_to(model1, param_dic[pre_idx])
            tool.set_flat_params_to(model2, param_dic[nex_idx])

        else:
            tool.set_flat_params_to(model1, param_dic[pre_idx][1])
            tool.set_flat_params_to(model2, param_dic[nex_idx][0])
        dic[i] = (copy.deepcopy(model1), copy.deepcopy(model2))
    return dic
