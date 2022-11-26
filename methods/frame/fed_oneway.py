# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/26 14:37

import copy
import math
import sys
import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
from methods.tool import tool

# 和fed_ring不同之处是保留一个自身的模型，将另一个模型顺时针发送

def fed_oneway(args, trainset, testset, part_data):
    path = tool.mk_path(args)
    writer_file = f"fed_oneway-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=True)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model = md.choose_model(options)

    private_models = {}
    for item in range(args.round_num):
        # 每过10轮学习率变为之前的0.1倍
        factor = 10 ** math.floor(item / 10)
        lr = args.lr / factor
        print(f"---Round:{item},lr={lr} ---")

        # 每轮选择所有客户端
        for k in range(args.client_num):
            if k % 5 == 0:
                print(f"training {k}th - {k + 5}th clients")
            # 如果是第一轮，所有客户端先训练一个自己的模型
            if item == 0:
                init_model = copy.deepcopy(model)
                local = LocalTrain(args, train_loaders[k],lr)
                param, loss = local.train(init_model)
                private_models[k] = copy.deepcopy(init_model)
            else:
                pre_idx = (k + args.client_num - 1) % args.client_num
                local = LocalTrain(args, train_loaders[k],lr)
                m1 = copy.deepcopy(private_models[k])  # 本地
                m2 = copy.deepcopy(private_models[pre_idx])  # 前一个
                pre, nex = local.train_mul(m1, m2)
                # 训练结束后只保存本地模型
                private_models[k] = m1
        all_acc = []
        for i in range(args.client_num):
            model_i = private_models[i]
            acc_i = tool.global_test(model_i, test_loader)
            all_acc.append(acc_i)
            print(f"model{i} acc : {acc_i}")
        avg_acc = sum(all_acc) / len(all_acc)
        print(f'Fed_oneway Round {item} Accuracy on global test set: {avg_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data': avg_acc}, item)


