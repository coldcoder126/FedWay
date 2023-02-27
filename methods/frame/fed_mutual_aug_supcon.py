# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/12/1 22:14
# 教师网络带有增强的fed_mu
import copy
import math

import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader

from methods.client.client_mul import ClientMul
from methods.client.local_train import LocalMutualAug, LocalMutualAug2, LocalMutualAug2_1
import src.models.model as md
from methods.tool import tool, mpl_tool, con_tool


# 实现fed_mutual方法
# 本地模型和全局模型互学习
from src.models.resnet_big import LinearClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def fed_mutual(args, testset, part_data):
    path = tool.mk_path(args)
    writer_file = f"fed_mutual_aug-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    trainset, aug_dataset = mpl_tool.get_train_set_aug(args)
    # 所有已经分好组的训练集和测试集
    train_loaders = con_tool.set_loader(args,part_data)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # 选择model
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model = md.choose_model(options)
    head = copy.deepcopy(model.fc)
    model.fc = nn.Identity()
    server_model = md.LocalModel(model, head)

    # server_encoder = md.choose_encoder(args.model,128)
    # server_classifier = LinearClassifier()
    # server_model = md.LocalModel(server_encoder,server_classifier)

    client_model_local = {idx: copy.deepcopy(server_model) for idx in range(args.client_num)}
    client_model_global = {idx: copy.deepcopy(server_model) for idx in range(args.client_num)}

    for item in range(args.round_num):
        # 每过10轮学习率变为之前的0.1倍
        client_data_num = []
        lr = tool.getLr(args, item)
        # 每轮随机选择部分客户端
        np.random.seed(item)
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        # idx_users=[8]
        print(f"selected clients:{idx_users}")
        for k in idx_users:
            client_data_num.append(train_loaders[k].sampler.num_samples)
            global_model = copy.deepcopy(server_model)
            client = ClientMul(args,train_loaders[k],lr,client_model_local[k],global_model)
            client_model_global[k] = client.train_encoder()

        con_tool.net_avg(idx_users,client_model_global,server_model,client_data_num)
        global_acc = tool.global_test(model, test_loader)
        print(f'FedMul Round {item} Accuracy on global test set: {global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data':global_acc}, item)




