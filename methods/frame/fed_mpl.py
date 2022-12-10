# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/29 1:44
import copy
import math
import sys
import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain, LocalMPL
import src.models.model as md
from methods.tool import tool, mpl_tool

# MPL需要数据增强，因此使用的数据和其他的数据要进行不同的transform处理
def fed_mpl(args,testset, part_data):
    # 所有处理过的有标签数据集和无标签数据集
    labeled_dataset, unlabeled_dataset = mpl_tool.get_train_set(args)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    path = tool.mk_path(args)
    writer_file = f"fed_MPL-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model = md.choose_model(options)

    # 需要记录每个客户端中的私有模型
    client_private_models = {}
    for item in range(args.round_num):
        # 每过10轮学习率变为之前的0.1倍
        lr = tool.getLr(args, item)

        # 每轮随机选择部分客户端
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        selected_params = []
        for k in range(args.clients_per_round):
            l_loader,u_loader = mpl_tool.l_u_split(args,labeled_dataset,unlabeled_dataset,part_data,idx_users[k])
            #
            teacher_model = copy.deepcopy(model)
            private_model = copy.deepcopy(model)
            if idx_users[k] in client_private_models.keys():
                print(f"client {idx_users[k]} in dict")
                private_model = client_private_models[idx_users[k]]

            local = LocalMPL(args, l_loader, u_loader, lr)
            local.train_MPL(teacher_model, private_model,item)

            x = teacher_model.parameters()
            selected_params.append(tool.get_flat_params_from(x))
            client_private_models[idx_users[k]] = copy.deepcopy(private_model)

        # 每轮训练结束后，将该轮选取的客户端模型聚合，得到最新的全局模型
        global_param = tool.aggregate_avg(selected_params)
        tool.set_flat_params_to(model, global_param)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedMpl Round {item} Accuracy on global test set: {global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data': global_acc}, item)

