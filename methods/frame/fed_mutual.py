# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/14 0:08
import copy
import time

import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
from methods.tool import tool
# 实现fed_mutual方法
# 本地模型和全局模型互学习
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def fed_mutual(args, trainset, testset, part_data):
    path = tool.mk_path(args)
    writer_file = f"fedMul-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model = md.choose_model(options).to(device)

    lr = args.lr
    # 需要记录每个客户端中的私有模型
    client_models = {idx: copy.deepcopy(model) for idx in range(args.client_num)}
    for item in range(1, args.round_num+1):
        lr = tool.getLr(lr, item)
        # 每轮随机选择部分客户端
        np.random.seed(item)
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        selected_params = []
        selected_data_num = []  #选中客户端的样本数量
        for k in idx_users:
            global_model = copy.deepcopy(model)
            # 训练每个选到的客户端
            local = LocalTrain(args, train_loaders[k],  lr)
            local.train_mul(client_models[k], global_model)
            # client_models[k] = con_tool.net_avg([k],client_models,copy.deepcopy(model),[1])

            selected_params.append(tool.get_flat_params_from(global_model.parameters()))
            selected_data_num.append(train_loaders[k].dataset.indices.size)

        # 每轮训练结束后，将该轮选取的客户端模型聚合，得到最新的全局模型
        global_param = tool.aggregate_avg(selected_params,selected_data_num)
        tool.set_flat_params_to(model, global_param)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedMul Round:{item} lr:{lr} clients:{idx_users} global_acc:{global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data':global_acc}, item)


