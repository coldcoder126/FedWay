# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/12/1 22:14
# fed_mul_aug +　本地模型每次训练时都是不同的增强
import copy
import math

import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalMutualAug, LocalMutualAug2, LocalMutualAug2_1
import src.models.model as md
from methods.tool import tool, mpl_tool, con_tool


# 实现fed_mutual方法
# 本地模型和全局模型互学习
def fed_mutual(args, part_data):
    path = tool.mk_path(args)
    writer_file = f"fed_mutual_aug-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    train_set, test_set = tool.get_data_set(args, False)
    ori_train_set = [Subset(train_set, part_data.client_dict[i]) for i in range(args.client_num)]



    # trainset, aug_dataset = mpl_tool.get_train_set_aug(args)
    # 所有已经分好组的训练集和测试集
    # train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    #                  for i in range(args.client_num)]
    # train_loader_aug = [DataLoader(Subset(aug_dataset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    #                  for i in range(args.client_num)]
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = 'ccnn'
    model = md.choose_model(options)

    client_options = md.generate_options(args.dataset, args.model)
    client_options['model'] = args.model
    client_model = md.choose_model(client_options)



    lr = args.lr
    # 需要记录每个客户端中的私有模型
    client_models = {idx: copy.deepcopy(client_model) for idx in range(args.client_num)}
    for i in range(1,args.client_num+1):
        client_models[i].load_state_dict(torch.load(tool.get_model_path(args,i)))


    for item in range(1,args.round_num+1):
        # 每过10轮学习率变为之前的0.1倍
        lr = tool.getLr(lr, item)
        # 每轮随机选择部分客户端
        np.random.seed(item)
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        # print(f"selected clients:{idx_users}")
        selected_params = []
        selected_data_num = []  # 选中客户端的样本数量
        for k in idx_users:
            global_model = copy.deepcopy(model)
            # 训练每个选到的客户端
            local = LocalMutualAug2(args, ori_train_set[k], lr)
            local.train_mul(client_models[k], global_model)
            # client_models[k] = con_tool.net_avg([k],client_models,copy.deepcopy(model),[1])

            selected_params.append(tool.get_flat_params_from(global_model.parameters()))
            selected_data_num.append(ori_train_set[k].indices.size)

        # 每轮训练结束后，将该轮选取的客户端模型聚合，得到最新的全局模型
        global_param = tool.aggregate_avg(selected_params, selected_data_num)
        # tool.set_flat_params_to(model, global_param)
        tool.set_flat_params_custom(model, global_param, 0.5)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedMul Round {item} lr={lr} Accuracy on global test set: {global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data':global_acc}, item)




