# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/9/16 20:24
import copy
import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
from methods.tool import tool, mpl_tool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实现fedavg算法
def fedavg(args, testset, part_data):
    path = tool.mk_path(args)
    writer_file = f"fedavg-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    trainset, aug_dataset = mpl_tool.get_train_set_aug(args)
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                     for i in range(args.client_num)]
    train_loader_aug = [DataLoader(Subset(aug_dataset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model']=args.model
    model = md.choose_model(options)
    model = model.to(device)
    for item in range(args.round_num):
        lr = tool.getLr(args, item)

        # 每轮随机选择部分客户端
        np.random.seed(item)
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        print(f"selected clients:{idx_users}")
        selected_params = []    #选中客户端的模型
        selected_data_num = []  #选中客户端的样本数量
        for k in range(args.clients_per_round):
            # 训练每个选到的客户端
            local = LocalTrain(args, train_loaders[idx_users[k]],lr)
            global_model = copy.deepcopy(model)
            param, loss = local.train(global_model)
            selected_params.append(tool.get_flat_params_from(param.parameters()))
            selected_data_num.append(train_loaders[idx_users[k]].sampler.num_samples)
            print(f"Client:{idx_users[k]} Loss:{loss}")

        # 每轮训练结束后，将该轮选取的客户端模型聚合，得到最新的全局模型
        global_param = tool.aggregate_avg(selected_params,selected_data_num)
        # tool.set_flat_params_to(model, global_param)
        tool.set_flat_params_custom(model,global_param,0.5)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedAvg Round {item} Accuracy on global test set: {global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data':global_acc}, item)






