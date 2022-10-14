# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/14 0:08
import sys
import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
from methods.tool import tool
# 实现fed_mutual方法
# 本地模型和全局模型互学习
def fed_mutual(args, trainset, testset, part_data):
    writer_file = f"fed_mutual-{args.dataset}-clientNum{args.client_num}-dir{str(args.alpha).replace('.', '_')}-seed{args.seed}"
    writer = SummaryWriter(f"{sys.path[0]}/{args.data_path}/run_result/{writer_file}")
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=True)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model = md.choose_model(options)

    # 需要记录每个客户端中的私有模型
    client_private_params={}
    for item in range(args.round_num):
        print(f"---Round:{item}---")

        # 每轮随机选择部分客户端
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        selected_params = []
        for k in range(args.clients_per_round):
            private_model = model
            if idx_users[k] in client_private_params.keys():
                tool.set_flat_params_to(private_model,client_private_params[idx_users[k]])

            # 训练每个选到的客户端
            local = LocalTrain(args, train_loaders[idx_users[k]])
            private_model,meme = local.train_mul(private_model,model)

            selected_params.append(tool.get_flat_params_from(meme))
            client_private_params[idx_users[k]] = tool.get_flat_params_from(private_model)

        # 每轮训练结束后，将该轮选取的客户端模型聚合，得到最新的全局模型
        global_param = tool.aggregate_avg(selected_params)
        tool.set_flat_params_to(model, global_param)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedMul Round {item} Accuracy on global test set: {global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data':global_acc}, item)


