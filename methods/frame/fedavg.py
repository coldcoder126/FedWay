# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/9/16 20:24
import sys
import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
from methods.tool import tool


# 实现fedavg算法
def fedavg(args, trainset, testset, part_data):
    writer_file = f"{args.dataset}-clientNum{args.client_num}-dir{str(args.alpha).replace('.','_')}-seed{args.seed}"
    writer = SummaryWriter(f"{sys.path[0]}/{args.data_path}/run_result/{writer_file}")
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=True)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model']=args.model
    model = md.choose_model(options)
    global_param = None
    for item in range(args.round_num):
        print(f"---Round:{item}---")

        # 每轮随机选择部分客户端
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        selected_params = []
        for k in range(args.clients_per_round):
            # 训练每个选到的客户端
            local = LocalTrain(args, train_loaders[idx_users[k]])
            param, loss = local.train(model)
            selected_params.append(tool.get_flat_params_from(param))
            print(f"Client:{idx_users[k]} Loss:{loss}")

        # 每轮训练结束后，将该轮选取的客户端模型聚合，得到最新的全局模型
        global_param = aggregate_avg(selected_params)
        tool.set_flat_params_to(model, global_param)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedAvg Round Accuracy on global test set: {global_acc}%')
        writer.add_scalars('Loss/train_wd',
                           {'All Data':global_acc}, item)


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



