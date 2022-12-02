# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/12/1 22:14
# 教师网络带有增强的fed_mu
import copy
import math
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalMutualAug
import src.models.model as md
from methods.tool import tool, mpl_tool


# 实现fed_mutual方法
# 本地模型和全局模型互学习
def fed_mutual(args, testset, part_data):
    path = tool.mk_path(args)
    writer_file = f"fed_mutual_aug-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    trainset, aug_dataset = mpl_tool.get_train_set_aug(args)
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=False)
                     for i in range(args.client_num)]
    train_loader_aug = [DataLoader(Subset(aug_dataset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=False)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model = md.choose_model(options)

    # 需要记录每个客户端中的私有模型
    client_private_params={}
    for item in range(args.round_num):
        # 每过10轮学习率变为之前的0.1倍
        factor = 10 ** math.floor(item / 10)
        lr = args.lr / factor
        print(f"---Round:{item},lr={lr} ---")

        # 每轮随机选择部分客户端
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        selected_params = []
        for k in range(args.clients_per_round):
            private_model = copy.deepcopy(model)
            if idx_users[k] in client_private_params.keys():
                print(f"client {idx_users[k]} in dict")
                tool.set_flat_params_to(private_model, copy.deepcopy(client_private_params[idx_users[k]]))

            # 训练每个选到的客户端
            local = LocalMutualAug(args, train_loaders[idx_users[k]], train_loader_aug[idx_users[k]], lr)
            private_param, meme_param = local.train_mul(private_model, model)

            selected_params.append(tool.get_flat_params_from(meme_param))
            client_private_params[idx_users[k]] = copy.deepcopy(tool.get_flat_params_from(private_param))

        # 每轮训练结束后，将该轮选取的客户端模型聚合，得到最新的全局模型
        global_param = tool.aggregate_avg(selected_params)
        tool.set_flat_params_to(model, global_param)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedMul Round {item} Accuracy on global test set: {global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data':global_acc}, item)




