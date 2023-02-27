# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/24 13:48

# 假设本地有一个已经训练好的模型
import copy
import math

import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalMutualAugCon, LocalMutualAug, LocalMutualAugFix
import src.models.model as md
from methods.tool import tool, mpl_tool, con_tool
from src.optimizer.loss_con import MySupConLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实现fed_mutual方法
# 本地模型和全局模型互学习
def fed_mutual(args, testset, part_data):
    path = tool.mk_path(args)
    writer_file = f"fed_mutual_aug-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    trainset, aug_dataset = mpl_tool.get_train_set_aug(args)
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
                     for i in range(args.client_num)]
    train_loader_aug = [DataLoader(Subset(aug_dataset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model_ = md.choose_model(options)
    head = copy.deepcopy(model_.fc)
    model_.fc = nn.Identity()
    model = md.LocalModel(model_, head).to(device)
    lr = args.lr
    # 需要记录每个客户端中的私有模型
    client_models = {idx: copy.deepcopy(model) for idx in range(args.client_num)}
    for i in range(args.client_num):
        client_models[i].load_state_dict(torch.load(tool.get_model_path(args,i)))



    # 在开始训练之前每个本地都有一个训练好的模型
    # for k in range(args.client_num):
    #     client_models
        # client_model = client_models[k]
        # optimizer = torch.optim.SGD(client_model.parameters(), lr=0.04, weight_decay=1e-3, momentum=0.9)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        # loss_func = nn.CrossEntropyLoss()
        # for e in range(2):
        #     # epoch_loss_ce = []
        #     # epoch_loss_con = []
        #     for batch_idx, (inputs, labels) in enumerate(train_loaders[k]):
        #
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         optimizer.zero_grad()
        #         # if torch.unique(labels).shape[0] > 1:
        #         #     reps = client_model.base(inputs)
        #         #     prediction = client_model.head(reps)
        #         #     con_loss = MySupConLoss(reps,labels,0.5)
        #         #     # epoch_loss_con.append(con_loss.item())
        #         #
        #         # else:
        #         prediction = client_model(inputs)
        #         ce_loss = loss_func(prediction, labels)
        #         loss = ce_loss
        #         # if torch.unique(labels).shape[0] > 1:
        #         #     loss += con_loss
        #
        #         # epoch_loss_ce.append(ce_loss.item())
        #
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1e5)
        #         optimizer.step()
        #     model_path = 'saved_models/client'+str(k)+'.pth'
        #     torch.save(client_model.state_dict(), model_path)
        #     scheduler.step()
            # print(f"client {k} epoch{e} ce_loss:{sum(epoch_loss_ce)/len(epoch_loss_ce)} ")
        # print(f"client {k} done")
    for item in range(args.round_num):
        # 每过10轮学习率变为之前的0.1倍
        lr = tool.getLr(lr, item)
        # 每轮随机选择部分客户端
        np.random.seed(item)
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        # idx_users = [14,1]
        print(f"selected clients:{idx_users}")
        selected_params = []
        selected_data_num = []  # 选中客户端的样本数量
        for k in idx_users:
            global_model = copy.deepcopy(model)
            # 训练每个选到的客户端
            local = LocalMutualAugFix(args, train_loaders[k], train_loader_aug[k], lr)
            local.train_mul(client_models[k], global_model)
            # client_models[k] = con_tool.net_avg([k],client_models,copy.deepcopy(model),[1])

            selected_params.append(tool.get_flat_params_from(global_model.parameters()))
            selected_data_num.append(train_loaders[k].dataset.indices.size)

        # 每轮训练结束后，将该轮选取的客户端模型聚合，得到最新的全局模型
        global_param = tool.aggregate_avg(selected_params, selected_data_num)
        # tool.set_flat_params_to(model, global_param)
        tool.set_flat_params_custom(model, global_param, 0.5)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedMul-fix Round {item} Accuracy on global test set: {global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data':global_acc}, item)
