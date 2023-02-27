# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/25 16:51

import copy

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader
import src.models.model as md
from methods.tool import tool

#使用原始数据集训练教师模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_and_save(args,trainset, part_data):
    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)
                     for i in range(args.client_num)]
    options = md.generate_options(args.dataset, args.model)
    options['model']=args.model
    model = md.choose_model(options)
    client_models = {idx: copy.deepcopy(model) for idx in range(args.client_num)}

    for k in range(args.client_num):
        client_model = client_models[k].to(device)
        optimizer = torch.optim.SGD(client_model.parameters(), lr=0.04, weight_decay=1e-3, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.8)
        loss_func = nn.CrossEntropyLoss()
        for e in range(1):
            for batch_idx, (inputs, labels) in enumerate(train_loaders[k]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                prediction = client_model(inputs)
                loss = loss_func(prediction, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1e5)
                optimizer.step()
            scheduler.step()
        torch.save(client_model.state_dict(), tool.get_model_path(args,k))
        print(f"client{k} done")