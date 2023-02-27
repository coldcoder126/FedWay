# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/13 13:32

# 有监督对比学习：
# 1. Server收集每个类的一个图片
# 2. Server使用Encoder和一个图片用对比学习计算出表示
# 3. 将Encoder、图片、表示发送给Clients
# 4. 客户端使用该类的图当作锚点，进行对比学习聚类
# 5.每一轮上传训练过的锚点值，并聚合锚点值，
# 6. 当所有锚点距离小于一个数值α后，训练并上传分类器

import copy
import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
from src.models.resnet_big import SupConResNet, LinearClassifier
from methods.tool import tool,con_tool
from methods.tool.mpl_tool import MyTensorDataset
from torchvision.transforms import transforms as T
from src.optimizer.loss_con import Contrastive_loss_batch
from methods.client.client_con import ClientCon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实现fed_con算法
def fed_con(args, trainset, testset, part_data):
    path = tool.mk_path(args)
    writer_file = f"fedavg-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    # 所有已经分好组的训练集和测试集
    train_loaders = con_tool.set_loader(args,part_data)

    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    # 获取每一类的图片各一张
    server_samples = con_tool.get_samples(test_loader)
    class_idx = list(server_samples[2].keys())
    class_tensor = list(server_samples[2].values())
    data = torch.stack(class_tensor)
    label = torch.tensor(class_idx)
    server_dataset_ori = MyTensorDataset(data, label,
                                   transforms =T.Compose([T.ToTensor(),T.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])]))
    server_dataset_aug = MyTensorDataset(data, label,
                                   transforms =T.Compose([T.RandomHorizontalFlip(),
                                                          T.RandomCrop(size=32,padding=int(32 * 0.125),fill=128,padding_mode='constant'),
                                                          T.ToTensor(),T.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])]))
    server_dataset = torch.utils.data.ConcatDataset([server_dataset_ori, server_dataset_aug])
    #

    server_sample_loader = DataLoader(server_dataset,batch_size=20)
    # 初始化一个编码器 将图片编码至128维
    server_encoder = SupConResNet('resnet18')
    server_encoder = server_encoder.to(device)
    server_classifier = LinearClassifier()
    server_optimizer = torch.optim.SGD(server_encoder.parameters(), lr=0.1, weight_decay=1e-3, momentum=0.9)
    server_scheduler = lr_scheduler.StepLR(server_optimizer, step_size=10, gamma=0.8)
    criteria = Contrastive_loss_batch(0.7)
    reps=None
    for e in range(3):
        for batch_idx, (inputs, labels) in enumerate(server_sample_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            server_optimizer.zero_grad()
            reps = server_encoder(inputs)
            loss = criteria(reps,labels)
            print(f" epoch-{e} server loss = {loss.item()}")
            loss.backward()
            server_optimizer.step()
        server_scheduler.step()
    server_reps_map = {idx.item():reps[idx] for idx in labels[:10]}
    con_tool.server_sim_test(server_reps_map)


    # 服务端训练完成后将Encoder、图片、表示发送给Clients
    client_encoders = {idx:copy.deepcopy(server_encoder) for idx in range(args.client_num)}
    client_classifiers = {idx:copy.deepcopy(server_classifier) for idx in range(args.client_num)}
    client_reps_anchor = {}
    for item in range(args.round_num):
        client_data_num = []
        lr = tool.getLr(args, item)
        # 每轮随机选择部分客户端
        np.random.seed(item)
        idx_client = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        classifier_condition = (item >= 1)
        for k in idx_client:
            local_data_loader = train_loaders[k]
            client_data_num.append(local_data_loader.sampler.num_samples)
            # shuffle=False
            #client_data_num.append(local_data_loader.sampler.data_source.indices.size)
            local_encoder = copy.deepcopy(server_encoder)

            # 对encoder进行训练
            client = ClientCon(args, local_data_loader, lr, server_reps_map, local_encoder)
            if not classifier_condition:
                client_encoders[k],client_reps_anchor[k] = client.train_encoder()
            # todo 什么标识说明anchor训练好了呢 （测量各个anchor之间的距离？）
            # 每轮训练结束后将encoder和anchor进行平均
            else: # 将encoder聚合好之后，固定encoder，训练分类器（暂定20 epoch后）
                local_classifier = copy.deepcopy(server_classifier)
                client_classifiers[k] = client.train_classifier(local_classifier)
        if not classifier_condition: #聚合anchor和encoder
            con_tool.anchor_avg(idx_client,client_reps_anchor,server_reps_map,client_data_num)
            con_tool.net_avg(idx_client,client_encoders,server_encoder,client_data_num)
        else: # 固定encoder 聚合classifier
            con_tool.net_avg(idx_client,client_classifiers,server_classifier,client_data_num)
            # 每个round测试精度
            global_acc = con_tool.global_test(server_encoder,server_classifier,test_loader)
            print(f'FedMul Round {item} Accuracy on global test set: {global_acc}%')
            writer.add_scalars('Loss/Epoch',
                               {'All Data': global_acc}, item)










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
            selected_params.append(tool.get_flat_params_from(param))
            selected_data_num.append(train_loaders[idx_users[k]].sampler.num_samples)
            xx = list(param)

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






