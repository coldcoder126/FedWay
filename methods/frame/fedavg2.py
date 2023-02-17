# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/3 20:10

# 服务端学习预测下一个全局模型的正确率
import copy
import torch
from torch import nn
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
import src.models.server_model as smd
from methods.tool import tool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实现fedavg算法
def fedavg(args, trainset, testset, part_data):
    path = tool.mk_path(args)
    writer_file = f"fedavg2-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")

    # 给所有客户端编号进行one-hot编码
    client_onehot = torch.nn.functional.one_hot(torch.tensor(np.arange(0,args.client_num),dtype=torch.int64))
    # 构建网络 f(Mg,[Mi];[Ci];Wi)


    # 所有已经分好组的训练集和测试集
    train_loaders = [DataLoader(Subset(trainset, part_data.client_dict[i]), batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers, drop_last=True)
                     for i in range(args.client_num)]
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model']=args.model
    # 分别为t-1,t,和t+1轮的模型
    gm = md.choose_model(options)
    gm_new = copy.deepcopy(gm)
    client_private_params = {}
    selected_layer = []
    # for ly in gm_new.named_parameters():
    #     print(ly[0])
    g1 = gm_new.parameters()
    g2 = list(g1)
    g3 = g2[-args.ln:]
    gm_shape = torch.cat([param.data.view(-1,) for param in g3]).shape[0]
    server_model = smd.WeightNet(args.client_num, gm_shape, args.clients_per_round)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, server_model.parameters()), lr=0.0001)
    server_model.train()

    for item in range(args.round_num):
        lr = tool.getLr(args, item)
        # 每轮随机选择部分客户端
        np.random.seed(item)
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        selected_client_onehot = client_onehot[idx_users]

        # 准备要下发的模型（上一轮收集的模型）
        if item>0:
            m_list = []
            gm_list = list(gm_new.parameters())
            gm_t = torch.cat([param.data.view(-1,) for param in gm_list[-args.ln:]])
            m_list.append(gm_t)
            for c in selected_layer:
                m_list.append(torch.cat([param.data.view(-1,) for param in c]))
            selected_layer.clear()
            # 将收集的tensor按dim=1拼接起来
            layers = torch.stack(m_list).to(device)
            # server_model = smd.WeightNet(args.client_num, gm_t.shape[0], args.clients_per_round)
            # 服务端准备训练
            # for name, param in server_model.named_children():
            #     if "fc" in name:
            #         for param in param.parameters():
            #             param.requires_grad = False
            server_model = server_model.to(device)
            optimizer.zero_grad()
            output = server_model(layers, selected_client_onehot.to(device))
            new_layers = output[0].squeeze()
            pred = output[1].squeeze()
            tool.set_layer_to_model(args,new_layers,gm_new)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        # global_acc = tool.global_test(gm_new, test_loader)
        # print(f'FedAvg2 Round {item} Accuracy on global test set: {global_acc}%')
        # writer.add_scalars('Loss/Epoch',
        #                    {'All Data':global_acc}, item)





        selected_params = []    #选中客户端的模型
        selected_data_num = []  #选中客户端的样本数量

        accs=[]
        for k in range(args.clients_per_round):
            # 将每个客户端的私有模型的前p层更换为全局模型中的参数
            if idx_users[k] in client_private_params.keys():
                private_model = client_private_params[idx_users[k]]
            else:
                private_model = copy.deepcopy(gm)
            tool.replace_layers(args,private_model,gm_new)
            # private_model.load_state_dict(global_model)
            # net2.linear.load_state_dict(net1.linear1.state_dict())

            # 训练每个选到的客户端，并收集全局模型在每个客户端上的正确率
            local = LocalTrain(args, train_loaders[idx_users[k]],lr)
            global_model = copy.deepcopy(gm_new)

            net, acc = local.train2(global_model)
            param_cp = copy.deepcopy(net)
            client_private_params[idx_users[k]] = param_cp
            param_cp_list = list(param_cp.parameters())
            selected_layer.append(param_cp_list[-args.ln:])
            selected_params.append(tool.get_flat_params_from(net.parameters()))
            selected_data_num.append(train_loaders[idx_users[k]].sampler.num_samples)

            accs.append(acc)
        # 拿到客户端的真实预测后，更新权重网络
        if item>0:
            criteria = nn.MSELoss(reduction='sum')
            label = torch.stack(accs)
            target = torch.ones_like(label,dtype=torch.float)
            loss1 = criteria(pred,label)
            # loss2 = criteria(pred,target)
            server_loss = loss1
            print(f"selected clients:{idx_users}, acc:{[round(i.item(),4) for i in accs ]}, pred={pred.data},loss={server_loss.item()}")
            server_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e3)
            optimizer.step()
            server_model.w1.data.clamp_(1, 10)




