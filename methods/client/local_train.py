# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/12 22:08
import copy
import time

import torch
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms as T

from methods.tool.augmentation import RandAugmentCIFAR
from methods.tool.mpl_tool import MyTensorDataset
from src.optimizer.loss_con import MySupConLoss
from src.optimizer.loss_mul import dkd_loss
from methods.tool import con_tool
import methods.tool.tool as tool
from src.optimizer.loss_rslad import rslad_inner_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# pytorch中state_dict()和named_parameters()的差别 https://blog.csdn.net/weixin_41712499/article/details/110198423

class LocalTrain(object):
    def __init__(self, args, train_loader, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.epoch = args.epoch
        self.lr = lr

    def train(self, net):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        net.train()
        net = net.to(device)
        epoch_loss = []
        for epoch in range(self.epoch):
            per_epoch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                prediction = net(inputs)
                loss = self.loss_func(prediction, labels)
                per_epoch_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1e5)
                optimizer.step()
            per_loss = sum(per_epoch_loss) / len(per_epoch_loss)
            epoch_loss.append(per_loss)
        return net, sum(epoch_loss) / len(epoch_loss)

    # 用于fedavg2
    def train_aug(self, loader, loader_aug, net):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        net.train()
        net = net.to(device)
        epoch_loss = []
        err_local_count = 0
        for epoch in range(self.epoch):
            per_epoch_loss = []
            for idx, (data, aug_data) in enumerate(zip(loader, loader_aug)):
                inputs, labels = data[0], data[1]
                inputs_aug, labels_aug = aug_data[0], aug_data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug = inputs_aug.to(device)
                optimizer.zero_grad()
                prediction = net(inputs)
                if epoch == 0:
                    pred_local = torch.argmax(prediction, dim=-1)
                    err_local = torch.eq(pred_local, labels)
                    err_local_count += (err_local == False).sum()
                loss = self.loss_func(prediction, labels)
                per_epoch_loss.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 100)
                optimizer.step()
            per_loss = sum(per_epoch_loss) / len(per_epoch_loss)
            epoch_loss.append(per_loss)
            data_sum = self.train_loader.dataset.indices.size
            acc = err_local_count /data_sum
        return net, acc


    def train_prox(self, net):
        global_net = copy.deepcopy(net)
        global_weight_collector = list(global_net.to(device).parameters())
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        net.train()
        net = net.to(device)
        epoch_loss = []
        for epoch in range(self.epoch):
            per_epoch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                prediction = net(inputs)
                loss = self.loss_func(prediction,labels)

                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((self.args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg
                per_epoch_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1e5)
                optimizer.step()
            per_loss = sum(per_epoch_loss) / len(per_epoch_loss)
            epoch_loss.append(per_loss)

    def train_scaffold(self, net, c):
        global_net = copy.deepcopy(net)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        net.train()
        cnt = 0
        net = net.to(device)
        epoch_loss = []
        c_global_para = global_net.state_dict()
        c_local_para = net.state_dict()
        for epoch in range(self.epoch):
            per_epoch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                prediction = net(inputs)
                loss = self.loss_func(prediction, labels)
                per_epoch_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1e5)
                optimizer.step()
                cnt += 1
                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - self.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)
            per_loss = sum(per_epoch_loss) / len(per_epoch_loss)
            epoch_loss.append(per_loss)

        # 更新c
        c_new_para = net.state_dict()
        c_delta_para = copy.deepcopy(net.state_dict())
        global_model_para = global_net.state_dict()
        net_para = net.state_dict()
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (
                        cnt * self.lr)
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        net.load_state_dict(c_new_para)
        return net.parameters(), sum(epoch_loss) / len(epoch_loss), c_delta_para


    # 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.9

        for e in range(self.epoch):
            # Training
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                meme_optimizer.zero_grad()

                output_local = model(inputs)
                output_meme = meme(inputs)
                # 蒸馏，互学习
                ce_local = CE_Loss(output_local, labels)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
                ce_meme = CE_Loss(output_meme, labels)
                kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))


                loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * ce_meme + (1 - beta) * kl_meme
                loss = loss_local + loss_meme
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e5)
                torch.nn.utils.clip_grad_norm_(meme.parameters(), 1e5)

                optimizer.step()
                meme_optimizer.step()

# 增强互学习，教师模型输入加噪的数据，本地客户端使用不加噪的数据
class LocalMutualAug(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.9

        for e in range(self.epoch):
            # loss_list = []
            # loss_ce_local = []
            # loss_ce_meme = []
            # Training
            for idx , (data, aug_data) in enumerate(zip(self.train_loader,self.train_loader_aug)):
                inputs,labels = data[0],data[1]
                inputs_aug, labels_aug = aug_data[0], aug_data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug = inputs_aug.to(device)
                # optimizer.zero_grad()
                meme_optimizer.zero_grad()

                # output_local = model(inputs)
                # output_meme = meme(inputs_aug)
                #
                # # 蒸馏，互学习
                # ce_local = CE_Loss(output_local, labels)
                # kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
                # ce_meme = CE_Loss(output_meme, labels)
                # kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))
                #
                # loss_ce_local.append(ce_local.item())
                # loss_ce_meme.append(ce_meme.item())
                #
                # loss_local = alpha * ce_local + (1 - alpha) * kl_local
                # loss_meme = beta * ce_meme + (1 - beta) * kl_meme
                # loss = loss_local + loss_meme
                # loss_list.append(loss.item())
                # loss.backward()

                output_T = model(inputs)
                epsilon = 8 / 255.0
                output_S_ = rslad_inner_loss(meme, output_T, inputs, labels, meme_optimizer,
                                              step_size=2 / 255.0, epsilon=epsilon, perturb_steps=10)
                output_S = meme(inputs)
                # output_S_ = meme(inputs_aug)

                kl1 = KL_Loss(LogSoftmax(output_S), Softmax(output_T.detach()))
                kl2 = KL_Loss(LogSoftmax(output_S_), Softmax(output_T.detach()))

                loss = kl1 + kl2
                loss.backward()
                # optimizer.step()
                meme_optimizer.step()
            # print(f"epoch {e} loss :{sum(loss_list)/len(loss_list)} ce_local={sum(loss_ce_local)/len(loss_ce_local)} ce_meme = {sum(loss_ce_meme)/len(loss_ce_meme)}")

class LocalMutualAugCon(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.9
        class_dis_map = {}
        for e in range(self.epoch):
            # Training
            loss_ce_local = []
            loss_ce_meme = []
            conl = []
            cong = []
            loss_cons = []
            for idx , (data, aug_data) in enumerate(zip(self.train_loader,self.train_loader_aug)):
                inputs,labels = data[0],data[1]
                inputs_aug, labels_aug = aug_data[0], aug_data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug = inputs_aug.to(device)
                if e == 0:
                    con_tool.count_class_dis(class_dis_map, labels)
                optimizer.zero_grad()
                meme_optimizer.zero_grad()
                reps_local = model.base(inputs)
                reps_aug = meme.base(inputs_aug)

                output_local = model.head(reps_local)
                output_meme = meme.head(reps_aug)

                # 方式一
                # fs = torch.cat([reps_local,reps_aug])
                # ls = torch.cat([labels,labels]).detach()
                #
                # if torch.unique(labels).shape[0] == 1:
                #     sup_con_loss = torch.tensor(0)
                # else:
                #     sup_con_loss = MySupConLoss(fs,ls, 0.5)

                # 方式二
                if torch.unique(labels).shape[0] > 1:
                    sup_con_loss_l = MySupConLoss(reps_local,labels, 0.5)
                    sup_con_loss_g = MySupConLoss(reps_aug,labels,0.5)
                    conl.append(sup_con_loss_l.item())
                    cong.append(sup_con_loss_g.item())
                    kl_l = KL_Loss(LogSoftmax(reps_local), Softmax(reps_aug.detach()))
                    kl_g = KL_Loss(LogSoftmax(reps_aug), Softmax(reps_local.detach()))
                    sup_con_loss = sup_con_loss_l+sup_con_loss_g + (1-alpha) * (kl_l + kl_g)
                    loss_cons.append(sup_con_loss.item())

                # 蒸馏，互学习
                ce_local = CE_Loss(output_local, labels)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
                ce_meme = CE_Loss(output_meme, labels)
                kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))

                loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * ce_meme + (1 - beta) * kl_meme
                loss = loss_local + loss_meme
                if torch.unique(labels).shape[0] > 1:
                    loss += sup_con_loss

                loss_ce_local.append(ce_local.item())
                loss_ce_meme.append(ce_meme.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e5)
                torch.nn.utils.clip_grad_norm_(meme.parameters(), 1e5)
                optimizer.step()
                meme_optimizer.step()
            if e == 0:
                print(class_dis_map)
            print(f"epoch {e} ce_local :{sum(loss_ce_local)/len(loss_ce_local)} ce_meme:{sum(loss_ce_meme)/(len(loss_ce_meme)+0.1)} con_l = {sum(conl)/(len(conl)+0.01)} con_g = {sum(cong)/(len(cong)+0.01)} loss_cons:{sum(loss_cons)/(len(loss_cons)+0.01)}  ")


class LocalMutualAugFix(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.lr/10, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.9
        class_dis_map = {}
        for e in range(self.epoch):
            # Training
            loss_ce_local = []
            loss_ce_meme = []
            conl = []
            cong = []
            loss_cons = []
            for idx , (data, aug_data) in enumerate(zip(self.train_loader,self.train_loader_aug)):
                inputs,labels = data[0],data[1]
                inputs_aug, labels_aug = aug_data[0], aug_data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug = inputs_aug.to(device)
                # if e == 0:
                #     con_tool.count_class_dis(class_dis_map, labels)
                # optimizer.zero_grad()
                meme_optimizer.zero_grad()
                reps_local = model.base(inputs)
                reps_aug = meme.base(inputs_aug)

                output_local = model.head(reps_local)
                output_meme = meme.head(reps_aug)


                # 方式一
                fs = torch.cat([reps_local,reps_aug])
                ls = torch.cat([labels,labels]).detach()

                if torch.unique(labels).shape[0] == 1:
                    sup_con_loss = torch.tensor(0)
                else:
                    sup_con_loss = MySupConLoss(fs,ls, 0.5)

                # 方式二
                # if torch.unique(labels).shape[0] > 1:
                #     # sup_con_loss_l = MySupConLoss(reps_local,labels, 0.5)
                #     sup_con_loss_g = MySupConLoss(reps_aug,labels,0.5)
                #     # conl.append(sup_con_loss_l.item())
                #     cong.append(sup_con_loss_g.item())
                #     # kl_l = KL_Loss(LogSoftmax(reps_local), Softmax(reps_aug.detach()))
                #     kl_g = KL_Loss(LogSoftmax(reps_aug), Softmax(reps_local.detach()))
                #     sup_con_loss = beta * sup_con_loss_g + (1-beta) *  kl_g
                #     loss_cons.append(sup_con_loss.item())

                # 蒸馏，互学习
                # ce_local = CE_Loss(output_local, labels)
                # kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
                ce_meme = CE_Loss(output_meme, labels)
                kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))

                # loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * ce_meme + (1 - beta) * kl_meme
                loss = loss_meme
                if torch.unique(labels).shape[0] > 1:
                    loss += sup_con_loss

                # loss_ce_local.append(ce_local.item())
                loss_ce_meme.append(ce_meme.item())

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e5)
                torch.nn.utils.clip_grad_norm_(meme.parameters(), 1e5)
                # optimizer.step()
                meme_optimizer.step()
            # if e == 0:
            #     print(class_dis_map)
            # print(f"epoch {e} ce_local :{sum(loss_ce_local)/len(loss_ce_local)} ce_meme:{sum(loss_ce_meme)/(len(loss_ce_meme)+0.1)} con_l = {sum(conl)/(len(conl)+0.01)} con_g = {sum(cong)/(len(cong)+0.01)} loss_cons:{sum(loss_cons)/(len(loss_cons)+0.01)}  ")


# 增强互学习，教师模型输入加噪的数据，本地客户端使用不加噪的数据，使用上次的CE差值
class LocalMutualAug2(object):
    def __init__(self, args, train_set, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = tool.get_local_loader(args,train_set)
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.9

        for e in range(self.epoch):
            # loss_list = []
            # loss_ce_local = []
            # loss_ce_meme = []
            # Training
            for batch_idx, (images, labels) in enumerate(self.train_loader):

                inputs,inputs_aug,labels = images[0].to(device), images[1].to(device), labels.to(device)
                meme_optimizer.zero_grad()

                output_T = model(inputs)
                output_S = meme(inputs)
                output_S_ = meme(inputs_aug)

                kl1 = KL_Loss(LogSoftmax(output_S), Softmax(output_T.detach()))
                kl2 = KL_Loss(LogSoftmax(output_S_), Softmax(output_T.detach()))

                loss = kl1 + kl2
                loss.backward()
                # optimizer.step()
                meme_optimizer.step()
            # print(f"epoch {e} loss :{sum(loss_list)/len(loss_list)} ce_local={sum(loss_ce_local)/len(loss_ce_local)} ce_meme = {sum(loss_ce_meme)/len(loss_ce_meme)}")


# 增强互学习，教师模型输入加噪的数据，本地客户端使用不加噪的数据，下一个epoch中添加上一个epoch中预测错的数据，再训练
cifar10_mean = (0.491400, 0.482158, 0.4465231)
cifar10_std = (0.247032, 0.243485, 0.2615877)
class LocalMutualAug2_1(object):
    def __init__(self, args, train_set, round, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_set = train_set  # 包含原始数据和增强数据
        self.epoch = args.epoch
        self.lr = lr
        self.round = round

    # 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.9

        err_inputs = torch.zeros([1, 3, 32, 32]).to(device)
        err_labels = torch.zeros([1, ], dtype=torch.int64).to(device)
        # 获取本地的增强数据集
        local_train_set = self.train_set[0]
        local_train_set_aug = self.train_set[1]
        for e in range(self.epoch):
            # Training
            # 获取预测错误的数据集
            if (self.round > self.args.threshold) & (e == 1) :
                # 将本地的dataset进行扩充, 预测错误数据集中第一个数据和标签不要
                err_data_set = MyTensorDataset(err_inputs[0:-1].cpu(), err_labels[0:-1].cpu(),
                                               transforms=T.Compose([T.RandomHorizontalFlip(),  # 旋转和翻转
                                                                     T.RandomCrop(size=self.args.resize,
                                                                                                                                                                                                                                                                                                                               padding=int(self.args.resize * 0.125),
                                                                                  fill=128,
                                                                                  padding_mode='constant')]))
                err_data_set_aug = MyTensorDataset(err_inputs[0:-1].cpu(), err_labels[0:-1].cpu(),
                                                   transforms=T.Compose([T.RandomHorizontalFlip(),  # 旋转和翻转
                                                                        T.RandomCrop(size=self.args.resize,
                                                                                  padding=int(self.args.resize * 0.125),
                                                                                  fill=128,
                                                                                  padding_mode='constant'),
                                                                         RandAugmentCIFAR(n=2, m=8)
                                                                         ]))
                local_train_set = torch.utils.data.ConcatDataset([local_train_set, err_data_set])
                local_train_set_aug = torch.utils.data.ConcatDataset([local_train_set_aug, err_data_set_aug])
                err_inputs = err_inputs[0:1]
                err_labels = err_labels[0:1]

            # 原数据的dataloader 和 增强数据的dataloader
            train_loader = DataLoader(local_train_set, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=self.args.num_workers,drop_last=True)
            train_loader_aug = DataLoader(local_train_set_aug, batch_size=self.args.batch_size, shuffle=False,
                                          num_workers=self.args.num_workers, drop_last=True)
            print(f"epoch:{e}")
            err_local_count = 0
            err_meme_count =0
            for idx, (data, aug_data) in enumerate(zip(train_loader, train_loader_aug)):
                inputs, labels = data[0], data[1]
                inputs_aug, labels_aug = aug_data[0], aug_data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug = inputs_aug.to(device)
                optimizer.zero_grad()
                meme_optimizer.zero_grad()

                output_local = model(inputs)
                output_meme = meme(inputs_aug)

                # getErrData(output_meme)
                # 蒸馏，互学习
                ce_local = CE_Loss(output_local, labels)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
                ce_meme = CE_Loss(output_meme, labels)
                kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))

                if (self.round > self.args.threshold) and (e == 0):
                    pred_local = torch.argmax(output_local, dim=-1)
                    pred_meme = torch.argmax(output_meme, dim=-1)

                    err_local = torch.eq(pred_local, labels)
                    err_meme = torch.eq(pred_meme, labels)
                    err_local_count += (err_local==False).sum()
                    err_meme_count += (err_meme==False).sum()
                    # print(f'err_meme_count:{err_meme_count}')
                    # 对两个预测错误的求或操作，（如果两个都预测错误，则不会加入两次）
                    # 看下本地和全局预测的错误数量
                    sum_local = (err_local==False).sum()
                    err_all = ~(err_local | err_meme)
                    # 收集预测错误的原始数据，只收集原始数据[ok]

                    err_inputs = torch.cat((err_inputs, inputs[err_all]), dim=0)
                    err_labels = torch.cat((err_labels, labels[err_all]), dim=0)

                loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * ce_meme + (1 - beta) * kl_meme
                loss = loss_local + loss_meme
                loss.backward()

                optimizer.step()
                meme_optimizer.step()
            if (err_local_count != 0) or (err_meme_count != 0):
                print(f"epoch-{e} local_err count:{err_local_count}, meme_err count:{err_meme_count}")
        return model.parameters(), meme.parameters()



# 学生使用弱增强学习，教师使用强增强学习，最后共同使用真实数据微调
class LocalMutualAug3(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.7

        for e in range(self.epoch):
            # Training
            for idx, (labeled_data, aug_data) in enumerate(zip(self.train_loader, self.train_loader_aug)):
                inputs,labels = labeled_data[0], labeled_data[1]
                inputs_aug_w, inputs_aug_s, targets_u = aug_data[0][0], aug_data[0][1], aug_data[1]

                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug_w = inputs_aug_w.to(device)
                inputs_aug_s = inputs_aug_s.to(device)
                optimizer.zero_grad()
                meme_optimizer.zero_grad()

                output_local = model(inputs)
                output_w = meme(inputs_aug_w)
                output_s = meme(inputs_aug_s)
                output_avg = (output_s + output_w)/2


                ce_local = CE_Loss(output_local, labels)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_avg.detach()))

                ce_meme = CE_Loss(output_avg, labels)
                kl_meme = KL_Loss(LogSoftmax(output_avg), Softmax(output_local.detach()))
                kl_meme_loss = KL_Loss(LogSoftmax(output_w), Softmax(output_s.detach()))


                # 蒸馏，互学习

                loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * ce_meme + 0.2 * kl_meme + 0.1 * kl_meme_loss
                loss = loss_local + loss_meme
                loss.backward()

                optimizer.step()
                meme_optimizer.step()
        return model.parameters(), meme.parameters()

# 教师网络 强增强和弱增强 互学习
class LocalMutualAug3(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.7

        for e in range(self.epoch):
            # Training
            for idx, (labeled_data, aug_data) in enumerate(zip(self.train_loader, self.train_loader_aug)):
                inputs,labels = labeled_data[0], labeled_data[1]
                inputs_aug_w, inputs_aug_s, targets_u = aug_data[0][0], aug_data[0][1], aug_data[1]

                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug_w = inputs_aug_w.to(device)
                inputs_aug_s = inputs_aug_s.to(device)
                optimizer.zero_grad()
                meme_optimizer.zero_grad()

                output_local = model(inputs)
                output_w = meme(inputs_aug_w)
                output_s = meme(inputs_aug_s)
                output_avg = (output_s + output_w)/2


                ce_local = CE_Loss(output_local, labels)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_avg.detach()))

                ce_meme1 = CE_Loss(output_s, labels)
                kl_meme1 = KL_Loss(LogSoftmax(output_s), Softmax(output_w.detach()))
                ce_meme2 = CE_Loss(output_s, labels)
                kl_meme2 = KL_Loss(LogSoftmax(output_w), Softmax(output_s.detach()))


                # 蒸馏，互学习

                loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * (ce_meme1+ce_meme2) + (1-beta) * (kl_meme1+kl_meme2)
                loss = loss_local + loss_meme/2
                loss.backward()

                optimizer.step()
                meme_optimizer.step()
        return model.parameters(), meme.parameters()

# 教师网络 现学现教
class LocalMutualAug4(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.9

        for e in range(self.epoch):
            # Training
            for idx, (data, aug_data) in enumerate(zip(self.train_loader, self.train_loader_aug)):
                inputs, labels = data[0], data[1]
                inputs_aug, labels_aug = aug_data[0], aug_data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug = inputs_aug.to(device)
                optimizer.zero_grad()
                meme_optimizer.zero_grad()

                # 教师先学习
                output_meme = meme(inputs)
                ce_meme = CE_Loss(output_meme, labels)
                ce_meme.backward()
                meme_optimizer.step()
                meme_optimizer.zero_grad()
                # 教师学完之后再和学生网络互学习


                output_local = model(inputs)
                output_meme2 = meme(inputs_aug)
                ce_local = CE_Loss(output_local, labels)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
                ce_meme2 = ce_meme = CE_Loss(output_meme2, labels)
                kl_meme = KL_Loss(LogSoftmax(output_meme2), Softmax(output_local.detach()))

                loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * ce_meme2 + (1-beta) * kl_meme
                loss = loss_local + loss_meme
                loss.backward()

                optimizer.step()
                meme_optimizer.step()

        return model.parameters(), meme.parameters()


# 教师网络 现学现教 先学原图，在学增强
KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def getLoss(output_local, output_meme, labels, alpha, beta):
    ce_local = CE_Loss(output_local, labels)
    kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme))
    ce_meme = CE_Loss(output_meme, labels)
    kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local))
    loss_local = alpha * ce_local + (1 - alpha) * kl_local
    loss_meme = beta * ce_meme + (1 - beta) * kl_meme
    loss = loss_local + loss_meme
    return loss

# 学生先对原图进行预测，在使用弱增强和教师模型互学习，教师模型使用强增强。
# 学生先更新，更新完之后再在原图上预测，用学生的反馈再更新教师。
class LocalMutualMpl(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型



    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)


        alpha = 0.9
        beta = 0.9


        for e in range(self.epoch):
            # Training
            for idx, (labeled_data, aug_data) in enumerate(zip(self.train_loader, self.train_loader_aug)):
                inputs, labels = labeled_data[0], labeled_data[1]
                inputs_aug_w, inputs_aug_s, targets_u = aug_data[0][0], aug_data[0][1], aug_data[1]

                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug_w, inputs_aug_s = inputs_aug_w.to(device), inputs_aug_s.to(device)

                output_old = model(inputs)
                s_loss_old = CE_Loss(output_old.detach(), labels)
                # 互学习
                output_aug_w = model(inputs_aug_w)
                output_aug_s = meme(inputs_aug_s)
                ce_w = CE_Loss(output_aug_w,labels)
                ce_s = CE_Loss(output_aug_s,labels)
                kl_w = KL_Loss(LogSoftmax(output_aug_w), Softmax(output_aug_s.detach()))
                kl_s = KL_Loss(LogSoftmax(output_aug_s), Softmax(output_aug_w.detach()))
                loss_local = alpha * ce_w + (1 - alpha) * kl_w
                loss_meme = beta * ce_s + (1 - beta) * kl_s
                loss_local.backward()
                optimizer.step()

                # 再预测
                output_new = model(inputs)
                s_loss_new = CE_Loss(output_new.detach(), labels)
                t_loss = loss_meme * (1+s_loss_old - s_loss_new)

                # 有个问题，s_loss的backward() 对教师模型的影响
                # meme_optimizer.zero_grad()
                t_loss.backward()
                meme_optimizer.step()

        return model.parameters(), meme.parameters()


# 原图和一个增强 一致性+蒸馏
class LocalMutualAug6(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr
# 接受两个网络互学习，model是本地模型，meme是全局模型
    def train_mul(self, net1, net2):
        model = net1
        meme = net2
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        meme_optimizer = torch.optim.SGD(meme.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)

        model.train()
        model = model.to(device)
        meme.train()
        meme = meme.to(device)

        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()
        alpha = 0.9
        beta = 0.9

        for e in range(self.epoch):
            # Training
            for idx , (data, aug_data) in enumerate(zip(self.train_loader,self.train_loader_aug)):
                inputs,labels = data[0],data[1]
                inputs_aug, labels_aug = aug_data[0], aug_data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                inputs_aug = inputs_aug.to(device)
                optimizer.zero_grad()
                meme_optimizer.zero_grad()

                output_local = model(inputs)
                output_meme = meme(inputs_aug)
                # 蒸馏，互学习
                ce_local = CE_Loss(output_local, labels)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
                ce_meme = CE_Loss(output_meme, labels)
                kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))

                loss_local = alpha * ce_local + (1 - alpha) * kl_local
                loss_meme = beta * ce_meme + (1 - beta) * kl_meme
                loss = loss_local + loss_meme
                loss.backward()

                optimizer.step()
                meme_optimizer.step()
        return model.parameters(), meme.parameters()

class LocalMPL(object):
    def __init__(self, args, labeled_loader, unlabeled_loader, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.epoch = args.epoch
        self.lr = lr

    def train_MPL(self, teacher_model, student_model,round_num):
        threshold = (self.args.round_num*0.15 +(round_num*0.8))/self.args.round_num  # 教师对无标签数据的预测门槛
        temperature = 1.2  # 温度
        uda_steps = self.args.round_num
        lambda_u = 1

        t_optimizer = torch.optim.SGD(teacher_model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        s_optimizer = torch.optim.SGD(student_model.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        # t_scheduler = tool.get_cosine_schedule_with_warmup(t_optimizer,
        #                                               30,
        #                                               self.args.round_num)
        # s_scheduler = tool.get_cosine_schedule_with_warmup(s_optimizer,
        #                                               30,
        #                                               self.args.round_num)
        # print(f"cur lr: {t_scheduler.get_last_lr()}")=

        teacher_model.train()
        teacher_model = teacher_model.to(device)
        student_model.train()
        student_model = student_model.to(device)
        before = time.time()
        for e in range(self.epoch):
            for idx, (labeled_data, unlabeled_data) in enumerate(zip(self.labeled_loader,self.unlabeled_loader)):
                images_l, targets = labeled_data[0],labeled_data[1]
                images_us, images_uw, targets_u = unlabeled_data[0][0],unlabeled_data[0][1],unlabeled_data[1]
                t_optimizer.zero_grad()
                s_optimizer.zero_grad()
                images_l = images_l.to(device)
                images_uw = images_uw.to(device)
                images_us = images_us.to(device)
                targets = targets.to(device)

                with amp.autocast(enabled=True): # 精度转换（暂时不要）
                    batch_size = images_l.shape[0]
                    t_images = torch.cat((images_l, images_uw, images_us))

                    t_logits = teacher_model(t_images)
                    # 有标签数据的Logit分数
                    t_logits_l = t_logits[:batch_size]
                    # 无标签数据两种增强的分数
                    t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
                    del t_logits
                    # 有标签数据和target之间的loss
                    t_loss_l = self.loss_func(t_logits_l, targets)
                    # 根据温度，将弱增强的的分数进行softmax
                    soft_pseudo_label = torch.softmax(t_logits_uw.detach() / temperature, dim=-1)
                    # 将分数按最大提取为硬标签
                    max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
                    # 根据置信度取出
                    mask = max_probs.ge(threshold).float()
                    t_loss_u = torch.mean(
                        -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
                    )
                    weight_u = lambda_u * min(1., (round_num + 1) / uda_steps)
                    t_loss_uda = t_loss_l + weight_u * t_loss_u

                    # 学生模型预测有标签数据和无标签_强增强数据
                    s_images = torch.cat((images_l, images_us))
                    s_logits = student_model(s_images)
                    s_logits_l = s_logits[:batch_size]
                    s_logits_us = s_logits[batch_size:]
                    del s_logits
                    # 学生模型更新前，计算有标签数据和真实数据的损失
                    s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
                    s_loss = self.loss_func(s_logits_us, hard_pseudo_label)

                s_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1e5)
                s_optimizer.step()
                # s_scheduler.step()


                # 更新教师模型
                with amp.autocast(enabled=True):
                    with torch.no_grad():
                        s_logits_l = student_model(images_l)
                    # 学生模型更新后，再预测有标签数据和真实值之间的损失
                    s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)
                    dot_product = s_loss_l_old - s_loss_l_new
                    # 前一个硬标签是t_logits_uw经过温度蒸馏后得到的
                    # 这一个硬标签是t_logits_us直接得到的
                    _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
                    t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
                    # t_loss = t_loss_uda + t_loss_mpl
                    t_loss = t_loss_uda
                    if round_num>20:
                        t_loss += t_loss_mpl * min(1., (round_num-20) / 20)

                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1e5)
                t_optimizer.step()
            endtime = time.time()
            print(f"epooch-{e}训练用时：{endtime - before}")
                # t_scheduler.step()
                # todo 微调
                # prediction = teacher_model(images_l)
                # loss = self.loss_func(prediction, targets)
                # loss.backward()
                # t_optimizer.step()


        # 结束后，学生模型作为本地模型，教师模型作为全局模型
        return teacher_model, student_model


class LocalDC(object):
    def __init__(self, args, train_loader, train_loader_aug, lr):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.train_loader_aug = train_loader_aug
        self.epoch = args.epoch
        self.lr = lr

    def train_fed_dc(self,net):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        net.train()
        net = net.to(device)
        epoch_loss = []
        for epoch in range(self.epoch):
            per_epoch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                prediction = net(inputs)
                loss = self.loss_func(prediction, labels)
                loss_f_i = loss/list(labels.size())[0]

                local_parameter = None
                for param in net.parameters():
                    if not isinstance(local_parameter, torch.Tensor):
                        # Initially nothing to concatenate
                        local_parameter = param.reshape(-1)
                    else:
                        local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)
                # loss_cp = alpha / 2 * torch.sum((local_parameter - (global_model_param - hist_i)) * (
                #             local_parameter - (global_model_param - hist_i)))
                # loss_cg = torch.sum(local_parameter * state_update_diff)


                per_epoch_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1e5)
                optimizer.step()
            per_loss = sum(per_epoch_loss) / len(per_epoch_loss)
            epoch_loss.append(per_loss)
        return net.parameters(), sum(epoch_loss) / len(epoch_loss)
