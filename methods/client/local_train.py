# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/12 22:08
import copy
import time

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import methods.tool.tool as tool

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
        return net.parameters(), sum(epoch_loss) / len(epoch_loss)

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
        return net.parameters(), sum(epoch_loss) / len(epoch_loss)

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

                optimizer.step()
                meme_optimizer.step()
        return model.parameters(), meme.parameters()

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

                # with amp.autocast(enabled=args.amp): 精度转换（暂时不要）
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
                if(round_num>20):
                    t_loss += t_loss_mpl * min(1., (round_num-20) / 20)

                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1e5)
                t_optimizer.step()

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
