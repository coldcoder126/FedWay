# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/15 13:41
import torch
from torch import nn
from torch.optim import lr_scheduler

from methods.tool import con_tool
from src.optimizer.loss_con import SupConLoss, CosLoss, SupConAnchorLoss, SupConNceLoss,MySupConLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ClientCon(object):
    def __init__(self, args, train_loader, lr, anchor,encoder):
        self.args = args
        self.contras_loss = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.lr = lr
        self.anchor = anchor
        self.encoder = encoder

    def train_encoder(self):
        encoder = self.encoder.to(device)
        optimizer = torch.optim.SGD(encoder.parameters(), self.lr, weight_decay=1e-3, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        encoder.train()

        criterion2 = CosLoss()
        reps_list_map = {}  # key:label， val:list[tensor]
        class_dis_map = {}
        for epoch in range(3):
            con_loss_list = []
            cos_loss_list = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(device), labels.to(device)
                features = encoder(images).to(device)
                con_loss = MySupConLoss(features,torch.cat([labels,labels]),0.5)
                # bsz = labels.shape[0]
                # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                # con_tool.collect_reps_by_class(reps_list_map,features,labels)
                # con_loss = criterion(features, labels)
                # cos_loss = criterion2(features,labels,self.anchor)
                con_loss_list.append(con_loss.item())
                # cos_loss_list.append(cos_loss.item())
                loss =  con_loss
                loss.backward()
                optimizer.step()
                if epoch == 0:
                    con_tool.count_class_dis(class_dis_map,labels)
            scheduler.step()
            print(f"epoch{epoch} lr = {optimizer.param_groups[0]['lr']} con_loss = {sum(con_loss_list)/len(con_loss_list)} cos_loss={sum(cos_loss_list)/len(cos_loss_list)}  ")
            if epoch == 1:
                print(class_dis_map)
            if epoch %2==0:
                reps_map = con_tool.cliemt_sim_check(reps_list_map, self.anchor)
        # 训练完成后，看下平均reps和anchor之间的相似度
        reps_map = con_tool.cliemt_sim_check(reps_list_map, self.anchor)
        return encoder, reps_map

    def train_encoder_with_anchor(self):
        encoder = self.encoder.to(device)
        optimizer = torch.optim.SGD(encoder.parameters(), self.lr, weight_decay=1e-3, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        encoder.train()


    def train_classifier(self,classifier):
        encoder = self.encoder.to(device)
        classifier = classifier.to(device)

        optimizer = torch.optim.SGD(classifier.parameters(), lr=self.lr, weight_decay=1e-3, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        classifier.train()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.args.epoch):
            all_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(device), labels.to(device)
                bsz = labels.shape[0]
                features = encoder(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1, f2]).detach()
                labels = torch.cat([labels,labels])
                pred = classifier(features)
                ce_loss = criterion(pred,labels)
                all_loss.append(ce_loss.item())
                ce_loss.backward()
                optimizer.step()
            print(f"epoch:{epoch} loss:{sum(all_loss)/len(all_loss)}")
            scheduler.step()
        return classifier






