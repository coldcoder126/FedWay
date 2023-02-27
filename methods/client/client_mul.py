# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/22 18:45
import torch
from torch import nn
from torch.optim import lr_scheduler

from methods.tool import con_tool
from src.optimizer.loss_con import SupConLoss, CosLoss, SupConAnchorLoss, SupConNceLoss,MySupConLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ClientMul(object):
    def __init__(self, args, train_loader, lr, local_model, global_model):
        self.args = args
        self.contras_loss = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.lr = lr
        self.local_model = local_model
        self.global_model = global_model



    def train_encoder(self):
        local_model = self.local_model.to(device)
        global_model = self.global_model.to(device)


        optimizer = torch.optim.SGD(local_model.parameters(), self.lr, weight_decay=1e-3, momentum=0.9)
        global_optimizer = torch.optim.SGD(global_model.parameters(), self.lr, weight_decay=1e-3, momentum=0.9)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        global_scheduler = lr_scheduler.StepLR(global_optimizer, step_size=5, gamma=0.8)

        local_model.train()
        global_model.train()
        KL_Loss = nn.KLDivLoss(reduction='batchmean')
        Softmax = nn.Softmax(dim=1)
        LogSoftmax = nn.LogSoftmax(dim=1)
        CE_Loss = nn.CrossEntropyLoss()

        for epoch in range(self.args.epoch):
            loss_list_l = []
            loss_list_g = []
            loss_list_ce = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                # 前10个epoch是本地单独训练，以模拟本地模型可以在空闲时间训练特征表示
                image_ori, image_aug, labels = images[0].to(device),images[1].to(device),labels.to(device)
                feature_ori = local_model.base(image_ori)
                feature_aug = global_model.base(image_aug)

                out = local_model.head(feature_ori)
                out_g = global_model.head(feature_aug)

                ce_loss = CE_Loss(out, labels)
                ce_loss_g = CE_Loss(out_g,labels)
                kl_cls = KL_Loss(LogSoftmax(out), Softmax(out_g.detach()))
                kl_cls_g = KL_Loss(LogSoftmax(out_g), Softmax(out.detach()))

                fs = torch.cat([feature_ori,feature_aug])
                ls = torch.cat([labels,labels]).detach()

                if torch.unique(labels).shape[0] == 1:
                    sup_con_loss_l = torch.tensor([0])
                else:
                    sup_con_loss_l = MySupConLoss(feature_aug,labels, 0.5)
                # sup_con_loss_g = MySupConLoss(feature_aug,labels.detach(), 0.5)
                loss_list_ce.append(ce_loss_g.item())
                # loss_list_l.append(sup_con_loss_l.item())
                # 表示层 kl loss
                # kl_local1 = KL_Loss(LogSoftmax(feature_ori), Softmax(feature_aug.detach()))
                # kl_local2 = KL_Loss(LogSoftmax(feature_aug), Softmax(feature_ori.detach()))



                # loss = sup_con_loss_l + sup_con_loss_g + 0.2*(kl_local1+kl_local2)
                # loss = sup_con_loss_l + ce_loss_g
                loss = ce_loss + ce_loss_g + 0.2*(kl_cls+kl_cls_g)

                loss.backward()
                optimizer.step()
                global_optimizer.step()

            scheduler.step()
            global_scheduler.step()

            print(f"epoch{epoch} lr = {optimizer.param_groups[0]['lr']} ce_loss = {sum(loss_list_ce)/(len(loss_list_ce)+1)} con_loss = {sum(loss_list_l)/(len(loss_list_l)+1)} cos_loss={sum(loss_list_g)/(len(loss_list_g)+1)}  ")
        return global_model


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






