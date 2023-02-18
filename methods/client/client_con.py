# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/15 13:41
import torch
from torch import nn
from torch.optim import lr_scheduler

from src.optimizer.loss_con import SupConLoss, CosLoss

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
        criterion = SupConLoss(temperature=self.args.temp)
        criterion2 = CosLoss()
        reps_map = {}
        for epoch in range(self.args.epoch):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(device), labels.to(device)
                bsz = labels.shape[0]
                features = encoder(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                if epoch == self.args.epoch-1:
                    f = (f1+f2)/2
                    labels_u = torch.unique(labels)
                    f_s = {i.item():[f[labels==i].mean(dim=0)] for i in labels_u}
                    temp_map = {k:f_s[k]+reps_map.get(k,[]) for k in f_s.keys()}
                    reps_map = temp_map
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                con_loss = criterion(features, labels)
                cos_loss = criterion2(features,labels,self.anchor)
                # print(f"epoch{epoch} b-idx {batch_idx} con_loss = {con_loss.item()} cos_loss={cos_loss.item()}  ")
                loss = cos_loss + con_loss
                loss.backward()
                optimizer.step()
            scheduler.step()
        # 训练完成之后，计算每个类表示的均值，并和encoder一同上传到server
        final_reps = {k:torch.stack(reps_map.get(k)).mean(dim=0) for k in reps_map.keys()}
        # 训练完成后，看下平均reps和anchor之间的相似度
        for k in final_reps.keys():
            cos_fun = nn.CosineSimilarity(dim=0,eps=1e-6)
            cos_sim = cos_fun(final_reps[k],self.anchor[k])
            print(f"平均label {k} cos_sim={cos_sim}")
        return encoder, final_reps

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






