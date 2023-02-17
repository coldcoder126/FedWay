# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2023/2/13 15:40
# fed_con的工具类
import copy

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from methods.tool import tool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_samples(data_loader):
    '''从数据集中获取每个类的图片各一张'''
    classes = data_loader.dataset.classes
    class_to_idx = data_loader.dataset.class_to_idx
    idxs = list(class_to_idx.values())
    targets = data_loader.dataset.targets
    target_idx = [ ]
    for idx in targets:
        if idx not in target_idx:
            target_idx.append(idx)
        if len(target_idx) == len(idxs):
            break
    samples = {idx:torch.tensor(data_loader.dataset.data[idx]) for idx in target_idx}
    return classes,class_to_idx,samples


class TwoCrargsransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def set_loader(args,part_data):
    path = f"{args.data_path}/{args.dataset}"
    # construct data loader
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif args.dataset == 'path':
        mean = eval(args.mean)
        std = eval(args.std)
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.resize, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=path,
                                         transform=TwoCrargsransform(train_transform),
                                         download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=path,
                                          transform=TwoCrargsransform(train_transform),
                                          download=True)
    elif args.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=path,
                                            transform=TwoCrargsransform(train_transform))
    else:
        raise ValueError(args.dataset)

    train_sampler = None
    train_loaders = [DataLoader(
        Subset(train_dataset, part_data.client_dict[i]),
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler) for i in range(args.client_num)]

    return train_loaders


def anchor_avg(idx_client, client_reps_anchor,server_reps_anchor):
    label_dic = {}
    dics = [client_reps_anchor[idx] for idx in idx_client]
    for dic in dics:
        for k in dic.keys():
            label_dic[k] = label_dic.get(k,[])+[dic[k]]
    for k in label_dic.keys():
        # 在此之前，看下每个客户端同类anchor之间的余弦相似
        cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
        if len(label_dic[k]) > 1:
            sim = cos_sim(label_dic[k][0],label_dic[k][1])
            print(f"label:{k} sim:{sim}")
        fs = torch.stack(label_dic[k])
        fs_mean = fs.mean(dim=0)
        server_reps_anchor[k] = (server_reps_anchor[k] + fs_mean)/2

# def encoder_avg(idx_client, client_encoders, server_encoder,client_data_num):
#      encoder_param = [tool.get_flat_params_from(m) for m in  [client_encoders[k] for k in idx_client]]
#      param_avg = tool.aggregate_avg(encoder_param,client_data_num)
#      tool.set_flat_params_to(server_encoder,param_avg)

# 聚合非bn参数
def net_avg(idx_client, client_nets, server_net,client_data_num):
    encoders = [client_nets[i] for i in idx_client]
    count = sum(client_data_num)
    rate = [ i/count for i in client_data_num]
    keys = [k for k in encoders[0].state_dict() if (k.find('bn')==-1 and k.find('num')==-1)]
    for k in keys:
        l = [encoders[i].state_dict()[k] * rate[i]  for i in range(len(encoders))]
        param = torch.stack(l).sum(dim=0)
        server_net.state_dict()[k].copy_(param)
    return server_net


def global_test(encoder, classifier, test_loader):
    correct = 0
    total = 0
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs = inputs.to(device)
            target = target.to(device)
            reps = encoder(inputs)
            outputs = classifier(reps)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    return acc