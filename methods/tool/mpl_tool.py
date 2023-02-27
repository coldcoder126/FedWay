# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/29 9:45
import time
from random import random

import PIL
import numpy as np
import math

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.transforms as T
from PIL.Image import Image
from torch.utils.data import Subset, DataLoader, Dataset
from methods.tool.augmentation import RandAugmentCIFAR


def x_u_split(labels, num_labeled, num_classes, expand_labels, batch_size, eval_step):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled

    if expand_labels or num_labeled < batch_size:
        num_expand_x = math.ceil(batch_size * eval_step / num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


cifar10_mean = (0.491400, 0.482158, 0.4465231)
cifar10_std = (0.247032, 0.243485, 0.2615877)
cifar100_mean = (0.507075, 0.486549, 0.440918)
cifar100_std = (0.267334, 0.256438, 0.276151)
svhn_mean = (0.485, 0.456, 0.406)
svhn_std = (0.229, 0.224, 0.225)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


# ratio无标签数据占数据集的占比
# 策略：随机分
# def l_u_split(args, all_train_set, train_loader, ratio):
#     all = train_loader.dataset.indices
#     sum = all.shape[0]
#     np.random.seed(args.seed)
#
#     unlabeled_idx = np.random.choice(all, int(math.floor(sum*ratio)), replace=False)
#     labeled_idx = np.setdiff1d(all, unlabeled_idx, assume_unique=True)
#
#     transform_labeled = transforms.Compose([
#         transforms.RandomHorizontalFlip(), # 旋转和翻转
#         transforms.RandomCrop(size=args.resize,  # 从图片中随机裁剪出尺寸为 size 的图片
#                               padding=int(args.resize * 0.125),
#                               fill=128,
#                               padding_mode='constant'),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
#     ])
#
#
#     unlabeled_loader = DataLoader(dataset=Subset(all_train_set, unlabeled_idx),
#                                   transforms = TransformMPL(resize=32,mean=cifar10_mean,std=cifar10_std),
#                                   sampler=train_loader,
#                                   batch_size=args.batch_size, shuffle=True)
#     labeled_loader = DataLoader(Subset(all_train_set, labeled_idx),
#                                 transforms = transform_labeled,
#                                 batch_size=args.batch_size, shuffle=True)
#     return labeled_loader, unlabeled_loader

# train_set_l是全部有标签数据集，train_set_u是全部无标签数据集,i是第i个客户端
# 返回每个客户端上有标签数据集和无标签数据集
def l_u_split(args, train_set_l, train_set_u, part_data, i):
    # 客户端i的所有数据集索引
    client_i_data_set_index = part_data.client_dict[i]
    # sum = client_i_data_set_index.shape[0]
    # np.random.seed(args.seed)
    # unlabeled_idx = np.random.choice(client_i_data_set_index, int(math.floor(sum * args.ratio)), replace=False)
    # labeled_idx = np.setdiff1d(client_i_data_set_index, unlabeled_idx, assume_unique=True)
    client_i_labeled_loader = DataLoader(dataset=Subset(train_set_l, client_i_data_set_index),
                                         batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    client_i_unlabeled_loader = DataLoader(dataset=Subset(train_set_u, client_i_data_set_index),
                                           batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    return client_i_labeled_loader, client_i_unlabeled_loader


# 获取有/无标签全体数据集
def get_train_set(args):
    path = f"{args.data_path}/{args.dataset}"
    # 如果是有标签数据集
    if args.dataset == "cifar10":
        transform_labeled = T.Compose([
            T.RandomHorizontalFlip(),  # 旋转和翻转
            T.RandomCrop(size=args.resize,
                         padding=int(args.resize * 0.125),
                         fill=128,
                         padding_mode='constant'),
            T.ToTensor(),
            T.Normalize(mean=cifar10_mean, std=cifar10_std),
        ])
        train_set_labeled = torchvision.datasets.CIFAR10(root=path,
                                                         train=True, download=True, transform=transform_labeled)
        train_set_unlabeled = torchvision.datasets.CIFAR10(root=path,
                                                           train=True, download=True,
                                                           transform=TransformMPL(resize=args.resize, mean=cifar10_mean,
                                                                                  std=cifar10_std))
        return train_set_labeled, train_set_unlabeled
    if args.dataset == "cifar100":
        transform_labeled = T.Compose([
            T.RandomHorizontalFlip(),  # 旋转和翻转
            T.RandomCrop(size=args.resize,
                         padding=int(args.resize * 0.125),
                         fill=128,
                         padding_mode='constant'),
            T.ToTensor(),
            T.Normalize(mean=cifar100_mean, std=cifar100_std),
        ])
        train_set_labeled = torchvision.datasets.CIFAR100(root=path,
                                                          train=True, download=True, transform=transform_labeled)
        train_set_unlabeled = torchvision.datasets.CIFAR100(root=path,
                                                            train=True, download=True,
                                                            transform=TransformMPL(resize=args.resize,
                                                                                   mean=cifar100_mean,
                                                                                   std=cifar100_std))
        return train_set_labeled, train_set_unlabeled
    if args.dataset == "svhn":
        transform_labeled = T.Compose([
            #T.RandomHorizontalFlip(),  # 旋转和翻转
            T.RandomAffine(degrees=0, translate=(0, 0), shear=45),
            T.RandomCrop(size=args.resize,
                         padding=int(args.resize * 0.125),
                         fill=128,
                         padding_mode='constant'),
            T.ToTensor(),
            T.Normalize(mean=svhn_mean, std=svhn_std),
        ])
        train_set_labeled = torchvision.datasets.SVHN(root=path,
                                                      split='train', download=True, transform=transform_labeled)
        train_set_unlabeled = torchvision.datasets.SVHN(root=path,
                                                        split='train', download=True,
                                                        transform=TransformMPL(resize=args.resize, mean=svhn_mean,
                                                                               std=svhn_std))
        return train_set_labeled, train_set_unlabeled


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TransformMPL(object):
    def __init__(self, resize, mean, std):
        # 是否是随机增强
        # if randaug:
        #     n, m = randaug
        # else:
        #     n, m = 2, 10  # default
        n, m = 2, 10
        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            T.RandomCrop(size=resize,
                         padding=int(resize * 0.125),
                         fill=128,
                         padding_mode='constant')])
        self.aug = transforms.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=resize,
                         padding=int(resize * 0.125),
                         fill=128,
                         padding_mode='constant'),
            RandAugmentCIFAR(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_train_set_aug(args):
    path = f"{args.data_path}/{args.dataset}"
    # 如果是有标签数据集
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
    transform_aug = T.Compose([
        transforms.RandomResizedCrop(size=args.resize, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])

    transform_aug2 = T.Compose([
        T.RandomHorizontalFlip(),  # 旋转和翻转
        T.RandomCrop(size=args.resize,
                     padding=int(args.resize * 0.125),
                     fill=128,
                     padding_mode='constant'),
        T.ToTensor(),
        T.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])


    train_set = torchvision.datasets.CIFAR10(root=path,
                                             train=True, download=True, transform=transform_aug)
    train_set_aug = torchvision.datasets.CIFAR10(root=path,
                                                 train=True, download=True, transform=transform_aug)
    return train_set, train_set_aug


def get_local_train_set_aug(args):
    path = f"{args.data_path}/{args.dataset}"
    # 如果是有标签数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
    transform_aug = T.Compose([
        T.RandomHorizontalFlip(),  # 旋转和翻转
        T.RandomCrop(size=args.resize,
                     padding=int(args.resize * 0.125),
                     fill=128,
                     padding_mode='constant'),
        T.ToTensor(),
        T.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    train_set = torchvision.datasets.CIFAR10(root=path,
                                             train=True, download=True, transform=transform)
    train_set_aug = torchvision.datasets.CIFAR10(root=path,
                                                 train=True, download=True, transform=transform_aug)
    return train_set, train_set_aug




def get_train_set_aug2(args):
    path = f"{args.data_path}/{args.dataset}"
    # 如果是有标签数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
    transform_aug = T.Compose([
        T.RandomHorizontalFlip(),  # 旋转和翻转
        T.RandomCrop(size=args.resize,
                     padding=int(args.resize * 0.125),
                     fill=128,
                     padding_mode='constant'),
        T.ToTensor(),
        T.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    train_set = torchvision.datasets.CIFAR10(root=path,
                                             train=True, download=True, transform=transform)
    train_set_aug = torchvision.datasets.CIFAR10(root=path,
                                                 train=True, download=True, transform=TransformMPL(resize=args.resize, mean=cifar10_mean,
                                                                                  std=cifar10_std))
    return train_set, train_set_aug

class MyTensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        data_tensor = self.data_tensor[index].view(3,32,32)
        if torch.is_tensor(data_tensor):
            data_tensor = transforms.ToPILImage()(data_tensor)
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        if self.target_tensor is None:
            return data_tensor
        if not torch.is_tensor(data_tensor):
            data_tensor = transforms.ToTensor()(data_tensor)

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

