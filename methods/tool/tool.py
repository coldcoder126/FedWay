# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/10/13 22:50
# 对参数的公共操作
import math
import os
import sys

import torch
import numpy as np
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from methods.tool.augmentation import RandAugmentCIFAR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 获取每个模型的参数(一维)
def get_flat_params_from(parameters):
    params = []
    for param in parameters:
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


# 更新模型中的参数
def set_flat_params_to(model, flat_params):
    prev_ind = 0
    model = model.to(device)
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def set_flat_params_custom(model, flat_params,percent):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()) * percent + param.data * (1-percent) )
        prev_ind += flat_size

# def net_avg(net1,nets,rate):
#     keys = [k for k in net1[0].state_dict() if (k.find('bn') == -1 and k.find('num') == -1)]
#     for k in keys:
#         l = [net1.state_dict()[k] * rate for i in range(len(encoders))]
#         param = torch.stack(l).sum(dim=0)
#         server_net.state_dict()[k].copy_((param + server_net.state_dict()[k]) / 2)



# 更新模型的指定层参数
def set_layer_to_model(args,layer_tensor, model):
    layers = list(model.parameters())[-args.ln:]
    prev_ind = 0
    for layer in layers:
        if len(layer.shape)==2:
            layer_size = layer.shape[0]*layer.shape[1]
            layer.data.copy_(layer_tensor[prev_ind:layer_size + prev_ind].view(layer.shape[0],-1))
        else:
            layer_size = layer.size()[0]
            layer.data.copy_(layer_tensor[prev_ind:layer_size+prev_ind])
        prev_ind += layer_size


# 将net1的最高p层替换为net2中的最高p层
def replace_layers(args, net1, net2):

    layers1 = list(net1.parameters())[-args.ln:]
    layers2 = list(net2.parameters())[-args.ln:]
    for i in range(len(layers1)):
        layers1[i].data = layers2[i].data
    print("done")





def global_test(model, test_loader):
    correct = 0
    total = 0
    model = model.to(device)
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    return acc


def getLr(lr, cur_round):
    if cur_round%10==0:
        lr = lr * 0.8
    if lr < 0.0001:
        lr = 0.0001
    return lr

def get_model_path(args,idx):
    net_name = args.model
    dataset_name = args.dataset
    client_num = args.client_num
    seed = args.seed
    alpha = args.alpha
    model_path = f'saved_models/{dataset_name}/{net_name}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = f'saved_models/{dataset_name}/{net_name}/client{idx}-{client_num}-{seed}-'+str(alpha).replace('.','_')+'.pth'
    return model_path

# ----------------处理 data_set和data_loader 的类和方法---------------

mean_cifar10 = (0.491, 0.482, 0.446)
std_cifar10 = (0.202, 0.199, 0.201)
mean_cifar100 = (0.507, 0.486, 0.440)
std_cifar100 = (0.267, 0.256, 0.276)
mean_svhn = (0.485, 0.456, 0.406)
std_svhn = (0.229, 0.224, 0.225)


def get_data_set(args,trans=False):
    path = f"{args.data_path}/{args.dataset}"
    if args.dataset == "mnist":
        train_set = torchvision.datasets.MNIST(path, train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))
        test_set = torchvision.datasets.MNIST(path, train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ]))
        return train_set, test_set
    if args.dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean_cifar10, std=std_cifar10)])
        if trans:
            train_set = torchvision.datasets.CIFAR10(root=path,train=True, download=True,transform=transform)
        else:
            train_set = torchvision.datasets.CIFAR10(root=path,train=True, download=True)
        test_set = torchvision.datasets.CIFAR10(root=path,train=False, download=True, transform=transform)
        return train_set, test_set

    if args.dataset == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean_cifar100, std=std_cifar100)])
        if trans:
            train_set = torchvision.datasets.CIFAR100(root=path,train=True, download=True, transform=transform)
        else:
            train_set = torchvision.datasets.CIFAR100(root=path, train=True, download=True)
        test_set = torchvision.datasets.CIFAR100(root=path,train=False, download=True, transform=transform)
        return train_set, test_set
    if args.dataset == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean_svhn, std=std_svhn)])
        if trans:
            train_set = torchvision.datasets.SVHN(root=path,split='train', download=True, transform=transform)
        else:
            train_set = torchvision.datasets.SVHN(root=path, split='train', download=True)
        test_set = torchvision.datasets.SVHN(root=path,split='test', download=True, transform=transform)
        return train_set, test_set

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class TwoTransforms:
    """Create two crops of the same image"""
    def __init__(self, transform1,transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

# 放入subset，生成原始数据和随机增强的数据
def get_local_loader(args,train_set):
    # construct data loader
    if args.dataset == 'cifar10':
        mean = mean_cifar10
        std = std_cifar10
    elif args.dataset == 'cifar100':
        mean = mean_cifar100
        std = std_cifar100
    elif args.dataset == 'path':
        mean = eval(args.mean)
        std = eval(args.std)
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    ori_transform = transforms.Compose([transforms.ToTensor(), normalize])

    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.resize, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    mpl_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                         padding=int(32 * 0.125),
                         fill=128,
                         padding_mode='constant'),
            RandAugmentCIFAR(n=2, m=10),
            transforms.ToTensor(),
            normalize])

    if args.dataset == 'cifar10':
        train_dataset = MyDataset(train_set,transform=TwoTransforms(ori_transform,mpl_transform))
        # train_dataset = datasets.CIFAR10(root=path,
        #                                  transform=TwoTransforms(train_transform),
        #                                  download=True)
    elif args.dataset == 'cifar100':
        train_dataset = MyDataset(train_set,transform=TwoTransforms(ori_transform,mpl_transform))
    else:
        raise ValueError(args.dataset)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,drop_last=True)
    return train_loader


# 对模型进行聚合
def aggregate_avg(flat_params,selected_data_num):
    """Aggregate local solutions and output new global parameter

    Args:
        flat_params: a generator or (list) with element (num_sample, local_solution)

    Returns:
        flat global model parameter
    """

    averaged_solution = torch.zeros_like(flat_params[0])
    # averaged_solution = np.zeros(self.latest_model.shape)

    # 简单平均
    # num = 0
    # for local_solution in flat_params:
    #     num += 1
    #     averaged_solution += local_solution
    # averaged_solution /= num

    # 按照模型中的数据量平均
    num = 0
    for i in selected_data_num:
        num += i
    for j in range(len(flat_params)):
        averaged_solution += flat_params[j] * (selected_data_num[j]/num)

    # for num_sample, local_solution in flat_params:
    #     averaged_solution += num_sample * local_solution
    # averaged_solution /= self.all_train_data_num

    # averaged_solution = from_numpy(averaged_solution, self.gpu)
    return averaged_solution.detach()

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def mk_path(args):
    path = f"{sys.path[0]}/{args.data_path}/run_result/{args.begin_time}-{args.desc}/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path
