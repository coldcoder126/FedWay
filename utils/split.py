# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/9/16 19:34
import math
import sys

import numpy as np
import torchvision
import pandas as pd
import matplotlib.pyplot as plt

from fedlab.utils.dataset.partition import CIFAR10Partitioner, MNISTPartitioner, CIFAR100Partitioner, SVHNPartitioner


# 使用fedlab实现自定义dirichlet分布，并将结果保存为文件，下次加载即可使用
from fedlab.utils.functional import partition_report


def dirichlet_part(args, trainset, alpha):
    num_clients = args.client_num
    seed = args.seed
    dataset = args.dataset
    if dataset == "cifar10":
        hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                             num_clients,
                                             balance=None,
                                             partition="dirichlet",
                                             dir_alpha=alpha,
                                             verbose=False,
                                             seed=seed)
    if dataset == "mnist":
        hetero_dir_part = MNISTPartitioner(trainset.targets,
                                           num_clients,
                                           partition="noniid-labeldir",
                                           dir_alpha=alpha,
                                           seed=seed)
    if dataset == "cifar100":
        hetero_dir_part = CIFAR100Partitioner(trainset.targets,
                                             num_clients,
                                             balance=None,
                                             partition="dirichlet",
                                             dir_alpha=alpha,
                                             verbose=False,
                                             seed=seed)
    if dataset == "svhn":
        hetero_dir_part = SVHNPartitioner(trainset.labels,
                                          num_clients,
                                          partition="noniid-labeldir",
                                          dir_alpha=alpha,
                                          verbose=False,
                                          seed=seed)
    return hetero_dir_part


# def load_dict(path):
#     return load_dict(path)
#
#
# def part_show(trainset, part, num_classes):
#     hist_color = '#4169E1'
#     csv_file = "cifar10_hetero_dir_0.3_100clients.csv"
#     partition_report(trainset.targets, part.client_dict,
#                      class_num=num_classes,
#                      verbose=False, file=csv_file)
#
#     hetero_dir_part_df = pd.read_csv(csv_file, header=1)
#     hetero_dir_part_df = hetero_dir_part_df.set_index('client')
#     col_names = [f"class{i}" for i in range(num_classes)]
#     for col in col_names:
#         hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
#
#     hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
#     plt.tight_layout()
#     plt.xlabel('sample num')
#     plt.savefig(f"cifar10_hetero_dir_0.3_100clients.png", dpi=400)


def split_test():
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root="D:\\WorkSpace\\Pycharm\\data\\cifar10", train=True, download=True)
    hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                         num_clients=20,
                                         balance=None,
                                         partition="dirichlet",
                                         dir_alpha=0.05,
                                         verbose=False,
                                         seed=1)
    csv_file="D:\\WorkSpace\\Pycharm\\FedWay\\utils\\cifar10_blc_none_005_20clients.csv"
    partition_report(trainset.targets, hetero_dir_part.client_dict,
                     class_num=num_classes,
                     verbose=False, file=csv_file)

    hetero_dir_part_df = pd.read_csv(csv_file, header=1)
    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(num_classes)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

    hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(f"cifar10_blc_none_005_20clients.png", dpi=400)


if __name__ == '__main__':
    split_test()

    # x = np.array(range(20))
    # print(x.shape[0])
    # np.random.seed(1)
    # y = np.random.choice(x, 7, replace=False)
    # z = np.setdiff1d(x, y, assume_unique=True)
    # print(x)
    # print(y)
    # print(z)

