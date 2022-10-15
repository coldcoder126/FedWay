# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/9/16 19:34
import sys
import torchvision
import pandas as pd
import matplotlib.pyplot as plt

from fedlab.utils.dataset.partition import CIFAR10Partitioner, MNISTPartitioner
from fedlab.utils.functional import partition_report, save_dict, load_dict


# 使用fedlab实现自定义dirichlet分布，并将结果保存为文件，下次加载即可使用
def dirichlet_part(args, trainset, alpha):
    num_clients = args.client_num
    seed = args.seed
    dataset = args.dataset
    filename = f"{sys.path[0]}/data/part-file/{args.dataset}-clientNum{args.client_num}-dir{str(alpha).replace('.','_')}-seed{args.seed}-partition"
    if dataset == "cifar10":
        hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                             num_clients,
                                             balance=None,
                                             partition="dirichlet",
                                             dir_alpha=alpha,
                                             verbose=False,
                                             seed=seed)
        save_dict(hetero_dir_part, filename)
    if dataset == "mnist":
        hetero_dir_part = MNISTPartitioner(trainset.targets,
                                           num_clients,
                                           partition="noniid-labeldir",
                                           dir_alpha=alpha,
                                           seed=seed)
        save_dict(hetero_dir_part, filename)
    if dataset == "cifar100":
        hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                             num_clients,
                                             balance=None,
                                             partition="dirichlet",
                                             dir_alpha=alpha,
                                             seed=seed)
        save_dict(hetero_dir_part, filename)
    return hetero_dir_part


def load_dict(path):
    return load_dict(path)


def part_show(trainset, part, num_classes):
    hist_color = '#4169E1'
    csv_file = "cifar10_hetero_dir_0.3_100clients.csv"
    partition_report(trainset.targets, part.client_dict,
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
    plt.savefig(f"cifar10_hetero_dir_0.3_100clients.png", dpi=400)


def split_test():
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root="D:\WorkSpace\Pycharm\data\cifar10", train=True, download=True)
    hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                         num_clients=50,
                                         balance=None,
                                         partition="dirichlet",
                                         dir_alpha=0.5,
                                         seed=1)

    # dict_path = "D:\WorkSpace\Pycharm\FedAvg\data\part-file\dict"
    #dict
    # dict = load_dict(dict_path)
    # print(dict)
    # save_dict(hetero_dir_part,dict_path)
    csv_file = "D:\WorkSpace\Pycharm\FedWay\cifar10_hetero_dir_0_5_50clients.csv"
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
    plt.savefig(f"cifar10_hetero_dir_0_5_50clients.png", dpi=400)


if __name__ == '__main__':
    split_test()
