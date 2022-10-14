# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/9/8 19:35
import argparse
import torchvision
import torchvision.transforms as transforms
from methods.frame import fedavg,fed_mutual
from utils import split

OPTIMIZERS = ['fedavg', "fed_mutual"]
DATASETS = ["mnist", "cifar10"]
MODELS = ["cnn","ccnn","lenet","vgg"]


def read_options():
    parser = argparse.ArgumentParser()
    # data_path和data_set联合组成数据集的路径
    parser.add_argument("--data_path", help="data path", type=str, default="../data")
    parser.add_argument("--optimizer", help="name of optimizer", type=str, choices=OPTIMIZERS, default="fedavg")
    parser.add_argument("--dataset", help="name of dataset", type=str, choices=DATASETS, default="mnist")
    parser.add_argument("--model", help="name of model", type=str, choices=MODELS, default="cnn")
    parser.add_argument("--client_num", help="count of all clients ",type=int,default=50)
    parser.add_argument("--epoch", help="epoch",type=int,default=5)
    parser.add_argument("--class_num", help="count of all classes ",type=int)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--round_num", help="number of rounds", type=int, default=100)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.003)
    parser.add_argument("--seed", help="seed for randomness",type=int)
    parser.add_argument("--alpha", help="dirichlet parameter alpha", type=float, default=0.2)

    try:
        parsed = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    # 设置随机种子
    # random.seed(1+parsed["seed"])
    # np.random.seed(12+parsed["seed"])
    # options = parsed.__dict__
    return parsed


def run_fed():
    args = read_options()

    # 加载数据集和划分数据
    train_set, test_set = load_loader(args)

    # 划分数据，如果划分过就直接从文件加载
    alpha = args.alpha

    part_file_name = f"data/part-file/{args.dataset}-clientNum{args.client_num}-dir{str(alpha).replace('.','_')}-seed{args.seed}-partition"

    part_data = split.dirichlet_part(args=args, trainset=train_set,alpha=alpha)

    # fedavg.fedavg(args, train_set, test_set, part_data)
    fed_mutual.fed_mutual(args,train_set,test_set,part_data)


def load_loader(args):
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
                                        transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

        train_set = torchvision.datasets.CIFAR10(root=path,
                                                 train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=path,
                                                train=False, download=True, transform=transform)
        return train_set, test_set


if __name__ == "__main__":
    run_fed()