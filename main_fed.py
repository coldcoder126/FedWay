# -*- codeing = utf-8 -*-
# @Author: 13483
# @Time: 2022/9/8 19:35
import argparse
import datetime

import torchvision
import torchvision.transforms as transforms
from methods.frame import fedavg,fedavg2,fedavg3,fed_con, fed_mutual, fed_ring, fed_oneway, fed_mpl, fed_mutual_aug, fed_mutual_aug2, fed_prox, fed_mul_aug2_1
from utils import split

OPTIMIZERS = ['fedavg', "fed_mutual"]
DATASETS = ["mnist", "cifar10", "cifar100","svhn"]
MODELS = ["cnn", "ccnn", "lenet", "vgg16", "resnet"]


def read_options():
    parser = argparse.ArgumentParser()
    # data_path和data_set联合组成数据集的路径
    parser.add_argument("--data_path", help="data path", type=str, default="../data")
    parser.add_argument("--optimizer", help="name of optimizer", type=str, choices=OPTIMIZERS, default="fedavg")
    parser.add_argument("--dataset", help="name of dataset", type=str, choices=DATASETS, default="cifar10")
    parser.add_argument("--model", help="name of model", type=str, choices=MODELS, default="cnn")
    parser.add_argument("--client_num", help="count of all clients ", type=int, default=50)
    parser.add_argument("--epoch", help="epoch", type=int, default=5)
    parser.add_argument("--class_num", help="count of all classes ", type=int)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--round_num", help="number of rounds", type=int, default=100)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.04)
    parser.add_argument("--seed", help="seed for randomness", type=int, default=1)
    parser.add_argument("--alpha", help="dirichlet parameter alpha", type=float, default=1)
    parser.add_argument("--begin_time", help="run begin time", type=str,
                        default=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    parser.add_argument("--num_workers", help="num workers in dataloader", type=int, default=2)
    parser.add_argument("--desc", help="describe of experiment", type=str, default='default experiment')
    parser.add_argument("--threshold", help="threshold of train round", type=int, default=-1)
    # 以下仅MPL算法需要
    parser.add_argument('--resize', default=32, type=int, help='resize image')
    parser.add_argument('--ratio', default=0.3, type=float, help='unlabeled data rate')

    # 以下仅fedProx算法需要
    parser.add_argument('--mu', default=0.01, type=float, help='resize image')

    # fedavg2算法需要
    parser.add_argument('--ln', default=4, type=int, help='number of layers in model to upload')
    parser.add_argument('--temp', default=0.07, type=float, help='temperature')
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

    part_data = split.dirichlet_part(args=args, trainset=train_set, alpha=alpha)

    fed_con.fed_con(args,train_set,test_set,part_data)
    # fedavg.fedavg(args, train_set, test_set, part_data)
    # fedavg2.fedavg(args, train_set, test_set, part_data)
    # fedavg3.fedavg(args, train_set, test_set, part_data)

    # fed_mutual.fed_mutual(args, train_set, test_set, part_data)
    # fed_mutual_aug.fed_mutual(args,test_set,part_data)
    fed_mul_aug2_1.fed_mutual(args,test_set,part_data)
    # fed_mutual_aug2.fed_mutual(args,test_set,part_data)
    # fed_ring.fed_ring(args,train_set,test_set,part_data)
    # fed_oneway.fed_oneway(args, train_set, test_set, part_data)
    # fed_mpl.fed_mpl(args, test_set, part_data)
    # fed_prox.fedprox(args,train_set, test_set,part_data)


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

    if args.dataset == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.507, 0.486, 0.440], std=[0.267, 0.256, 0.276])])

        train_set = torchvision.datasets.CIFAR100(root=path,
                                                  train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=path,
                                                 train=False, download=True, transform=transform)
        return train_set, test_set
    if args.dataset == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_set = torchvision.datasets.SVHN(root=path,
                                              split='train', download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(root=path,
                                             split='test', download=True, transform=transform)
        return train_set, test_set


if __name__ == "__main__":
    run_fed()
