import copy
import torch
from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import Subset, DataLoader
from methods.client.local_train import LocalTrain
import src.models.model as md
from methods.tool import tool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 实现fedavg算法
def fed_avg_aggr_greedy(args, part_data):
    train_set, test_set = tool.get_data_set(args, True)
    path = tool.mk_path(args)
    writer_file = f"fedavg-{args.dataset}-clientNum{args.client_num}-dir{args.alpha}-seed{args.seed}-lr{args.lr}"
    writer = SummaryWriter(f"{path}/{writer_file}")
    # 所有已经分好组的训练集和测试集

    train_loaders = [DataLoader(Subset(train_set, part_data.client_dict[i]), batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True)
                     for i in range(args.client_num)]
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # 选择模型
    options = md.generate_options(args.dataset, args.model)
    options['model'] = args.model
    model = md.choose_model(options)
    model = model.to(device)
    lr = args.lr
    for item in range(1, args.round_num + 1):
        lr = tool.getLr(lr, item)

        # 每轮随机选择部分客户端
        np.random.seed(item)
        idx_users = np.random.choice(range(args.client_num), args.clients_per_round, replace=False)
        selected_params = []  # 选中客户端的模型
        selected_data_num = []  # 选中客户端的样本数量
        for k in idx_users:
            # 训练每个选到的客户端
            local = LocalTrain(args, train_loaders[k], lr)
            global_model = copy.deepcopy(model)
            param, loss = local.train(global_model)
            selected_params.append(tool.get_flat_params_from(param.parameters()))
            selected_data_num.append(train_loaders[k].sampler.num_samples)
            # print(f"Client:{k} Loss:{loss}")

        global_param = tool.aggregate_avg(selected_params, selected_data_num)
        # (20轮以后生效)训练结束后，独立聚合每个模型，如果聚合后效果超越了之前的全局模型，则更新全局模型，否则不更新
        if item > 20:
            pre_acc = tool.global_test(model, test_loader)
            cur_model = copy.deepcopy(model)
            cur_round_acc = []
            for client_param in selected_params:
                tool.set_flat_params_custom(cur_model,client_param,1)
                cur_acc = tool.global_test(cur_model,test_loader)
                cur_round_acc.append(cur_acc)

                pre_model = copy.deepcopy(model)
                tool.set_flat_params_custom(pre_model, client_param, 0.5)
                after_aggr_acc = tool.global_test(pre_model,test_loader)
                if after_aggr_acc > pre_acc:
                    model = pre_model
        else:
            tool.set_flat_params_to(model, global_param)
        # tool.set_flat_params_custom(model, global_param, 0.5)
        # 每一轮训练完，测试全局模型在全局测试集上的表现
        global_acc = tool.global_test(model, test_loader)
        print(f'FedAvg Round:{item} lr:{lr} clients:{idx_users} global_acc:{global_acc}%')
        writer.add_scalars('Loss/Epoch',
                           {'All Data': global_acc}, item)
