from __future__ import print_function
import argparse
import torch
import random
from torch.autograd import Variable
from fl_api.models.resnet9 import ResNet9
import os
import copy
import numpy as np
from torch.utils.data import DataLoader
from fl_api.data_preprocessing.data import get_datasets, distribute_data_dirichlet, distribute_data, DatasetSplit, \
    poison_dataset
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter


# Training settings
def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--model', type=str, default='resnet9', metavar='N',
                        help="network architecture, supporting 'CNN_cifar10', 'MLP_mnist', 'CNN_fmnist'"
                             ", 'vgg11', 'vgg16', 'resnet9', 'resnet18', 'resnet34'")
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training, cifar10, cifar100, mnist, fmnist, tinyimagenet')
    parser.add_argument('--n_classes', type=int, default=10, metavar='N',
                        help='local batch size for training')
    parser.add_argument('--num_clients', type=int, default=40, metavar='N',
                        help='number of clients')
    parser.add_argument('--frac', type=int, default=1,
                        help="fraction of clients per round")
    parser.add_argument('--comm_round', type=int, default=200,
                        help='total communication rounds')
    parser.add_argument('--client_lr', type=float, default=0.1, metavar='LR',
                        help='learning rate, 0.0001')
    parser.add_argument('--epochs', type=int, default=2, metavar='EP',
                        help='local training epochs for each client')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--client_optimizer', type=str, default='SGD',
                        help='SGDm, Adam, SGD')
    parser.add_argument('--lr_decay', type=float, default=1, metavar='LR_decay',
                        help='learning rate decay')
    parser.add_argument('--wd', help='weight decay parameter', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='NN',
                        help='momentum')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='local batch size for training')
    parser.add_argument('--non_iid', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=0,
                        help="num of workers for multithreading")
    parser.add_argument('--target_class', type=int, default=7,
                        help="target class for backdoor attack")
    parser.add_argument('--pattern_type', type=str, default='plus',
                        help="shape of bd pattern")
    parser.add_argument('--attack', type=str, default="badnet")
    parser.add_argument('--poison_frac', type=float, default=0.5,
                        help="fraction of dataset to corrupt for backdoor attack")
    parser.add_argument('--aggr', type=str, default='avg',
                        help="aggregation function to aggregate agents' local weights")
    parser.add_argument('--num_corrupt', type=int, default=4,
                        help="number of corrupt agents")
    parser.add_argument('--server_lr', type=float, default=1, help='servers learning rate for signSGD')
    parser.add_argument('--dense_ratio', type=float, default=0.25,
                        help="num of workers for multithreading")
    parser.add_argument('--method', type=str, default="TopK",
                        help="num of workers for multithreading")
    parser.add_argument('--theta', type=int, default=0.5,
                        help="the ratio of partition")
    parser.add_argument('--topk', type=int, default=0.6,
                        help="the ratio of topk")
    parser.add_argument('--cease_poison', type=float, default=100000)

    return parser


def load_data(args):
    train_dataset, val_dataset = get_datasets(args.dataset, data_dir='../../data')

    if args.dataset == "cifar100":
        num_target = 100
    elif args.dataset == "tinyimagenet":
        num_target = 200
    else:
        num_target = 10

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    if args.non_iid:
        user_groups = distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = distribute_data(train_dataset, args, n_classes=num_target)

    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    if args.dataset != "tinyimagenet":
        # poison the validation dataset
        poisoned_val_set = DatasetSplit(copy.deepcopy(val_dataset), idxs)
        poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args)

    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=False)

    if args.dataset != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = DatasetSplit(copy.deepcopy(val_dataset), idxs)
        poison_dataset(poisoned_val_set_only_x.dataset, args, idxs, poison_all=True, modify_label=False)
    else:
        poisoned_val_set_only_x = DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args,
                                               modify_label=False)

    poisoned_val_only_x_loader = DataLoader(poisoned_val_set_only_x, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)

    return train_dataset, poisoned_val_loader, poisoned_val_only_x_loader, user_groups, val_loader

def CLP(net):
    channel_lips_conv = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            channel_lips = []
            weights_norm = []
            for idx in range(m.weight.shape[0]):
                weight = m.weight[idx]
                weight = weight.reshape(weight.shape[0], -1).cpu()
                channel_lips.append(torch.svd(weight)[1].max())
                weights_norm.append(float(torch.norm(weight)))
            channel_lips = torch.Tensor(channel_lips)
            channel_lips_conv.append(channel_lips)
    return channel_lips_conv

def model_clp(model_path):
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(0)

    channel_lips_conv = CLP(model)
    channel_lips_conv_last = channel_lips_conv[-1]
    channel_lips_conv_last = [round(num.item(), 2) for num in channel_lips_conv_last]
    channel_lips_conv_last_dict = Counter(channel_lips_conv_last)

    X = []
    y = []
    for key in sorted(list(channel_lips_conv_last_dict.keys())):
        X.append(key)
        y.append(channel_lips_conv_last_dict[key])
    return X, y


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    parser = add_args(argparse.ArgumentParser(description='Isolation'))
    args = parser.parse_args()

    train_dataset, poisoned_val_loader, poisoned_val_only_x_loader, user_groups, val_loader = load_data(args)

    model = ResNet9()
    model_path_rlr = '/home/lgz/papers/federated_learning_20231222/codes/Isolation/fl_experiments/standalone/RLR/logs/cifar10/checkpoints/RLR_AckRatio4_40_Methodrlr_datacifar10_alpha0.5_Rnd200_Epoch2_inject0.5_Aggavg_noniidFalse_maskthreshold8_attackbadnet.pt'
    model_path_isolation = '/home/lgz/papers/federated_learning_20231222/codes/Isolation/fl_experiments/standalone/Isolation/logs/cifar10/checkpoints/Isolation_AckRatio4_40_MethodTopK_datacifar10_alpha0.5_Rnd200_Epoch2_inject0.5_Aggavg_noniidFalse_maskthreshold0.5_attackbadnet_topk0.6.pt'

    X_rlr, y_rlr = model_clp(model_path_rlr)
    X_isolation, y_isolation = model_clp(model_path_isolation)
    print(X_rlr, y_rlr)
    print(X_isolation, y_isolation)

    names = sorted(list(set(X_rlr) | set(X_isolation)))
    y_rlr_1 = []
    y_isolation_1 = []
    for name in names:
        if name in X_rlr:
            ind = X_rlr.index(name)
            y_rlr_1.append(y_rlr[ind])
        else:
            y_rlr_1.append(0)
        if name in X_isolation:
            ind = X_isolation.index(name)
            y_isolation_1.append(y_isolation[ind])
        else:
            y_isolation_1.append(0)

    X = [str(name) for name in names]
    plt.bar(X, y_rlr_1, label='RLR')
    plt.bar(X, y_isolation_1, label='Isolation (Our)')
    plt.xticks(rotation=75)
    plt.ylabel('# of channels')
    plt.legend()
    # plt.xlabel('Chanel Lipschitzness')
    plt.show()
