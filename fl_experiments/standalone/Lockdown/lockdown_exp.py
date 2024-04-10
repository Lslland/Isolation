import argparse
# import os.path
import os
import sys
import copy
import numpy as np
import random

sys.path.insert(0, os.path.abspath("/home/lgz/papers/federated_learning_20231222/codes/Isolation/"))
from fl_api.standalone.Lockdown.lockdown_model_trainer import MyModelTrainer
from fl_api.standalone.Lockdown.lockdown_api import LockdownAPI
from fl_api.data_preprocessing.data import get_datasets, distribute_data_dirichlet, distribute_data, DatasetSplit, \
    poison_dataset
import torch
from fl_api.utils.logger import CompleteLogger
from torch.utils.data import DataLoader
from datetime import datetime


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--model', type=str, default='resnet9', metavar='N',
                        help="network architecture, supporting 'CNN_cifar10', 'MLP_mnist', 'CNN_fmnist', "
                             "'resnet18', 'vgg11', 'vgg16', 'resnet9', 'resnet18', 'resnet34'")
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
    parser.add_argument('--method', type=str, default="TopKnoPer",
                        help="num of workers for multithreading")
    parser.add_argument('--mask_init', type=str, default="ERK")
    parser.add_argument('--dense_ratio', type=float, default=0.25,
                        help="num of workers for multithreading")
    parser.add_argument('--anneal_factor', type=float, default=0.0001,
                        help="num of workers for multithreading")
    parser.add_argument('--dis_check_gradient', action='store_true', default=False)
    parser.add_argument('--theta', type=int, default=0.4,
                        help="the ratio of partition")
    parser.add_argument('--topk', type=int, default=0.6,
                        help="the ratio of topk")
    parser.add_argument('--cease_poison', type=float, default=100000)

    return parser


def load_data(args):
    train_dataset, val_dataset = get_datasets(args.dataset)

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


def custom_model_trainer(args, model, logger):
    return MyModelTrainer(model, args, logger)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    parser = add_args(argparse.ArgumentParser(description='Lockdown'))
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    log_path = os.path.join('./logs/', args.dataset)
    log_name = "Lockdown_AckRatio{}_{}_method{}_data{}_alpha{}_epoch{}_inject{}_agg{}_nonIID{}_theta{}_attack{}_topk{}".format(
        args.num_corrupt, args.num_clients, args.method, args.dataset, args.alpha, args.epochs,
        args.poison_frac, args.aggr, args.non_iid, args.theta, args.attack, args.topk)
    logger = CompleteLogger(log_path, log_name)
    args.client_num_per_round = int(args.num_clients * args.frac)

    print("torch version{}".format(torch.__version__))
    dataset = load_data(args)
    print("start-time: ", datetime.now())
    print("{}".format(args))
    mnt_flAPI = LockdownAPI(dataset, device, args, logger)
    mnt_flAPI.train()
    # mnt_flAPI.test(global_model=None, round='Test')
    print("end-time: ", datetime.now())
