import copy
import logging
import os

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from math import floor
from collections import defaultdict
import random
import math

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """

    def __init__(self, dataset, idxs, runtime_poison=False, args=None, client_id=-1, modify_label=True):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        self.runtime_poison = runtime_poison
        self.args = args
        self.client_id = client_id
        self.modify_label = modify_label
        if client_id == -1:
            poison_frac = 1
        elif client_id < self.args.num_corrupt:
            poison_frac = self.args.poison_frac
        else:
            poison_frac = 0
        self.poison_sample = {}
        self.poison_idxs = []
        if runtime_poison and poison_frac > 0:
            self.poison_idxs = random.sample(self.idxs, floor(poison_frac * len(self.idxs)))
            for idx in self.poison_idxs:
                self.poison_sample[idx] = add_pattern_bd(copy.deepcopy(self.dataset[idx][0]), None, args.data,
                                                         pattern_type=args.pattern_type, agent_idx=client_id,
                                                         attack=args.attack)
                # plt.imshow(self.poison_sample[idx].permute(1, 2, 0))
                # plt.show()


    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(target.type())
        if self.idxs[item] in self.poison_idxs:
            inp = self.poison_sample[self.idxs[item]]
            if self.modify_label:
                target = self.args.target_class
            else:
                target = self.dataset[self.idxs[item]][1]
        else:
            inp, target = self.dataset[self.idxs[item]]

        return inp, target


def distribute_data_dirichlet(dataset, args):
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    N = len(labels_sorted[1])
    K = len(labels_dict)
    # logging.info((N, K))
    client_num = args.num_clients

    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]
        for k in labels_dict:
            idx_k = labels_dict[k]

            # get a list of batch indexes which are belong to label k
            np.random.shuffle(idx_k)
            # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
            # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
            proportions = np.random.dirichlet(np.repeat(args.alpha, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # distribute data to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_clients):
        dict_users[user_idx] = idx_batch[user_idx]
        np.random.shuffle(dict_users[user_idx])

    # num = [ [ 0 for k in range(K) ] for i in range(client_num)]
    # for k in range(K):
    #     for i in dict_users:
    #         num[i][k] = len(np.intersect1d(dict_users[i], labels_dict[k]))
    # logging.info(num)
    # print(dict_users)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # # logging.info( [len(intersection (dict_users[i], dict_users[i+1] )) for i in range(args.num_clients)] )
    return dict_users


def distribute_data(dataset, args, n_classes=10):
    # logging.info(dataset.targets)
    # logging.info(dataset.classes)
    class_per_agent = n_classes

    if args.num_clients == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    # sort labels
    labels_sorted = torch.tensor(dataset.targets).sort()
    # print(labels_sorted)
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    # split indexes to shards
    shard_size = len(dataset) // (args.num_clients * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
    hey = copy.deepcopy(labels_dict)
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_clients):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1
        np.random.shuffle(dict_users[user_idx])
    # num = [ [ 0 for k in range(n_classes) ] for i in range(args.num_clients)]
    # for k in range(n_classes):
    #     for i in dict_users:
    #         num[i][k] = len(np.intersect1d(dict_users[i], hey[k]))
    # logging.info(num)
    # logging.info(args.num_clients)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # logging.info( len(intersection (dict_users[0], dict_users[1] )))

    return dict_users


def get_datasets(data, data_dir='../../../data'):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    # data_dir = '../../../data'

    if data == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
    elif data == 'cifar100':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])])
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                   std=[0.2675, 0.2565, 0.2761])])
        train_dataset = datasets.CIFAR100(data_dir,
                                          train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_dir,
                                         train=False, download=True, transform=valid_transform)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
    elif data == "tinyimagenet":
        # _data_transforms = {
        #     'train': transforms.Compose([
        #         transforms.ToTensor()
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.ToTensor()
        #     ]),
        # }
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        _data_dir = '../../../data/tiny-imagenet-200/'
        train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                             transform)
        # print(train_dataset[0][0].shape)
        test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                            test_transform)
        train_dataset.targets = torch.tensor(train_dataset.targets)
        test_dataset.targets = torch.tensor(test_dataset.targets)
    return train_dataset, test_dataset


def add_pattern_bd(x, y, dataset='cifar10', pattern_type='square', agent_idx=-1, attack="DBA"):
    """
    adds a trojan pattern to the image
    """

    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10' or dataset == "cifar100":
        x = np.array(x.squeeze())
        # logging.info(x.shape)
        row = x.shape[0]
        column = x.shape[1]

        if attack == "periodic_trigger":
            for d in range(0, 3):
                for i in range(row):
                    for j in range(column):
                        x[i][j][d] = max(min(x[i][j][d] + 20 * math.sin((2 * math.pi * j * 6) / column), 255), 0)
            # import matplotlib.pyplot as plt
            # # plt.imsave("visualization/input_images/backdoor2.png", x)
            # print(y)
            # plt.show()
        else:
            if pattern_type == 'plus':
                start_idx = 5
                size = 6
                if agent_idx == -1:
                    # vertical line
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx + size + 1):
                            if d == 2:
                                x[i, start_idx][d] = 0
                            else:
                                x[i, start_idx][d] = 255
                    # horizontal line
                    for d in range(0, 3):
                        for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                            if d == 2:
                                x[start_idx + size // 2, i][d] = 0
                            else:
                                x[start_idx + size // 2, i][d] = 255
                else:
                    if attack == "DBA":
                        # DBA attack
                        # upper part of vertical
                        if agent_idx % 4 == 0:
                            for d in range(0, 3):
                                for i in range(start_idx, start_idx + (size // 2) + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 0
                                    else:
                                        x[i, start_idx][d] = 255

                        # lower part of vertical
                        elif agent_idx % 4 == 1:
                            for d in range(0, 3):
                                for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 0
                                    else:
                                        x[i, start_idx][d] = 255

                        # left-part of horizontal
                        elif agent_idx % 4 == 2:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 0
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                        # right-part of horizontal
                        elif agent_idx % 4 == 3:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 0
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                    else:
                        # vertical line
                        for d in range(0, 3):
                            for i in range(start_idx, start_idx + size + 1):
                                if d == 2:
                                    x[i, start_idx][d] = 0
                                else:
                                    x[i, start_idx][d] = 255
                        # horizontal line
                        for d in range(0, 3):
                            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                                if d == 2:
                                    x[start_idx + size // 2, i][d] = 0
                                else:
                                    x[start_idx + size // 2, i][d] = 255

                # import matplotlib.pyplot as plt
                #
                # plt.imsave("visualization/input_images/backdoor2.png", x)
                # # print(y)
                # plt.show()

    elif dataset == 'tinyimagenet':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            # vertical line
            for d in range(0, 3):
                for i in range(start_idx, start_idx + size + 1):
                    if d == 2:
                        x[d][i][start_idx] = 0
                    else:
                        x[d][i][start_idx] = 1
            # horizontal line
            for d in range(0, 3):
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    if d == 2:
                        x[d][start_idx + size // 2][i] = 0
                    else:
                        x[d][start_idx + size // 2][i] = 1

            # if agent_idx == -1:
            #     # plt.imsave("visualization/input_images/backdoor2.png", x)
            #     print(y)
            #     plt.show()
            # plt.savefig()

    elif dataset == 'fmnist':
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            else:
                if attack == "DBA":
                    # DBA attack
                    # upper part of vertical
                    if agent_idx % 4 == 0:
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx] = 255

                    # lower part of vertical
                    elif agent_idx % 4 == 1:
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx] = 255

                    # left-part of horizontal
                    elif agent_idx % 4 == 2:
                        for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                            x[start_idx + size // 2, i] = 255

                    # right-part of horizontal
                    elif agent_idx % 4 == 3:
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i] = 255
                else:
                    # vertical line
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx] = 255
                    # horizontal line
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i] = 255

    elif dataset == 'mnist':
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 1
            size = 2
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            else:
                if attack == "DBA":
                    # DBA attack
                    # upper part of vertical
                    if agent_idx % 4 == 0:
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx] = 255

                    # lower part of vertical
                    elif agent_idx % 4 == 1:
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx] = 255

                    # left-part of horizontal
                    elif agent_idx % 4 == 2:
                        for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                            x[start_idx + size // 2, i] = 255

                    # right-part of horizontal
                    elif agent_idx % 4 == 3:
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i] = 255
                else:
                    # vertical line
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx] = 255
                    # horizontal line
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i] = 255
    # import matplotlib.pyplot as plt
    # if agent_idx == -1:
    #     # plt.imsave("visualization/input_images/backdoor2.png", x)
    #     plt.imshow(x)
    #     print(y)
    #     plt.show()
    return x

def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1, modify_label=True):
    # if data_idxs != None:
    #     all_idxs = list(set(all_idxs).intersection(data_idxs))
    if data_idxs != None:
        all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
        all_idxs = list(set(all_idxs).intersection(data_idxs))
    else:
        all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
    poison_frac = 1 if poison_all else args.poison_frac
    poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs)))
    for idx in poison_idxs:
        if args.dataset == 'fedemnist':
            clean_img = dataset.inputs[idx]
        elif args.dataset == "tinyimagenet":
            clean_img = dataset[idx][0]
        else:
            clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, dataset.targets[idx], args.dataset, pattern_type=args.pattern_type,
                                agent_idx=agent_idx, attack=args.attack)
        if args.dataset == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        elif args.dataset == "tinyimagenet":
            # don't do anything for tinyimagenet, we poison it in run time
            return
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        if modify_label:
            dataset.targets[idx] = args.target_class
    return poison_idxs