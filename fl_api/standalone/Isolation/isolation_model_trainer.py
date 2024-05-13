import logging
import math
import time
from collections import Counter
import torch.nn.functional as F
import numpy as np

from fl_api.utils.general import CLP
from fl_core.trainer.model_trainer import ModelTrainer
import torch.nn as nn
from torch.cuda.amp import autocast
import torch
import copy
from torch.nn.utils import parameters_to_vector


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None, mask=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger
        self.mask = copy.deepcopy(mask)
        self.update = []

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_model_gradients(self):
        return torch.cat([p.grad.clone().flatten() for p in self.model.parameters() if p.grad is not None])

    def train(self, train_loader, device, round_idx, criterion, neurotoxin_mask, poison_idxs):
        model = self.model
        model.to(device)
        initial_global_model_params = parameters_to_vector(
            [model.state_dict()[name] for name in model.state_dict()]).detach()

        lr = self.args.client_lr * (self.args.lr_decay ** round_idx)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=self.args.wd)
        model.train()

        for _ in range(self.args.epochs):
            for _, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=device, non_blocking=True), \
                    labels.to(device=device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in model.named_parameters():
                        param.grad.data = neurotoxin_mask[name].to(device) * param.grad.data
                # if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                #     for name, param in model.named_parameters():
                #         param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data
        # print(model)
        mask = self.get_topk_mask(model, self.args.topk)
        # with torch.no_grad():
        #     after_train = parameters_to_vector(
        #         [model.state_dict()[name] for name in model.state_dict()]).detach()
        #     self.update = after_train - initial_global_model_params

        # channel_lips_conv_last = CLP(model.res2[1][0].weight)
        # channel_lips_conv_last = [round(num.item(), 2) for num in channel_lips_conv_last]
        # channel_lips_conv_last_dict = Counter(channel_lips_conv_last)
        # channel_lips_dict = {}
        # if poison_idxs is not None:
        #     channel_lips_dict[f'poison_{self.id}'] = channel_lips_conv_last_dict
        # else:
        #     channel_lips_dict[f'benign_{self.id}'] = channel_lips_conv_last_dict
        # print(channel_lips_dict)
        # computing the channel lipschitzness of last convolutional layer -- end

        client_grads = {}
        for name, param in model.named_parameters():
            # print(name)
            client_grads[name] = param.grad
        # print(model.conv1[0].weight)
        # print(model)

        return mask, client_grads

    def get_topk_mask(self, model, k):
        mask = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                n = param.numel()
                k1 = min(int(n * k), n)  # computing the k of i-th layer
                # selecting the topk params of i-th layer based on gradient
                maxIndices = torch.topk(torch.abs(param.grad.flatten()), k=k1).indices
                mask[name] = maxIndices
        return mask

    def test(self, test_data, device, args, round, k):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'communication round': round,
            'client_index': self.id,
            'server_to_client_k': k,
            'test_correct': 0,
            'test_acc': 0.0,
            'test_loss': 0,
            'test_total': 0
        }
        # print('test=========')
        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                # correct = predicted.eq(target).sum()
                # _, predicted = torch.max(pred.data, 1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        return metrics
