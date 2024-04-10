import logging
import math
import time
import numpy as np

from fl_core.trainer.model_trainer import ModelTrainer
import torch.nn as nn
from torch.cuda.amp import autocast
import torch
import copy
from torch.nn.utils import parameters_to_vector


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger
        self.mask = None
        self.update = []
        self.num_remove = None
        self.device = None

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_init_mask(self, mask):
        self.mask = mask

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_model_gradients(self):
        return torch.cat([p.grad.clone().flatten() for p in self.model.parameters() if p.grad is not None])

    def train(self, train_loader, device, round_idx, criterion, global_mask=None, neurotoxin_mask=None):
        model = self.model
        model.to(device)
        self.device = device
        initial_global_model_params = parameters_to_vector(
            [model.state_dict()[name] for name in model.state_dict()]).detach()

        for name, param in model.named_parameters():
            self.mask[name] = self.mask[name].to(device)
            param.data = param.data * self.mask[name]
        if self.num_remove != None:
            if self.id >= self.args.num_corrupt or self.args.attack != "fix_mask":
                gradient = self.screen_gradients(model, train_loader)
                self.mask = self.update_mask(self.mask, self.num_remove, gradient)

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
                for name, param in model.named_parameters():
                    param.grad.data = self.mask[name].to(device) * param.grad.data
                optimizer.step()

        if self.id < self.args.num_corrupt:
            if self.args.attack == "fix_mask":
                self.mask = self.mask

            elif self.args.attack == "omniscient":
                if len(global_mask):
                    self.mask = copy.deepcopy(global_mask)
                else:
                    self.mask = self.mask
            elif self.args.attack == "neurotoxin":
                if len(neurotoxin_mask):
                    self.mask = neurotoxin_mask
                else:
                    self.mask = self.mask
            else:
                self.mask, self.num_remove = self.fire_mask(model.state_dict(), self.mask, round_idx)

        else:
            self.mask, self.num_remove = self.fire_mask(model.state_dict(), self.mask, round_idx)

        with torch.no_grad():
            after_train = parameters_to_vector(
                [model.state_dict()[name] for name in model.state_dict()]).detach()
            array_mask = parameters_to_vector(
                [self.mask[name].to(device) for name in model.state_dict()]).detach()
            self.update = (array_mask * (after_train - initial_global_model_params))
            if "scale" in self.args.attack:
                logging.info("scale update for" + self.args.attack.split("_", 1)[1] + " times")
                if self.id < self.args.num_corrupt:
                    self.update = int(self.args.attack.split("_", 1)[1]) * self.update
        return self.update, self.mask

    def screen_gradients(self, model, train_loader):
        model.train()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        gradient = {name: 0 for name, param in model.named_parameters()}
        # # sample 10 batch  of data
        batch_num = 0
        for _, (x, labels) in enumerate(train_loader):
            batch_num += 1
            model.zero_grad()
            x, labels = x.to(self.device), labels.to(self.device)
            log_probs = model.forward(x)
            minibatch_loss = criterion(log_probs, labels.long())
            loss = minibatch_loss
            loss.backward()
            for name, param in model.named_parameters():
                gradient[name] += param.grad.data
        return gradient

    def update_mask(self, masks, num_remove, gradient=None):
        for name in gradient:
            if self.args.dis_check_gradient:
                temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]), torch.zeros_like(masks[name]))
                idx = torch.multinomial(temp.flatten().to(self.device), num_remove[name], replacement=False)
                masks[name].view(-1)[idx] = 1
            else:
                temp = torch.where(masks[name].to(self.device) == 0, torch.abs(gradient[name]),
                                   -100000 * torch.ones_like(gradient[name]))
                sort_temp, idx = torch.sort(temp.view(-1), descending=True)
                masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return masks

    def fire_mask(self, weights, masks, round):

        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / (self.args.comm_round)))

        # logging.info(drop_ratio)
        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name].to(self.device))
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)

        for name in masks:
            if num_remove[name] > 0 and "track" not in name and "running" not in name:
                temp_weights = torch.where(masks[name].to(self.device) > 0, torch.abs(weights[name]),
                                           100000 * torch.ones_like(weights[name]))
                x, idx = torch.sort(temp_weights.view(-1).to(self.device))
                masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return masks, num_remove

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
