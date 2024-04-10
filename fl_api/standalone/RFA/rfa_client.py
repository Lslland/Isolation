import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from fl_api.data_preprocessing.data import DatasetSplit, poison_dataset


class Client:

    def __init__(self, client_idx, args, device, model_trainer, train_dataset, logger, data_idxs):
        self.logger = logger
        self.client_idx = client_idx
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

        if self.args.dataset != "tinyimagenet":
            self.train_dataset = DatasetSplit(train_dataset, data_idxs)
            if self.client_idx < args.num_corrupt:
                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                poison_dataset(train_dataset, args, data_idxs, agent_idx=self.client_idx)
        else:
            self.train_dataset = DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                              client_id=self.client_idx)

        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def check_poison_timing(self, round_idx):
        if round_idx > self.args.cease_poison:
            self.train_dataset = DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)


    def train(self, w, round_idx, criterion, neurotoxin_mask=None):
        if self.client_idx < self.args.num_corrupt:
            self.check_poison_timing(round_idx)

        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_model_params(w)

        updates = self.model_trainer.train(self.train_loader, self.device, round_idx, criterion, neurotoxin_mask)
        local_model = self.model_trainer.get_model_params()
        return updates, local_model

    def test(self, w, round):
        self.model_trainer.set_model_params(w, mask=None)
        self.model_trainer.set_id(self.client_idx)
        test_data = self.local_test_data
        metrics = self.model_trainer.test(test_data, self.device, self.args, round)
        return metrics
