import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from fl_api.data_preprocessing.data import DatasetSplit, poison_dataset
from fl_api.utils.general import parameters_to_vector, vector_to_name_param


class Client:

    def __init__(self, client_idx, args, device, model_trainer, train_dataset, logger, data_idxs):
        self.logger = logger
        self.client_idx = client_idx
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.poison_idxs = None

        if self.args.dataset != "tinyimagenet":
            self.train_dataset = DatasetSplit(train_dataset, data_idxs)
            if self.client_idx < args.num_corrupt:
                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                self.poison_idxs = poison_dataset(train_dataset, args, data_idxs, agent_idx=self.client_idx)
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

        mask, client_grads = self.model_trainer.train(self.train_loader, self.device, round_idx, criterion, neurotoxin_mask, self.poison_idxs)
        local_model = self.model_trainer.get_model_params()

        if "scale" in self.args.attack:
            after_train = parameters_to_vector(
                [local_model[name] for name in local_model]).detach()
            before_train = parameters_to_vector(
                [w[name] for name in w]).detach()
            after_train, before_train = after_train.to(self.args.gpu), before_train.to(self.args.gpu)
            update = after_train - before_train
            # logging.info("scale update for" + self.args.attack.split("_", 1)[1] + " times")
            if self.client_idx < self.args.num_corrupt:
                update = int(self.args.attack.split("_", 1)[1]) * update
                local_model_vec = before_train + update
                local_model = vector_to_name_param(local_model_vec, copy.deepcopy(w))

        return mask, local_model, client_grads

    def test(self, w, round):
        self.model_trainer.set_model_params(w, mask=None)
        self.model_trainer.set_id(self.client_idx)
        test_data = self.local_test_data
        metrics = self.model_trainer.test(test_data, self.device, self.args, round)
        return metrics
