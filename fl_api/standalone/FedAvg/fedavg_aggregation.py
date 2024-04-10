import copy
import torch
from torch.nn.utils import  parameters_to_vector
import logging
from fl_api.utils.general import vector_to_name_param
import time


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, device, writer):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0
        self.device = device
        self.common_mask = None
        self.specific_mask = None

    def aggregate_updates(self, client_params_dict, client_indexes):
        avg_global_model = self.avg_client_params(client_params_dict, client_indexes)

        return avg_global_model

    def avg_client_params(self, clients_params_dict, client_indexes):
        new_global_model = {name: torch.zeros_like(param).to(self.device) for name, param in
                            clients_params_dict[client_indexes[0]].items()}

        for name, param in new_global_model.items():
            count = 0
            for clnt_id in client_indexes:
                new_global_model[name] += clients_params_dict[clnt_id][name].to(self.device)
                count += 1
            new_global_model[name] = new_global_model[name] / count
        return new_global_model

