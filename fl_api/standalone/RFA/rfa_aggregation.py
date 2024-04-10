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

    def aggregate_updates(self, client_updates_dict, client_indexes, before_train_params):
        cur_client_updates_dict = {i: client_updates_dict[i] for i in client_indexes}
        cur_global_params = parameters_to_vector(
            [params for name, params in before_train_params.items()]).detach()
        aggregated_updates = self.agg_comed(cur_client_updates_dict)
        new_global_params_vec = cur_global_params.to(self.device) + aggregated_updates
        new_global_params = vector_to_name_param(new_global_params_vec, before_train_params)
        return new_global_params


    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values

