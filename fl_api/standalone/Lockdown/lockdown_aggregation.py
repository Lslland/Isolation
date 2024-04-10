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
        aggregated_updates = self.agg_avg(cur_client_updates_dict)
        new_global_params_vec = cur_global_params.to(self.device) + aggregated_updates
        new_global_params = vector_to_name_param(new_global_params_vec, before_train_params)
        return new_global_params

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

