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
        aggregated_updates = self.agg_fltrust(cur_client_updates_dict)
        new_global_params_vec = cur_global_params.to(self.device) + aggregated_updates
        new_global_params = vector_to_name_param(new_global_params_vec, before_train_params)
        return new_global_params

    def agg_fltrust(self, agent_updates_dict):
        # presume client 0 as the root dataset holder
        # print(agent_updates_dict)
        average_update = self.compute_robusttrust(agent_updates_dict, 0)
        return average_update

    def compute_robusttrust(self, agent_updates, id):
        total_TS = 0
        TSnorm = {}
        keys = list(agent_updates.keys())
        avg_clients_updates = torch.zeros_like(agent_updates[keys[0]])
        for key in keys:
            avg_clients_updates += agent_updates[key]
        avg_clients_updates = avg_clients_updates / len(agent_updates)

        for key in agent_updates:
            if id != key:
                update = agent_updates[key]
                # TS = torch.dot(update, agent_updates[id]) / (torch.norm(update) * torch.norm(agent_updates[id]))
                TS = torch.dot(update, avg_clients_updates) / (torch.norm(update) * torch.norm(avg_clients_updates))
                if TS < 0:
                    TS = 0
                total_TS += TS
                # logging.info(TS)
                # norm = torch.norm(agent_updates[id]) / torch.norm(update)
                norm = torch.norm(avg_clients_updates) / torch.norm(update)
                TSnorm[key] = TS * norm
        average_update = 0

        for key in agent_updates:
            if id != key:
                average_update += TSnorm[key] * agent_updates[key]
        average_update /= (total_TS + 1e-6)
        return average_update

