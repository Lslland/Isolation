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
        aggregated_updates = self.agg_krum(cur_client_updates_dict)
        new_global_params_vec = cur_global_params.to(self.device) + aggregated_updates
        new_global_params = vector_to_name_param(new_global_params_vec, before_train_params)
        return new_global_params

    def agg_krum(self, agent_updates_dict):
        krum_param_m = 1

        def _compute_krum_score(vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i] - vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]
        return_gradient = [agent_updates_dict[i] for i in score_index]
        return sum(return_gradient) / len(return_gradient)


