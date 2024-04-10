import copy
import torch
from torch.nn.utils import  parameters_to_vector
import logging
from fl_api.utils.general import vector_to_name_param


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

    def aggregate_updates(self, w_global, client_updates_dict):
        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.device)
        if self.args.method != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(client_updates_dict, self.args.theta)

        aggregated_updates = 0
        if self.args.aggr == 'avg':
            aggregated_updates = self.agg_avg(client_updates_dict)
        if self.args.aggr == "clip_avg":
            for _id, update in client_updates_dict.items():
                weight_diff_norm = torch.norm(update).item()
                logging.info(weight_diff_norm)
                update.data = update.data / max(1, weight_diff_norm / 2)
            aggregated_updates = self.agg_avg(client_updates_dict)
            logging.info(torch.norm(aggregated_updates))
        elif self.args.aggr == 'comed':
            aggregated_updates = self.agg_comed(client_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(client_updates_dict)
        elif self.args.aggr == "krum":
            aggregated_updates = self.agg_krum(client_updates_dict)
        # elif self.args.aggr == "gm":
        #     aggregated_updates = self.agg_gm(agent_updates_dict, cur_global_params)
        elif self.args.aggr == "tm":
            aggregated_updates = self.agg_tm(client_updates_dict)

        cur_global_params_vec = parameters_to_vector([copy.deepcopy(w_global[name]) for name in list(w_global.keys())])
        new_global_params_vec = (cur_global_params_vec.to(self.device) + lr_vector * aggregated_updates).float()
        new_global_params = vector_to_name_param(new_global_params_vec, copy.deepcopy(w_global))

        malicious_params_vec = (cur_global_params_vec.to(self.device) + client_updates_dict[0]).float()
        malicious_params = vector_to_name_param(malicious_params_vec, copy.deepcopy(w_global))
        return new_global_params, malicious_params


    def compute_robustLR(self, agent_updates_dict, theta):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask = torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < theta] = 0
        mask[sm_of_signs >= theta] = 1
        sm_of_signs[sm_of_signs < theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= theta] = self.server_lr
        return sm_of_signs.to(self.device), mask

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

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values

    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)
