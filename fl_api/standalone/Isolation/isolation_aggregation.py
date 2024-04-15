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

    def aggregate_updates(self, client_topk_mask_dict, client_params_dict, client_indexes, round_idx, client_grads_dict):
        common_mask, specific_mask = self.get_mask(client_topk_mask_dict, client_indexes)
        avg_global_model = self.avg_client_params(client_params_dict, client_indexes)

        common_params = self.get_params_based_mask(common_mask, avg_global_model)

        specific_params = {client_idx: {} for client_idx in client_indexes}
        personal_params_dict = {client_idx: {} for client_idx in client_indexes}
        for client_idx in client_indexes:
            spe_params = self.get_params_based_mask(specific_mask[client_idx], client_params_dict[client_idx])
            for name, params in common_params.items():
                if self.args.method == 'TopKnoPer' and round_idx>99:
                    specific_params[client_idx][name] = common_params[name]
                else:
                    specific_params[client_idx][name] = common_params[name] + spe_params[name]
                    personal_params_dict[client_idx][name] = spe_params[name]

        return specific_params, common_params

    def get_mask(self, client_topk_mask_dict, client_indexes):
        common_mask, specific_mask = {}, {i: {} for i in client_indexes}
        for name in client_topk_mask_dict[client_indexes[0]].keys():
            mask_list = [client_topk_mask_dict[k][name] for k in client_topk_mask_dict.keys()]
            concatenated_tensor = torch.cat(mask_list)
            unique_elements, counts = torch.unique(concatenated_tensor, return_counts=True)

            # common_mask[name] = [unique_elements[i] for i in range(len(counts)) if counts[i] != 1]
            threshold = int(self.args.theta * len(client_indexes))
            indices = (counts > threshold).nonzero().squeeze()
            common_mask[name] = unique_elements[indices]

            for i in client_indexes:
                client_mask = client_topk_mask_dict[i][name]
                specific_mask[i][name] = client_mask[~torch.isin(client_mask, common_mask[name])]
                # specific_mask[i][name] = torch.tensor(list(set(client_topk_mask_dict[i][name]) - set(common_mask[name])))

        return common_mask, specific_mask


    def get_params_based_mask(self, masks, params_dict):
        new_params = {}
        for name, params in params_dict.items():
            params = params.to(self.device)
            new_params[name] = torch.zeros_like(params)
            new_params[name].flatten()[masks[name]] = params.flatten()[masks[name]]
        return new_params

    def get_params_based_mask_to_list(self, masks, params_dict):
        new_params = {}
        for name, params in params_dict.items():
            params = params.to(self.device)
            new_params[name] = params.flatten()[masks[name]]
        return new_params


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

    def get_topk_params(self, model_params, k):
        mask = {}
        local_model = {}
        for name, param in model_params.items():
            n = param.numel()

            k1 = min(int(n * k), n)  # 当前层参数的看k%
            maxIndices = torch.topk(torch.abs(param.flatten()), k=k1).indices
            mask[name] = maxIndices
            local_model[name] = param.flatten()[maxIndices]
        return mask, local_model

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
