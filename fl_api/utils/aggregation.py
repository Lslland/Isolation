import copy

import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
import logging
from utils import vector_to_model
import utils

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer, criterion=None):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0
        self.krum_update_cache = None
        self.return_gradient_cache =None
        self.criterion = criterion
    
    
    def aggregate_updates(self, client_id,  nei_indexs, after_train_params, before_train_params, global_model, round, agents= None):
        # logging.info(nei_indexs)
        if self.args.aggr =="avg":
            weight_dict ={}
            for _id, weight in enumerate(after_train_params):
                if _id in nei_indexs or _id == client_id:
                    weight_dict[_id] = weight
            average_weights =   self.decentralized_avg(weight_dict)
        elif self.args.aggr =="krum" or self.args.aggr =="rlr" or self.args.aggr =="fltrust" or self.args.aggr =="grad_aggr" or self.args.aggr =="bulyan":
            if round>1:
                updates = {}
                for _id, weight in enumerate(after_train_params):
                    if _id in nei_indexs or _id == client_id:
                        updates[_id] = weight  - before_train_params[client_id]
        if self.args.aggr =="krum":
            if round>1:
                if self.args.topology !="full":
                    krum_update,return_gradient = self.agg_krum(updates)
                else:
                    # for full topology, we reuse the update compute by other clients to save computation
                    if self.krum_update_cache==None:
                        self.krum_update_cache, return_gradient = self.agg_krum(updates)
                        krum_update= self.krum_update_cache 
                    else:
                        krum_update= self.krum_update_cache
                average_weights = krum_update+before_train_params[client_id]
            else:
                # logging.info("hi")
                weight_dict ={}
                for _id, weight in enumerate(after_train_params):
                    if _id in nei_indexs or _id == client_id:
                        weight_dict[_id] = weight
                average_weights =   self.decentralized_avg(weight_dict)
        elif self.args.aggr == "rlr":
            if round>1:
                # logging.info(updates)
                self.args.theta = len(updates)*0.5
                lr, _= self.compute_robustLR(updates)
                average_update =   self.decentralized_avg(updates )
                average_weights = lr* average_update+before_train_params[client_id]
            else:
                # logging.info("hi")
                weight_dict ={}
                for _id, weight in enumerate(after_train_params):
                    if _id in nei_indexs or _id == client_id:
                        weight_dict[_id] = weight
                average_weights =   self.decentralized_avg(weight_dict)
        elif self.args.aggr == "grad_aggr":
            if round>1:
                # logging.info(updates)
                average_update =   self.decentralized_avg(updates )
                average_weights =  average_update+before_train_params[client_id]
            else:
                # logging.info("hi")
                weight_dict ={}
                for _id, weight in enumerate(after_train_params):
                    if _id in nei_indexs or _id == client_id:
                        weight_dict[_id] = weight
                average_weights =   self.decentralized_avg(weight_dict)
                
        elif self.args.aggr == "fltrust":
            # logging.info(updates)
            if round>1:
                # logging.info("fuck")
                average_update= self.compute_robusttrust(updates, client_id )
                average_weights = average_update+before_train_params[client_id]
            else:
                # logging.info("hi")
                weight_dict ={}
                for _id, weight in enumerate(after_train_params):
                    if _id in nei_indexs or _id == client_id:
                        weight_dict[_id] = weight
                average_weights =   self.decentralized_avg(weight_dict)
        elif self.args.aggr == "bulyan":
            if round>1:
                if self.args.topology !="full":
                    krum_updatem, return_gradient = self.agg_krum(updates,3)
                else:
                    # for full topology, we reuse the update compute by other clients to save computation
                    if self.return_gradient_cache==None:
                        self.krum_update_cache, self.return_gradient_cache = self.agg_krum(updates,3)
                        return_gradient= self.return_gradient_cache 
                    else:
                        return_gradient= self.return_gradient_cache
                aggr_res = self.TM(return_gradient,1)
                average_weights = aggr_res+before_train_params[client_id]
            else:
                # logging.info("hi")
                weight_dict ={}
                for _id, weight in enumerate(after_train_params):
                    if _id in nei_indexs or _id == client_id:
                        weight_dict[_id] = weight
                average_weights =   self.decentralized_avg(weight_dict)
        elif self.args.aggr == "ironforge":
            weight_dict ={}
            model_acc= []
            # ironforge need to evluate model performance
            test_model = copy.deepcopy(global_model)
            test_model.to(self.args.device)
            for client in nei_indexs:
                state_dict = utils.vector_to_model(copy.deepcopy(after_train_params[client]), test_model)
                test_model.load_state_dict(state_dict)
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(test_model, self.criterion, agents[client_id].valid_loader,
                                                                                        self.args, 0, 10)
                model_acc +=[val_acc]
            # print(model_acc)
            
            
            index = np.argsort(model_acc)[-int(len(nei_indexs)*0.5):]
            top_k_id = np.array(nei_indexs)[index]
            # print(top_k_id)
            for _id, weight in enumerate(after_train_params):
                if  (_id in top_k_id  and _id in nei_indexs) or _id == client_id:
                        weight_dict[_id] = weight
            average_weights =   self.decentralized_avg(weight_dict)
            
        average_update = average_weights-before_train_params[client_id]
        neurotoxin_mask = {}
        updates_dict = vector_to_model(average_update, global_model)
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))
        return    None, neurotoxin_mask, average_weights
    
    def compute_robusttrust(self, agent_updates, id):
        total_TS = 0
        TSnorm = {}
        for key in agent_updates:
            if id!=key:
                update = agent_updates[key]
                TS = torch.dot(update,agent_updates[id])/(torch.norm(update)*torch.norm(agent_updates[id]))
                if TS < 0:
                    TS = 0
                total_TS += TS
                # logging.info(TS)
                norm = torch.norm(agent_updates[id])/torch.norm(update)
                TSnorm[key] = TS*norm
        average_update =  0
        
        for key in agent_updates :
            if id!=key:
                average_update += TSnorm[key]*agent_updates[key]
        average_update /= (total_TS + 1e-6)
        return average_update
    
    def compute_robustLR(self, agent_updates):

        agent_updates_sign = [torch.sign(update) for update in agent_updates.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs, mask

    def decentralized_avg(self,  agent_updates_dict):
        # Use the received models to infer the consensus model
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data
    
    def TM(self, inputs,b):
        if len(inputs) - 2 * b > 0:
            b = b
        else:
            b = b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError

        stacked = torch.stack(inputs, dim=0)
        largest, _ = torch.topk(stacked, b, 0)
        neg_smallest, _ = torch.topk(-stacked, b, 0)
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= (len(inputs) - 2 * b)
        return new_stacked
    
    def agg_krum(self, agent_updates_dict, krum_param_m=1):
        def _compute_krum_score( agent_updates_list, byzantine_client_num):
            with torch.no_grad():
                krum_scores = []
                num_client = len(agent_updates_list)
                # logging.info(num_client)
                for i in range(0, num_client):
                    dists = []
                    for j in range(0, num_client):
                        if i != j:
                            dists.append(
                                torch.norm(agent_updates_list[i].to(self.args.device)- agent_updates_list[j].to(self.args.device))
                                .item() ** 2
                            )
                    dists.sort()  # ascending
                    score = dists[0: num_client - byzantine_client_num - 2]
                    krum_scores.append(sum(score))
            # logging.info("finish")
            return krum_scores

        # Compute list of scores
        agent_updates_list = list(agent_updates_dict.values())
        byzantine_num = min( self.args.num_corrupt, len(agent_updates_dict)-1) 
        krum_scores = _compute_krum_score(agent_updates_list, byzantine_num)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]
        return_gradient = [agent_updates_list[i] for i in score_index]
        return (sum(return_gradient)/len(return_gradient)).to("cpu"),return_gradient
    
    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=   n_agent_data *  update
            total_data += n_agent_data
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)


    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2) 
            update.div_(max(1, l2_update/self.args.clip))
        return
                  
    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])
                              
        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)
        
        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round) 
        return
        
   

        
  
