import copy
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector
from fl_api.standalone.Isolation.isolation_client import Client
from fl_api.models.resnet9 import ResNet9, ResNet9_tinyimagenet
from fl_api.models.vgg import VGG16, VGG11
from fl_api.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from fl_api.models.cnn_mnist import CNN_MNIST
from fl_api.models.mlp_mnist import MLP_MNIST
from fl_api.models.cnn_cifar10 import CNN_CIFAR
from fl_api.standalone.Isolation.isolation_model_trainer import MyModelTrainer
from fl_api.standalone.Isolation.isolation_aggregation import Aggregation
from fl_api.utils.general import parameters_to_vector, visual_loss_distribution, smooth_grad_CAM, compute_cosine_similarity, write_data_to_csv


class IsolationAPI:
    def __init__(self, dataset, device, args, logger):
        self.args = args
        self.logger = logger
        self.device = device
        self.client_list = []
        self.agent_data_sizes = {}
        # self.train_loaders, self.test_loader = dataset
        self.model = self._create_model(self.args.model)
        self.train_dataset, self.poisoned_val_loader, self.poisoned_val_only_x_loader, self.user_groups, self.val_loader = dataset

        self.model_trainer = self._custom_model_trainer(self.args, self.model, self.logger)

        self.n_model_params = len(
            parameters_to_vector([self.model.state_dict()[name] for name in self.model.state_dict()]))
        self.criterion = nn.CrossEntropyLoss().to(device)
        self._setup_clients()
        self.acc_vec = []
        self.asr_vec = []
        self.pacc_vec = []
        self.per_class_vec = []
        self.cum_poison_acc_mean = 0

    def _setup_clients(self):
        print("############setup_clients (START)#############")
        for client_idx in range(self.args.num_clients):
            c = Client(client_idx, self.args, self.device, self.model_trainer, self.train_dataset, self.logger,
                       self.user_groups[client_idx])
            self.client_list.append(c)
            self.agent_data_sizes[client_idx] = c.n_data
        print("############setup_clients (END)#############")

    def _create_model(self, model_name):
        # create
        model = None
        if model_name == "resnet18":
            model = ResNet18(num_classes=self.args.n_classes)
        if model_name == "resnet34":
            model = ResNet34(num_classes=self.args.n_classes)
        if model_name == "resnet50":
            model = ResNet50(num_classes=self.args.n_classes)
        if model_name == "resnet101":
            model = ResNet101(num_classes=self.args.n_classes)
        elif model_name == 'vgg16':
            model = VGG16(num_classes=self.args.n_classes)
        elif model_name == 'vgg11':
            model = VGG11(num_classes=self.args.n_classes)
        elif model_name == 'cnn_cifar10':
            model = CNN_CIFAR()
        if model_name == 'CNN_mnist' or self.args.dataset == 'fmnist':
            return CNN_MNIST()
        if model_name == 'MLP_mnist' or self.args.dataset == 'mnist':
            return MLP_MNIST()
        elif model_name == 'resnet9' and self.args.dataset != 'tinyimagenet':
            model = ResNet9(num_classes=self.args.n_classes)
        elif model_name == 'resnet9' and self.args.dataset == 'tinyimagenet':
            model = ResNet9_tinyimagenet(num_classes=self.args.n_classes)
        return model

    def _custom_model_trainer(self, args, model, logger):
        return MyModelTrainer(model, args, logger)

    def _client_sampling(self, num_clients):
        client_indexes = np.random.choice(num_clients, math.floor(num_clients * self.args.frac), replace=False)
        return client_indexes


    def train(self):
        neurotoxin_mask = {}
        aggregator = Aggregation(self.agent_data_sizes, self.n_model_params, self.poisoned_val_loader, self.args, self.device, None)
        w_global = self.model_trainer.get_model_params()
        # 初始化
        w_per_models = {client_id: copy.deepcopy(w_global) for client_id in range(self.args.num_clients)}
        cos_list = []

        loss_dict_global = {}
        loss_dict_common = {}
        loss_dict_common_per = {}

        for round_idx in range(1, self.args.comm_round+1):
            print("################Communication round : {}".format(round_idx))

            client_updates_dict = {}
            client_params_dict = {}
            client_topk_mask_dict = {}
            # client_clp_dict = {}
            client_grads_dict = {}
            client_indexes = self._client_sampling(self.args.num_clients)

            client_indexes = np.sort(client_indexes)

            print("client_indexes = " + str(client_indexes))


            for client_id in client_indexes:
                client = self.client_list[client_id]
                topk_mask, client_params, client_grads = client.train(copy.deepcopy(w_per_models[client_id]), round_idx, self.criterion, neurotoxin_mask = neurotoxin_mask)
                client_params_dict[client_id] = copy.deepcopy(client_params)
                client_topk_mask_dict[client_id] = copy.deepcopy(topk_mask)
                client_grads_dict[client_id] = copy.deepcopy(client_grads)

            w_client_round, w_global, avg_global_model, common_params_malicious, personal_params_malicious, common_grad_malicious, common_grad_benign = aggregator.aggregate_updates(client_topk_mask_dict, client_params_dict, client_indexes, round_idx=round_idx, client_grads_dict=client_grads_dict)
            # Updating the params of each client
            for idx in w_client_round.keys():
                w_per_models[idx] = w_client_round[idx]


            cos = compute_cosine_similarity(common_grad_benign, common_grad_malicious)
            cos_list.append(cos)
            print('cos: ',cos)

            names_global, y_clean_global, y_poison_global = self.test(w_global, round_idx, title=f'Round: {round_idx} after split', name=f'round_{round_idx}_global')
        #     names_common, y_clean_common, y_poison_common = self.test(common_params_malicious, round_idx, title=f'Round: {round_idx} Malicious Common', name=f'round_{round_idx}_malicious_common')
        #     names_common_per, y_clean_common_per, y_poison_common_per = self.test(personal_params_malicious, round_idx, title=f'Round: {round_idx} Malicious Common + Personal', name=f'round_{round_idx}_malicious_common_person')
        #
        #     loss_dict_global[f'name_{round_idx}'] = names_global
        #     loss_dict_global[f'clean_{round_idx}'] = y_clean_global
        #     loss_dict_global[f'poison_{round_idx}'] = y_poison_global
        #
        #     loss_dict_common[f'name_{round_idx}'] = names_common
        #     loss_dict_common[f'clean_{round_idx}'] = y_clean_common
        #     loss_dict_common[f'poison_{round_idx}'] = y_poison_common
        #
        #     loss_dict_common_per[f'name_{round_idx}'] = names_common_per
        #     loss_dict_common_per[f'clean_{round_idx}'] = y_clean_common_per
        #     loss_dict_common_per[f'poison_{round_idx}'] = y_poison_common_per
        #
        # write_data_to_csv(loss_dict_global, f'logs/{self.args.dataset}/losses/{self.args.method}_global.csv')
        # write_data_to_csv(loss_dict_common, f'logs/{self.args.dataset}/losses/{self.args.method}_common.csv')
        # write_data_to_csv(loss_dict_common_per, f'logs/{self.args.dataset}/losses/{self.args.method}_common_person.csv')


        cos_df = pd.DataFrame({'cosine similarity': cos_list})
        cos_df.to_csv(f'logs/{self.args.dataset}/cosine_similarity/{self.args.method}.csv', encoding='utf-8')


    def test(self, global_model, round_idx, title, name):

        val_loss, (val_acc, val_per_class_acc), _, val_samples_losses = self.get_loss_n_accuracy(global_model, self.criterion, self.val_loader,
                                                                              self.args, round_idx, self.args.n_classes, name=None)
        print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
        print(f'| Val_Per_Class_Acc: {val_per_class_acc} |')
        self.acc_vec.append(val_acc)
        self.per_class_vec.append(val_per_class_acc)

        poison_loss, (asr, _), fail_samples, poison_samples_losses = self.get_loss_n_accuracy(global_model, self.criterion,
                                                                        self.poisoned_val_loader, self.args, round_idx, self.args.n_classes, name=name)
        self.cum_poison_acc_mean += asr
        self.asr_vec.append(asr)
        print(f'| Attack Loss/Attack Success Ratio: {poison_loss:.3f} / {asr:.3f} |')
        names, y_clean, y_poison = visual_loss_distribution(val_samples_losses, poison_samples_losses, title)
        # visual_loss_distribution(val_grad_norm, poison_grad_norm, title+'-grads')

        poison_loss, (poison_acc, _), fail_samples, _ = self.get_loss_n_accuracy(global_model, self.criterion,
                                                                                self.poisoned_val_only_x_loader, self.args,
                                                                               round_idx, self.args.n_classes, name=None)
        self.pacc_vec.append(poison_acc)
        print(f'| Poison Loss/Poison accuracy: {poison_loss:.3f} / {poison_acc:.3f} |')

        save_frequency = 25
        PATH = "logs/{}/checkpoints/Isolation_AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_Agg{}_noniid{}_maskthreshold{}_attack{}_topk{}.pt".format(
            self.args.dataset, self.args.num_corrupt, self.args.num_clients, self.args.method, self.args.dataset,
            self.args.alpha, round_idx, self.args.epochs,
            self.args.poison_frac, self.args.aggr, self.args.non_iid, self.args.theta, self.args.attack, self.args.topk)
        if round_idx % save_frequency == 0:
            torch.save({
                'option': self.args,
                'model_state_dict': self.model.state_dict(),
                'acc_vec': self.acc_vec,
                "asr_vec": self.asr_vec,
                'pacc_vec': self.pacc_vec,
                "per_class_vec": self.per_class_vec,
            }, PATH)

            # loss_dict = {'val_loss': val_samples_losses, 'poisoned_loss': poison_samples_losses}
            # df = pd.DataFrame(loss_dict)
            # df.to_csv("logs/{}/losses/{}.csv".format(self.args.dataset, title), encoding='utf-8')



        # if round_idx == self.args.comm_round:
        #     self.plot_data(PATH)
        return names, y_clean, y_poison

    def get_loss_n_accuracy(self, w_global, criterion, data_loader, args, round_idx, num_classes=10, name=None):
        """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """

        self.model.load_state_dict(w_global)
        self.model.eval()
        self.model.to(self.device)

        # if name is not None:
        #     smooth_grad_CAM(self.model, data_loader, self.device, name)

        total_loss, correctly_labeled_samples = 0, 0
        confusion_matrix = torch.zeros(num_classes, num_classes)
        not_correct_samples = []
        all_labels = []
        all_losses = []

        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device=self.device, non_blocking=True), \
                labels.to(device=self.device, non_blocking=True)
            # compute the total loss over minibatch
            outputs = self.model(inputs)
            avg_minibatch_loss = criterion(outputs, labels)
            total_loss += avg_minibatch_loss.item() * outputs.shape[0]

            all_losses.append(avg_minibatch_loss.item())

            # get num of correctly predicted inputs in the current batch
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            all_labels.append(labels.cpu().view(-1))
            # correct_inputs = labels[torch.nonzero(torch.eq(pred_labels, labels) == 0).squeeze()]
            # not_correct_samples.append(  wrong_inputs )
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
            # fill confusion_matrix
            for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = correctly_labeled_samples / len(data_loader.dataset)
        per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
        return avg_loss, (accuracy, per_class_accuracy), not_correct_samples, all_losses

    def plot_data(self, pt_path):
        output_path = pt_path.replace('.pt', '.png').replace('checkpoints', 'visualize')
        data = torch.load(pt_path)
        acc_vec = data['acc_vec']
        asr_vec = data['asr_vec']
        pacc_vec = data['pacc_vec']

        x = [i for i in range(1, len(asr_vec)+1)]
        plt.plot(x, asr_vec, label='Attack Success Ratio')
        plt.plot(x, pacc_vec, label='Poison accuracy')
        plt.plot(x, acc_vec, label='Val Acc')

        # plt.title('theta == 2')
        plt.xlabel('Communication Rounds')
        plt.ylabel('Accuracy')

        plt.legend()
        plt.savefig(output_path)
        plt.show()


    def _computing_cosine_similarity_gradients(self, g_locals, global_g=None):
        cos_sim_dict = {}
        keys = list(g_locals.keys())
        if global_g is not None:
            # The cosine similarity between the global model and the clients' models.
            for key in keys:
                cos_sim = torch.dot(global_g, g_locals[key]) / (torch.norm(global_g) * torch.norm(g_locals[key]))
                cos_sim_dict['g_%s' % key] = cos_sim
        else:
            # The cosine similarity between clients' models.
            for i in range(len(g_locals)):
                for j in range(i + 1, len(g_locals)):
                    g1 = g_locals[keys[i]]
                    g2 = g_locals[keys[j]]
                    cos_sim = torch.dot(g1, g2) / (torch.norm(g1) * torch.norm(g2))
                    cos_sim_dict['%s_%s' % (keys[i], keys[j])] = cos_sim
        return cos_sim_dict

    def _aggregate_gradient(self, g_locals):
        keys = list(g_locals.keys())
        global_g = torch.zeros_like(g_locals[keys[0]])
        for key in keys:
            global_g += g_locals[key]
        return global_g / len(keys)

    def _computing_transfer_gain(self, w_locals, client_indexes, curr_round):
        acc = {}
        res = {}
        lambda_list = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
        for i in range(len(client_indexes)):
            print(client_indexes[i])
            acc[client_indexes[i]] = self.test(w_locals[i], curr_round)
            for j in range(i + 1, len(client_indexes)):
                for lam in lambda_list:
                    w_2 = {}
                    for name, params in w_locals[j].items():
                        w_2[name] = params * lam
                    new_model = self._aggregate([w_locals[i], w_2])
                    acc['TC_%s_%s_%s' % (client_indexes[i], client_indexes[j], lam)] = self.test(new_model, curr_round)
        for i in range(len(client_indexes)):
            for j in range(i + 1, len(client_indexes)):
                for lam in lambda_list:
                    res['TC_%s_%s_%s' % (client_indexes[i], client_indexes[j], lam)] = [
                        acc['TC_%s_%s_%s' % (client_indexes[i], client_indexes[j], lam)]['accuracy'] -
                        acc[client_indexes[i]]['accuracy'],
                        acc['TC_%s_%s_%s' % (client_indexes[i], client_indexes[j], lam)]['accuracy'] -
                        acc[client_indexes[j]]['accuracy']]
        print(res)

