from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import logging
from typing import Iterable, Optional
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import save_image

from fl_api.utils.cam import SmoothGradCAMpp
from fl_api.utils.visualize import reverse_normalize, visualize
from torchvision.utils import save_image
import pandas as pd


def write_data_to_csv(data, path):
    df = pd.DataFrame(pd.DataFrame.from_dict(data, orient='index').values.T, columns=list(data.keys()))
    df.to_csv(path, encoding='utf-8')

def compute_channel_sign(sign_tensor):
    # computing the value of sign for each channel
    wk_size = sign_tensor.size()
    out_channel = wk_size[0]
    channel_sign_list = []
    for oc in range(out_channel):
        channel_sign_list.append(torch.sum(sign_tensor[oc, :, :, :]).item())
    return channel_sign_list

def  compute_cosine_similarity(tensor0, tensor1):
    tensor0 = tensor0.unsqueeze(0)
    tensor1 = tensor1.unsqueeze(0)

    return F.cosine_similarity(tensor0, tensor1).item()

def smooth_grad_CAM(model, data_loader, device, name):
    # 从DataLoader中获取第一个batch
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # 从这个batch中获取第一张图片
    first_image, first_label = images[10].to(device), labels[10].to(device)
    save_image(first_image, f'logs/cifar10/visualize/gradcam/image1.png')
    first_image = first_image.unsqueeze(0)

    # wrapper for class activation mapping. Choose one of the following.
    # wrapped_model = CAM(model, target_layer)
    # wrapped_model =GradCAM(model, target_layer)
    # wrapped_model = GradCAMpp(model, target_layer)
    target_layers = [model.conv1, model.conv2, model.res1[0], model.res1[1], model.conv3, model.conv4, model.res2[0], model.res2[1]]

    wrapped_model = SmoothGradCAMpp(model, model.conv4, n_samples=1, stdev_spread=0.01)
    cam, idx = wrapped_model(first_image)
    # visualize only cam
    # plt.imshow(cam.cpu().squeeze().numpy(), alpha=0.5, cmap='jet')

    img = reverse_normalize(first_image)
    save_image(img, f'logs/cifar10/visualize/gradcam/image.png')
    heatmap = visualize(img.cpu(), cam.cpu())
    # hm = (heatmap.cpu().squeeze().numpy().transpose(1, 2, 0)).astype(np.int32)
    save_image(heatmap, f'logs/cifar10/visualize/gradcam/{name}.png')



def CLP(weights):
    channel_lips = []
    weights_norm = []
    for idx in range(weights.shape[0]):
        weight = weights[idx]
        weight = weight.reshape(weight.shape[0], -1).cpu()
        channel_lips.append(torch.svd(weight)[1].max())
        weights_norm.append(float(torch.norm(weight)))
    channel_lips = torch.Tensor(channel_lips)
    return channel_lips

def visual_loss_distribution(clean_losses, poison_losses, title):
    clean_losses = [round(i, 1) for i in clean_losses]
    poison_losses = [round(i, 1) for i in poison_losses]
    clean_losses_dict = Counter(clean_losses)
    poison_losses_dict = Counter(poison_losses)

    X_clean = list(clean_losses_dict.keys())
    y_clean = list(clean_losses_dict.values())
    X_poison = list(poison_losses_dict.keys())
    y_poison = list(poison_losses_dict.values())

    names = sorted(list(set(X_clean) | set(X_poison)))
    names = [i for i in np.arange(0, max(names)+0.1, 0.1)]
    y_clean_1 = []
    y_poison_1 = []
    for name in names:
        if name in X_clean:
            ind = X_clean.index(name)
            y_clean_1.append(y_clean[ind])
        else:
            y_clean_1.append(0)
        if name in X_poison:
            ind = X_poison.index(name)
            y_poison_1.append(y_poison[ind])
        else:
            y_poison_1.append(0)

    names = [str(i) for i in names]
    # plt.bar(names, [(i / sum(y_clean_1)) * 100 for i in y_clean_1], label='Clean samples')
    # plt.bar(names, [(i / sum(y_clean_1)) * 100 for i in y_poison_1], label='Poisoned samples')
    # plt.xticks(rotation=75)
    # plt.ylabel('Proportion (%)')
    # plt.xlabel('Loss Value')
    # plt.title(title)
    # plt.legend()
    # plt.show()
    return names, y_clean_1, y_poison_1


def model_clp(model, w_global):
    model.load_state_dict(w_global)
    channel_lips_conv = CLP(model)
    channel_lips_conv_last = channel_lips_conv[-1]
    channel_lips_conv_last = [round(num.item(), 2) for num in channel_lips_conv_last]
    channel_lips_conv_last_dict = Counter(channel_lips_conv_last)

    X = []
    y = []
    for key in sorted(list(channel_lips_conv_last_dict.keys())):
        X.append(key)
        y.append(channel_lips_conv_last_dict[key])
    return X, y


# 定义数据划分方法
def split_dataset(dataset, num_clients):
    client_data = []
    data_size = len(dataset) // num_clients

    for _ in range(num_clients - 1):
        client_part, dataset = random_split(dataset, [data_size, len(dataset) - data_size])
        client_data.append(client_part)

    client_data.append(dataset)
    return client_data


def count_communication_params(parameters):
    num_non_zero_weights = 0
    for name in parameters:
        num_non_zero_weights += torch.count_nonzero(parameters[name])
    mb_size = num_non_zero_weights * 4 / (1024 ** 2)  # MB
    return num_non_zero_weights, float(mb_size)


def count_communication_params_channel(num_non_zero_weights):
    mb_size = num_non_zero_weights * 4 / (1024 ** 2)  # MB
    return num_non_zero_weights, mb_size


def calculate_sparsities(args, params, tabu=[], distribution="ERK"):
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - args.dense_ratio
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        logging.info('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()

        density = args.dense_ratio
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu or "running" in name or "track" in name:
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(params[name].shape) / np.prod(params[name].shape)
                                              ) ** 1
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities


def init_masks(params, sparsities):
    masks = {}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
        masks[name] = masks[name].to("cpu")
    return masks


def vector_to_model(vec, model):
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param
    model.load_state_dict(state_dict)
    return state_dict


def vector_to_name_param(vec, name_param_map):
    pointer = 0
    for name in name_param_map:
        # The length of the parameter
        num_param = name_param_map[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        name_param_map[name].data = vec[pointer:pointer + num_param].view_as(name_param_map[name]).data
        # Increment the pointer
        pointer += num_param

    return name_param_map



def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device
