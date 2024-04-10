import torch
import torch.nn as nn
from fl_api.models.resnet9 import ResNet9
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = ResNet9()
    pth_path = '/home/lgz/papers/federated_learning_20231222/codes/Isolation/fl_experiments/standalone/RLR/logs/cifar10/checkpoints/RLR_AckRatio4_40_Methodrlr_datacifar10_alpha0.5_Rnd200_Epoch2_inject0.5_Aggavg_noniidFalse_maskthreshold8_attackbadnet.pt'
    # pth_path = '/home/lgz/papers/federated_learning_20231222/codes/Isolation/fl_experiments/standalone/Isolation/logs/cifar10/checkpoints/Isolation_AckRatio4_40_MethodTopK_datacifar10_alpha0.5_Rnd200_Epoch2_inject0.5_Aggavg_noniidFalse_maskthreshold0.5_attackbadnet_topk0.6.pt'
    model.load_state_dict(torch.load(pth_path)['model_state_dict'])
    print(model.res2[1][0])

    conv_layer = model.res2[1][0]

    weights = conv_layer.weight

    l2_norms = torch.norm(weights, p=2, dim=(1, 2, 3))
    l2_norms_sort = sorted(l2_norms.detach().numpy(), reverse=True)

    # show
    plt.bar([i for i in range(len(l2_norms_sort))], l2_norms_sort)
    plt.title('L2 norm of last convolutional layer')
    plt.ylabel('L2 norm')
    plt.show()

    print(l2_norms_sort)
