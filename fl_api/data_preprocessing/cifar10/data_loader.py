import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fl_api.utils.general import split_dataset

# 加载CIFAR-10数据集
def load_cifar10_data(num_clients, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../../../data', train=True, download=True, transform=transform)

    # 划分数据集给每个客户端
    client_datasets = split_dataset(trainset, num_clients)

    # 创建训练集和测试集的数据加载器
    trainloaders = [DataLoader(client_data, batch_size=batch_size, shuffle=True) for client_data in client_datasets]
    testset = torchvision.datasets.CIFAR10(root='../../../data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloaders, testloader
