import torch.nn as nn
import torch.nn.functional as F

class MLP_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(28*28, 100, bias=False)
        self.fc2 = nn.Linear(100, num_classes, bias=False)

    def forward(self, x):
        x= x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x