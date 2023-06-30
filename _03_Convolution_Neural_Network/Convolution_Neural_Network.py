import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable


# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # 定义卷积层和池化层
        self.conv_layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv_layer4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv_layer5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层和dropout层
        self.fc1 = nn.Linear(256 * 2 * 2, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()

        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # 前向传播
        x = self.conv_layer1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv_layer2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv_layer3(x)
        x = self.relu3(x)

        x = self.conv_layer4(x)
        x = self.relu4(x)

        x = self.conv_layer5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x