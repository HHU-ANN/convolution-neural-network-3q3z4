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


#定义数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
trainset = dset.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())  # 训练集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = dset.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())  # 测试集
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# 实例化AlexNet模型并将其移动到GPU上进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=10).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度接下来，我将向您展示如何在训练完成后测试AlexNet模型的精度。

# 测试模型的函数
def test(model, testloader):
    correct = 0  # 正确预测的数量
    total = 0  # 总共预测的数量

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %.2f %%' % accuracy)


# 训练模型
for epoch in range(150):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每100个batch输出一次损失值
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
torch.save(model.state_dict(), './pth/model.pth')

# 测试模型的精度
test(model, testloader)