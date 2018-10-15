import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# http://www.cs.toronto.edu/~kriz/cifar.html
# root：Root directory of dataset where directory ``cifar-10-batches-py`` exists or will be saved to if download is set to True.

## 数据加载
# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])

# 训练集
# 下载数据或者提取数据
trainset = tv.datasets.CIFAR10(
                    root='D:/Machine_learning/1.Dataset/1.CIFAR10',# windows
#                     root = '/Users/iker/Downloads',# mac
                    train = True,   # True 则作为训练集，否则为测试集
                    download=True,
                    transform=transform)

# 定义一个数据迭代器
trainloader = t.utils.data.DataLoader(
                    trainset, 
                    batch_size=4,
                    shuffle=True,  # 设置为True时会在每个epoch重新打乱数据，训练集打乱，测试集不打乱
                    num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
                    root = 'D:/Machine_learning/1.Dataset/1.CIFAR10',# windows
#                     root = '/Users/iker/Downloads',# mac
                    train=False, 
                    download=True, 
                    transform=transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4, 
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## 网络搭建
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # (2,2)相当于把图像变成原来一半
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x

# 损失函数和参数更新算法
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 优化更新算法

## 参数训练
t.set_num_threads(8)
for epoch in range(2):  
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): # enumerate可以同时返回数据下标和数据
        
        # 输入数据
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()
        
        # forward + backward 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()   
        
        # 更新参数 
        optimizer.step()
        
        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 2000 == 1999: # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f'% (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

## 测试集检验
correct  =  0  # 预测正确的图片数# 预测正确的图 
total = 0 # 总共的图片数

# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with t.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))