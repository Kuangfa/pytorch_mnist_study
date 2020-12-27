#-*-encoding:utf-8-*-
"""
# function/功能 : 
# @File : LeNet_model.py 
# @Time : 2020/7/22 9:33 
# @Author : kf
# @Software: PyCharm
"""
import torch.nn as nn
import torch.nn.functional as F# 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等

# 创建模型
class LeNet(nn.Module):
    def __init__(self):#输入为28*28*3
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16*5*5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, 16*5*5)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x
   # def num_flat_features(self, x):
   #     size = x.size()[1:]
   #     num_features = 1
   #     for s in size:
   #         num_features *= s
   #     return num_features

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.BatchNorm1d(84),
                                 nn.ReLU(), nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    xx=LeNet5()
    print(xx)
