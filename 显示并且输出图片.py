# -*-encoding:utf-8-*-
"""
# function/功能 : 
# @File : 显示并且输出图片.py 
# @Time : 2020/12/24 19:23 
# @Author : kf
# @Software: PyCharm
"""
import os

import matplotlib.pyplot as plt  # 用于显示图片
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 忽略警告
import warnings

warnings.filterwarnings('ignore')
# 选择运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 获得数据
def get_dataset(data_dir, batch_size=64):
    dataset = datasets.MNIST(root=data_dir,
                             transform=transforms.ToTensor(),
                             train=True,
                             download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)
    return dataloader


# 获得数据
def get_testdataset(data_dir, batch_size=64):
    dataset = datasets.MNIST(root=data_dir,
                             transform=transforms.ToTensor(),
                             train=False,
                             download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)
    return dataloader


outputDir='sample/'
if not  os.path.exists(outputDir):
    os.makedirs(outputDir)
data_loader_train = get_dataset(r'MNIST_data', batch_size=100)
data_loader_test = get_testdataset(r'MNIST_data', batch_size=100)
for i, (images, labels) in enumerate(data_loader_train):
    if (i + 1) % 10 == 0:
        print('batch_number [{}/{}]'.format(i + 1, len(data_loader_train)))
        for j in range(len(images)):
            image = images[j].resize(28, 28)  # 将(1,28,28)->(28,28)
            image=image.numpy()*255
            cv2.imwrite(outputDir+str(i)+'.jpg',image)
            plt.imshow(image)  # 显示图片,接受tensors, numpy arrays, numbers, dicts or lists
            plt.axis('off')  # 不显示坐标轴
            plt.title("$The {} picture in {} batch, label={}$".format(j + 1, i + 1, labels[j]))
            plt.show()
