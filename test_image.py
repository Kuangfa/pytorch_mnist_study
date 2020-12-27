import cv2
import numpy
import torch
import torch.utils.data
from PIL import Image, ImageDraw, ImageFont
from numpy import unicode
from torchvision import transforms

from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 权重初始化
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('DeformConv2d') == -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


# 模型训练初始化
def model_init(temp_model):
    temp_model = nn.DataParallel(temp_model)
    temp_model.train(mode=True)
    temp_model.apply(weight_init)
    temp_model.to(device)
    return temp_model


def drawText(image, text,imagepath):
    # 图像从OpenCV格式转换成PIL格式
    # img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_PIL = Image.fromarray(image)
    font = ImageFont.load_default()
    # 需要先把输出的中文字符转换成Unicode编码形式
    if not isinstance(text, unicode):
        text = text.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text((0, 0), text, fill=128, font=font)
    # 使用PIL中的save方法保存图片到本地
    img_PIL.save(imagepath)

import os
def test(imagepath='sample/99.jpg'):
    save_dir = 'output/'
    my_model = LeNet5()
    my_model = model_init(my_model)
    # 训练
    if torch.cuda.is_available():
        my_model.load_state_dict(torch.load(save_dir + 'model_lastest.pt'))
    else:
        my_model.load_state_dict(
            torch.load(save_dir + 'model_lastest.pt', map_location='cpu'))
    with torch.no_grad():
        image = transforms.ToTensor()(Image.open(imagepath).convert('L')).unsqueeze(0)
        image = image.to(device)
        outputs = my_model(image)
        _, predicted = torch.max(outputs.data, 1)
        print('label:', predicted.item())
        # outimage = image.squeeze().numpy() * 255
        # outimage=outimage.astype(numpy.uint8)
        # text='label:{}'.format(predicted.item())
        # drawText(outimage,text,save_dir+os.path.split(imagepath)[1])



if __name__ == '__main__':
    imagepath = 'sample/9.jpg'
    test(imagepath)
