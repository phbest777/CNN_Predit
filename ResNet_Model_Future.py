import csv
import os.path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot
from pandas import to_datetime
import math,time
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import variable
import tushare as ts
import torchvision.datasets as tor_ds
import torchvision as tor
import json
import mplfinance as mplf
import cv2
import copy
import torchvision.transforms as transforms
from torchvision import models



def initialize_model(model_name, num_class, feature_extract, use_pretrained=True):
    '''
    :param model_name: 采用的模型名称
    :param num_class: 目的要分成的类别
    :param feature_extract:是否冻住参数
    :param use_pretrained: 是否下载别人训练好的模型 c\用户\torch\cachez中
    :return:
    model:新构建的模型

    '''
    model_ft = None
    input_size = 0

    if model_name == 'resnet':
        # 加载模型 pretrained 要不要把网络模型下载下来
        model_ft = models.resnet152(pretrained=use_pretrained)

        # 迁移学习
        #  model_ft：选用的模型
        # feature_extract：True False 选择是否冻结参数 若是True 则冻住参数 反之不冻住参数
        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False

        # 得到最后一层的数量512 把最后的全连接层改成2048——>102
        num_ftrs = model_ft.fc.in_features
        # 修改最后一层的模型
        model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, num_class),
            torch.nn.LogSoftmax(dim=1)
        )
        input_size = 224
    else:
        print('采用了其他模型，还没来得及编写模型代码。。。')

    return model_ft, input_size


def train_model(model, train_loader, valid_loader, optimizer, criterion, nums_epoch, filename):
    '''

    :param model: 训练模型
    :param train_loader: 训练集
    :param valid_loader: 测试集
    :param optimizer: 优化器
    :param criterion: 损失器
    :param nums_epoch: 训练轮次
    :param filename: 模型保存路径
    :return:
    '''

    # 保存最好的准确率
    best_acc = 0
    # 如果GPU可用 在GPU中运行
    model.to(device)

    # 保存模型的参数
    best__model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(nums_epoch):
        # 训练
        running_loss = 0
        for batch_idx, (input, label) in enumerate(train_loader, 0):
            # 如果GPU可用 则放入道GPU中进行运算
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(input)
            _, pres = torch.max(outputs, dim=1)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 8 == 7:
                print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
                running_loss = 0

        # 每经过一轮数据训练  测试准确度
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                image, label = data
                image=image.to(device)
                label=label.to(device)
                output = model(image)
                # _代表最大的元素 pre代表索引  索引本质就代表了预测元素值
                _, pre = torch.max(output.data, dim=1)
                total += label.size(0)
                correct += (pre == label).sum().item()

        epoch_acc = 100 * correct / total
        print('准确率为：%d %%' % (epoch_acc))

        # 记录准确度最好的模型权重 并保存再文件中
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best__model_wts = copy.deepcopy(model.state_dict())
            state = {
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, filename)

        # 返回一个准确度最高的模型
        model.load_state_dict(best__model_wts)

    return model


if __name__=='__main__':
    data_dir = 'D:\PythonProject\AI\PyTorch\CNN_Predit\DATA\IMG\ResNet\MIDDLE'
    train_dir = data_dir + '\\train'
    valid_dir = data_dir + '\\valid'

    data_tranforms = {
        'train': transforms.Compose([transforms.Resize(224),
                                     transforms.RandomRotation(45),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                     transforms.RandomGrayscale(p=0.025),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.408], [0.229, 0.224, 0.225]),
                                     ]),
        'valid': transforms.Compose([transforms.Resize(256),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.408], [0.229, 0.224, 0.225]),
                                    ])
    }

    batch_size = 8
    image_datasets = {x: tor_ds.ImageFolder(os.path.join(data_dir, x), data_tranforms[x]) for x in ['train', 'valid']}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                    ['train', 'valid']}
    #print(data_loaders['train'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    with open('lable.json', 'r') as f:
        lableToName = json.load(f)


    def im_convert(tensor):
        '''展示数据'''
        image = tensor.to('cpu').clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)
        return image


    fig = plt.figure(figsize=(20, 12))
    columns = 4
    rows = 2
    dataiter = iter(data_loaders['valid'])
    inputs, classes = dataiter.next()
    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        ax.set_title(lableToName[str(class_names[classes[idx]])])
        #ax.set_title(class_names[idx])
        plt.imshow(im_convert(inputs[idx]))
    plt.show()

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print("CUDA is not available. Training on cpu...")
    else:
        print("CUDA is available.Training on gpu")
    # 如果存在GPU 则选用GPU 否则选用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置哪些层需要进行训练
    model_ft, input_size = initialize_model("resnet", 5, feature_extract=True, use_pretrained=True)

    # 是否训练所有层
    # 打印需要学习的参数
    print('Parameter to learn:')
    param_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            param_to_update.append(param)
            print("\t", name)

    # 打印修改之后的网络模型
    print(model_ft)

    # 模型保存
    filename = 'checkpoint152_Future.pth'
    # 构建优化器
    optimizer_ft = torch.optim.Adam(param_to_update, lr=0.01)
    # #对于学习率 每7个epoch衰减成10分之一
    # schedule = torch.optim.lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)

    # 构建损失
    # 为什么使用这个损失函数而不用torch.nn.CrossEntropyLoss
    # 因为torch.nn.CrossEntropyLoss相当于LogSoftmax()和torch.nn.NLLLoss()的集合
    criterion = torch.nn.NLLLoss()

    model = train_model(model_ft, data_loaders['train'], data_loaders['valid'], optimizer_ft, criterion, 20, filename)