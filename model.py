import torch
import torch.nn as nn
import torchvision

import numpy as np
from sklearn.metrics import r2_score

#RESNETS
class Resnet18(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(Resnet18, self).__init__()
        
        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)

        kernelCount = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.resnet18(x)
        return x

class Resnet34(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(Resnet34, self).__init__()
        
        self.resnet34 = torchvision.models.resnet34(pretrained=isTrained)

        kernelCount = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.resnet34(x)
        return x

class Resnet50(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(Resnet50, self).__init__()
        
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        kernelCount = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.resnet50(x)
        return x

class Resnet101(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(Resnet101, self).__init__()
        
        self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)

        kernelCount = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.resnet101(x)
        return x

class Resnet152(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(Resnet152, self).__init__()
        
        self.resnet152 = torchvision.models.resnet152(pretrained=isTrained)

        kernelCount = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.resnet152(x)
        return x

#DENSENETS
class DenseNet121(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(DenseNet121, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet161(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(DenseNet161, self).__init__()
        
        self.densenet161 = torchvision.models.densenet161(pretrained=isTrained)

        kernelCount = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.densenet161(x)
        return x

class DenseNet169(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)

        kernelCount = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.densenet169(x)
        return x

class DenseNet201(nn.Module):

    def __init__(self, classCount=38, isTrained=True):
    
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)

        kernelCount = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.densenet201(x)
        return x

def conv_block(inplanes, outplanes, kernel_size, stride, padding, drop_rate):

    block = nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(outplanes),
        nn.LeakyReLU(0.01),
        nn.Dropout2d(drop_rate)
        )
    return block    

class shallow_cnn(nn.Module):    
    
    def __init__(self, classCount=38, drop_rate=0.2):
        
        super(shallow_cnn, self).__init__()

        self.conv1 = conv_block(3, 32, 7, 2, 3, drop_rate)
        self.conv2 = conv_block(32, 32, 3, 1, 1, drop_rate)
        self.conv3 = conv_block(32, 64, 3, 2, 1, drop_rate)
        self.conv4 = conv_block(64, 128, 3, 2, 1, drop_rate)
        self.conv5 = conv_block(128, 256, 3, 2, 1, drop_rate)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, classCount)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.conv5(self.conv4(self.conv3(self.conv2(x))))
        x=self.avgpool(x)
        x=torch.flatten(x, 1)
        out=self.fc(x)

        return out