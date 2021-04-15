'''

'''

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import pdb
import random


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, input_channels=6): #-- if not concanated left and right, the input_channels = 3
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1,1])
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class ELU_plus(nn.modules.Module):

    def __init__(self, alpha=1., inplace=False):
        super(ELU_plus, self).__init__()
        self.alpha = alpha
        self.inplace = inplace


    def forward(self, input):

        return F.elu(input, self.alpha, self.inplace)+self.alpha

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' + 'alpha=' + str(self.alpha) + inplace_str + ')'



class GraspNet(nn.Module):

    def __init__(self, num_class=1, num_dim=1): 
        super(GraspNet, self).__init__()

        self.num_class = num_class
        self.num_dim = num_dim        
        resnet_i = ResNet(block = BasicBlock, layers = [2, 2, 2, 2])
        self.basenet=resnet_i


    def forward(self, image):
        '''
        :param image: Bx3xHxW
        :return: update_graspSucc B x 1 x 1
        '''
        B = image.size()[0] #get batch size
        image_height = image.size()[2]
        image_width = image.size()[3]
        #--new added

        graspSucc = self.basenet.forward(image) # graspSucc shape is Bx1

        graspSucc_reshape = graspSucc.view(B, self.num_class, self.num_dim)
        if graspSucc.is_cuda:
            update_graspSucc = Variable(torch.cuda.FloatTensor(graspSucc_reshape.size()[0], graspSucc_reshape.size()[1],graspSucc_reshape.size()[2]).fill_(0))
        else:
            update_graspSucc = Variable(torch.zeros(graspSucc_reshape.size()[0], graspSucc_reshape.size()[1], graspSucc_reshape.size()[2]))

        update_graspSucc[:] = graspSucc_reshape[:]
        # update_graspSucc shape(B,1, 1)

        return update_graspSucc


class GraspLoss(nn.Module):

    def __init__(self):
        super(GraspLoss, self).__init__()
        self.grasp_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, graspSucc, ground_truth):
    
        '''
        :param graspSucc: Bx1x1
        :param ground_truth: Bx1x1
        :return: the loss, a scalar
        '''

        estimated_grasp = graspSucc[:] #Bx1x1
        
        ground_truth_grasp = ground_truth[:]#Bx1x1

        loss_grasp = self.grasp_loss(estimated_grasp, ground_truth_grasp) #Bx1
        
        return loss_grasp




