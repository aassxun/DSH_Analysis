from __future__ import absolute_import
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from torch.hub import load_state_dict_from_url
import math
from torch.nn import Parameter
from models.resnet import resnet50
# from resnet import resnet50
class SW_layer(nn.Module):

    def __init__(self, bit, margin, gamma, step, maxx, likehood_coef,num_classes):
        super(SW_layer, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.maxx = maxx
        self.step = step
        self.likehood_coef = likehood_coef
        self.mu = Parameter(torch.Tensor(bit, num_classes))  # [bits, cls]
        self.sigma = Parameter(torch.Tensor(bit, num_classes))
        self.sigma.requires_grad = False
        self.cnt = 0

    def forward(self, input, label):  # input:[N, bits], label:[N,]
        self.cnt += 1
        if self.cnt % (self.step) == (self.step - 1):
            self.margin = min(self.margin * self.gamma, self.maxx)
            print('margin becomes %.2f\n' % self.margin)
        # ipdb.set_trace()
        max_likehood_loss = 0.5 * ((input - self.mu[:, label].transpose(1, 0)) ** 2 / (torch.max(
            torch.Tensor([0]).to(input.device) + 1e-9, self.sigma[:, label].transpose(1, 0)))).sum(dim=1)
        max_likehood_loss -= torch.log(torch.max(torch.Tensor(
            [0]).to(input.device) + 1e-9, (self.sigma[:, label].prod(dim=0))**(-0.5)))
        max_likehood_loss = self.likehood_coef * max_likehood_loss.mean(dim=0)
        x = input.view(input.size(0), input.size(1), 1)  # [N, bits, 1]

        pro_res = (((x - self.mu)**2) / (torch.max(self.sigma,
                                                   torch.Tensor([0]).to(input.device) + 1e-9))).sum(dim=1)  # distance [N, cls]
        pro_res = (-0.5) * pro_res
        label = label.view(-1, 1)  # size=(B,1)

        index = pro_res.detach() * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.detach().view(-1, 1), 1)
        # index = index.byte()
        index = index.bool() 
        # index = Variable(index)
        index = torch.tensor(index)
        pro_res[index] += self.margin * pro_res[index]  #margin

        det = self.sigma.prod(dim=0)
        res = pro_res + \
            (torch.log(torch.max(torch.Tensor(
                [0]).to(input.device) + 1e-9, det ** (-0.5))))
        # [N, cls]
        return res, max_likehood_loss



class daph_resnet(nn.Module):
    def __init__(self, code_length, pretrained,num_classes):
        super().__init__()
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.hash = nn.Linear(2048, code_length, bias=True)
        self.hash.weight.data.normal_(0, 0.01)
        self.hash.bias.data.fill_(0.0)

        self.imgs_semantic = SW_layer(code_length, 0.1, 2, 5000, 0.2, 0.01,num_classes)
        init.kaiming_normal_(self.imgs_semantic.mu.data,
                             mode='fan_out', nonlinearity='relu')  # 
        init.constant_(self.imgs_semantic.sigma.data, 1.0)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.hash(x)

        return x


def dpah(code_length,pretrained,num_classes):
    model = daph_resnet(code_length, pretrained,num_classes)
    return model
if __name__ == "__main__":
    device = torch.device('cuda:1')
    model = ortho(12, 200, 4, 2048, device).to(device)
    model.eval()
    img = torch.rand(2, 3, 224, 224, device=device)
    output = model(img)
    print(output.shape)
