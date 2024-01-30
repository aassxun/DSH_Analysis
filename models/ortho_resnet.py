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
from models.resnet import resnet50
# from resnet import resnet50


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, codebook=None, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits
class OrthoCos_resnet(nn.Module):
    def __init__(self, code_length, num_classes, pretrained):
        super().__init__()
        prob = torch.ones(num_classes, code_length) * 0.5
        codebook = torch.bernoulli(prob) * 2. - 1.
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.model.fc = nn.Linear(512, classes)
        self.fc = nn.Linear(2048, code_length)
        self.bn = nn.BatchNorm1d(code_length, momentum=0.1)
        self.ce_fc = CosSim(code_length, num_classes, codebook, learn_cent=False)
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
        x = self.fc(x)
        v = self.bn(x)
        if not self.training:
            return v
        u = self.ce_fc(v)
        return u, v


def ortho(code_length, num_classes,pretrained):
    model = OrthoCos_resnet(code_length, num_classes,pretrained)
    return model
if __name__ == "__main__":
    device = torch.device('cuda:1')
    model = ortho(12, 200, 4, 2048, device).to(device)
    model.eval()
    img = torch.rand(2, 3, 224, 224, device=device)
    output = model(img)
    print(output.shape)
