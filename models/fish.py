import torch
import torch.nn as nn
from torchvision import models


class fish_ResNet50(nn.Module):
    def __init__(self, bits, classes, class_mask_rate,device, pretrained=True):
        super(fish_ResNet50, self).__init__()
        self.model = models.resnet50(pretrained)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.model.fc = nn.Linear(512, classes)
        self.model.fc = nn.Linear(2048, classes)
        self.model.b = nn.Linear(classes, bits)
        self.device = device
        self.class_mask_rate = class_mask_rate

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        fm = x
        A = torch.sum(fm.detach(), dim=1, keepdim=True)
        a = torch.mean(A, dim=[2, 3], keepdim=True)
        M = (A > a).float().detach() + (A < a).float().detach() * 0.5#0.1
        # print(M.size())
        x = x * M

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        x_mask = torch.ones(x.size()).detach().to(self.device) * self.class_mask_rate#0.7
        for i in range(x_mask.size()[0]):
            x_mask[i, torch.argmax(x[i])] = 1

        x_b = x * x_mask
        b = self.model.b(x_b)
        return fm, x, b

# if __name__ == '__main__':
#     net = ResNet18(64, 10, 0.7)
#     net = net.to('cuda')
#     x = torch.ones(2, 3, 224, 224).to('cuda')
#     fm,x,b =net(x)
#     print(fm.shape)
#     print(x.shape)
#     print(b.shape)
