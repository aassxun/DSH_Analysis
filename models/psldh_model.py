import torch.nn as nn
from torchvision import models
import torch
from models.resnet import resnet50
class hash_net(nn.Module):
    def __init__(self, classes,pretrained=True):
        super(hash_net, self).__init__()
        self.model = models.resnet50(pretrained)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.model.fc = nn.Linear(512, classes)
        self.hash_layer = nn.Sequential(nn.Dropout(),
                                        nn.Linear(2048,classes),
                                        nn.Tanh())

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x= self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.hash_layer(x)
        return x

class Label_net(nn.Module):
    def __init__(self, label_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(Label_net, self).__init__()
        self.module_name = "text_model"
        # 400
        cl1 = nn.Linear(label_dim, 512)

        cl2 = nn.Linear(512, bit)

        self.cl_text = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            cl2,
            nn.Tanh()
        )
    def forward(self, x):
        y = self.cl_text(x)
        return y
