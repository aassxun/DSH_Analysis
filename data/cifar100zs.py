import os
import sys
import pickle
from PIL import Image, ImageFile
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from data.transform import encode_onehot
# from transform import encode_onehot
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

def load_data(root, batch_size, num_workers,zero_shot_classes = 25, sampler=None):
    #1

    query_dataset = CIFAR100Test(root, zero_shot_classes=zero_shot_classes, zs=False, transform=transform_test())
    train_dataset = CIFAR100Train(root, zero_shot_classes=zero_shot_classes, zs=False, drop = True,transform=transform_train())
    retrieval_dataset = CIFAR100Train(root, zero_shot_classes=zero_shot_classes, zs=False, transform=transform_test())
    query4zero_shot_dataset = CIFAR100Test(root, zero_shot_classes=zero_shot_classes, zs=True, transform=transform_test())
    database4zero_shot_dataset = CIFAR100Train(root, zero_shot_classes=zero_shot_classes, zs=True, transform=transform_test())


    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers
    )

    query4zero_shot_dataloader = DataLoader(
        query4zero_shot_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers
    )

    database4zero_shot_dataloader = DataLoader(
        database4zero_shot_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers
    )
    print(len(query_dataloader))
    print(len(train_dataloader))
    print(len(retrieval_dataloader))

    return query_dataloader, train_dataloader, retrieval_dataloader, query4zero_shot_dataloader, database4zero_shot_dataloader



class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, zero_shot_classes = 25,zs = False,drop = False,transform=None):
        class4train = 100-zero_shot_classes
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data_all = pickle.load(cifar100, encoding='bytes')
            # img_labels = np.array(self.data_all['fine_labels'.encode()])[idx]
            idx = None
        if zs:
            idx = np.array(self.data_all['fine_labels'.encode()]) >= class4train
            self.targets = np.array(self.data_all['fine_labels'.encode()])[idx]
            # self.targets = encode_onehot(self.targets-class4train, zero_shot_classes)
            self.targets = encode_onehot(self.targets, 100)

        else:
            idx = np.array(self.data_all['fine_labels'.encode()]) < class4train
            self.targets = np.array(self.data_all['fine_labels'.encode()])[idx]
            if drop:
                self.targets = encode_onehot(self.targets, class4train)
            else:
                self.targets = encode_onehot(self.targets,100)
        self.data = self.data_all['data'.encode()][idx,:]#.reshape(-1, 3, 32, 32)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __getitem__(self, index):
        label = self.targets[index]
        r = self.data[index, :1024].reshape(32, 32)
        g = self.data[index, 1024:2048].reshape(32, 32)
        b = self.data[index, 2048:].reshape(32, 32)
        # # label = self.data_all['fine_labels'.encode()][index]
        # # r = self.data_all['data'.encode()][index, :1024].reshape(32, 32)
        # # g = self.data_all['data'.encode()][index, 1024:2048].reshape(32, 32)
        # # b = self.data_all['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))
        image = Image.fromarray(image)
        # image = self.data[index]

        if self.transform:
            image = self.transform(image)
        return image, label,index

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, zero_shot_classes = 25,zs = False,drop = False,transform=None):
        class4train = 100-zero_shot_classes
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data_all = pickle.load(cifar100, encoding='bytes')
            # img_labels = np.array(self.data_all['fine_labels'.encode()])[idx]
            idx = None
        if zs:
            idx = np.array(self.data_all['fine_labels'.encode()]) >= class4train
            self.targets = np.array(self.data_all['fine_labels'.encode()])[idx]
            # self.targets = encode_onehot(self.targets - class4train, zero_shot_classes)
            self.targets = encode_onehot(self.targets , 100)
        else:
            idx = np.array(self.data_all['fine_labels'.encode()]) < class4train
            self.targets = np.array(self.data_all['fine_labels'.encode()])[idx]
            if drop:
                self.targets = encode_onehot(self.targets, class4train)
            else:
                self.targets = encode_onehot(self.targets, 100)
        self.data = self.data_all['data'.encode()][idx,:]#.reshape(-1, 3, 32, 32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __getitem__(self, index):
        label = self.targets[index]
        r = self.data[index, :1024].reshape(32, 32)
        g = self.data[index, 1024:2048].reshape(32, 32)
        b = self.data[index, 2048:].reshape(32, 32)
        # # label = self.data_all['fine_labels'.encode()][index]
        # r = self.data_all['data'.encode()][index, :1024].reshape(32, 32)
        # g = self.data_all['data'.encode()][index, 1024:2048].reshape(32, 32)
        # b = self.data_all['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))
        image = Image.fromarray(image)
        # image = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, label,index

# CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
def transform_train():
    # normalize = transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
    #                                  std=CIFAR100_TRAIN_STD)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def transform_test():
    # normalize = transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
    #                                  std=CIFAR100_TRAIN_STD)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
# transform_train = transforms.Compose([
#     #transforms.ToPILImage(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
# ])



if __name__ =="__main__":
    myset = CIFAR100Train('/2T/dataset/cifar-100-python', zero_shot_classes=25, zs=False)
