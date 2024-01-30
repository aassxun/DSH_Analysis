import torch
import numpy as np
from PIL import Image
import os
import sys
import pickle

# from transform import encode_onehot
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from data.transform import encode_onehot
# from data.transform import train_transform, query_transform, Onehot, encode_onehot


def load_data(root, batch_size, num_workers,zero_shot_classes = 2):
    """
    Load cifar10 dataset.

    """
    CIFAR10.init(root, zero_shot_classes)
    query_dataset = CIFAR10('query', transform=transform_test())
    train_dataset = CIFAR10('train', transform=transform_train())
    retrieval_dataset = CIFAR10( 'retrieval', transform_test())
    query4zero_shot_dataset = CIFAR10( 'query4zero_shot', transform_test())
    database4zero_shot_dataset = CIFAR10('database4zero_shot', transform_test())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
      )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
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

    return query_dataloader, train_dataloader, retrieval_dataloader, query4zero_shot_dataloader, database4zero_shot_dataloader


class CIFAR10(Dataset):
    """
    Cifar10 dataset.
    """
    @staticmethod
    def init(root, num_zs):
        train_data_list = ['data_batch_1',
                     'data_batch_2',
                     'data_batch_3',
                     'data_batch_4',
                     'data_batch_5'
                     ]
        base_folder = 'cifar-10-batches-py'

        train_data = []
        train_targets = []

        for file_name in train_data_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2: #sys.version_info is sys.version_info(major=3, minor=7, micro=16, releaselevel='final', serial=0)
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                train_data.append(entry['data'])
                if 'labels' in entry:
                    train_targets.extend(entry['labels'])
                else:
                    train_targets.extend(entry['fine_labels'])

        train_data = np.vstack(train_data).reshape(-1, 3, 32, 32)
        train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC
        train_targets = np.array(train_targets)


        test_data_list = ['test_batch']

        test_data = []
        test_targets = []

        for file_name in test_data_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2: #sys.version_info is sys.version_info(major=3, minor=7, micro=16, releaselevel='final', serial=0)
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                test_data.append(entry['data'])
                if 'labels' in entry:
                    test_targets.extend(entry['labels'])
                else:
                    test_targets.extend(entry['fine_labels'])

        test_data = np.vstack(test_data).reshape(-1, 3, 32, 32)
        test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC
        test_targets = np.array(test_targets)
# data 里是图片，targets是图片的标签

        # Split data, targets
        CIFAR10.QUERY_IMG = test_data[test_targets<10-num_zs, :]
        CIFAR10.QUERY_TARGET = encode_onehot(test_targets[test_targets<10-num_zs], 10)
        CIFAR10.TRAIN_IMG = train_data[train_targets<10-num_zs, :]
        CIFAR10.TRAIN_TARGET = encode_onehot(train_targets[train_targets<10-num_zs],10-num_zs )

        CIFAR10.RETRIEVAL_IMG = train_data[train_targets<10-num_zs, :]
        CIFAR10.RETRIEVAL_TARGET = encode_onehot(train_targets[train_targets<10-num_zs],10 )

        CIFAR10.QUERY_DATA4ZERO_SHOT = test_data[test_targets>=10-num_zs, :]
        CIFAR10.QUERY_TARGETS4ZERO_SHOT = encode_onehot(test_targets[test_targets>=10-num_zs], 10)

        CIFAR10.DATABASE_DATA4ZERO_SHOT = train_data[train_targets>=10-num_zs, :]
        CIFAR10.DATABASE_TARGETS4ZERO_SHOT = encode_onehot(train_targets[train_targets>=10-num_zs],10 )

    def __init__(self, mode='train',
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.data = CIFAR10.TRAIN_IMG
            self.targets = CIFAR10.TRAIN_TARGET
        elif mode == 'query':
            self.data = CIFAR10.QUERY_IMG
            self.targets = CIFAR10.QUERY_TARGET
        elif mode == 'retrieval':
            self.data = CIFAR10.RETRIEVAL_IMG
            self.targets = CIFAR10.RETRIEVAL_TARGET
        elif mode == 'query4zero_shot':
            self.data = CIFAR10.QUERY_DATA4ZERO_SHOT
            self.targets = CIFAR10.QUERY_TARGETS4ZERO_SHOT
        elif mode == 'database4zero_shot':
            self.data = CIFAR10.DATABASE_DATA4ZERO_SHOT
            self.targets = CIFAR10.DATABASE_TARGETS4ZERO_SHOT
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

        # self.onehot_targets = encode_onehot(self.targets, 10)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.from_numpy(self.targets).float()


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
