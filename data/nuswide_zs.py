import torch
import os
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, batch_size, num_workers,zero_shot_classes):
    """
    Loading nus-wide dataset.

    Args:
        tc(int): Top class.
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """

    query_dataset = NusWideDatasetTC21(
        root,
        'test_img.txt',
        'test_label_onehot.txt',
        zs = False,
        transform=query_transform(),
        zero_shot_classes=zero_shot_classes,
    )
    train_dataset = NusWideDatasetTC21(
        root,
        'database_img.txt',
        'database_label_onehot.txt',
        zs = False,
        drop= True,
        transform=train_transform(),
        zero_shot_classes=zero_shot_classes,
    )
    retrieval_dataset = NusWideDatasetTC21(
        root,
        'database_img.txt',
        'database_label_onehot.txt',
        zs = False,
        transform=query_transform(),
        zero_shot_classes=zero_shot_classes,
    )

    query4zero_shot_dataset = NusWideDatasetTC21(
        root,
        'test_img.txt',
        'test_label_onehot.txt',
        zs = True,
        transform=query_transform(),
        zero_shot_classes=zero_shot_classes,
    )


    database4zero_shot_dataset = NusWideDatasetTC21(
        root,
        'database_img.txt',
        'database_label_onehot.txt',
        zs = True,
        transform=query_transform(),
        zero_shot_classes=zero_shot_classes,
    )


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
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    query4zero_shot_dataloader = DataLoader(
        query4zero_shot_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    database4zero_shot_dataloader = DataLoader(
        database4zero_shot_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    return query_dataloader, train_dataloader, retrieval_dataloader, query4zero_shot_dataloader, database4zero_shot_dataloader



class NusWideDatasetTC21(Dataset):
    """
    Nus-wide dataset, 21 classes.

    Args
        root(str): Path of image files.
        img_txt(str): Path of txt file containing image file name.
        label_txt(str): Path of txt file containing image label.
        transform(callable, optional): Transform images.
        train(bool, optional): Return training dataset.
        num_train(int, optional): Number of training data.
    """
    def __init__(self, root, img_txt, label_txt,zs, drop = False,transform=None, zero_shot_classes=11,train=None, num_train=None):
        self.root = root
        self.transform = transform

        img_txt_path = os.path.join(root, img_txt)
        label_txt_path = os.path.join(root, label_txt)
        class4train = 21-zero_shot_classes
        label = np.loadtxt(label_txt_path, dtype=np.float32)
        idx = label[:,:class4train].sum(axis=1)== 0 ###
        if zs:
        # Read files
            with open(img_txt_path, 'r') as f:
                self.data = np.array([i.strip() for i in f])[idx]
            self.targets = np.loadtxt(label_txt_path, dtype=np.float32)[idx]
            if drop:
                self.targets = self.targets[:,class4train:]
        else:
            idx = idx==0
            with open(img_txt_path, 'r') as f:
                self.data = np.array([i.strip() for i in f])[idx]
            self.targets = np.loadtxt(label_txt_path, dtype=np.float32)[idx]
            if drop:
                self.targets = self.targets[:,:class4train]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()
