import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.transform import encode_onehot
from data.transform import train_transform, query_transform

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

    query_dataset = ImageList(
        root,
        os.path.join(root,'Data','CLS-LOC','val'),
        os.path.join(root,'imagelist','test.txt'),
        zs = False,
        transform=query_transform(),
        zero_shot_classes=zero_shot_classes,
    )
    train_dataset = ImageList(
        root,
        os.path.join(root,'Data','CLS-LOC','train'),
        os.path.join(root,'imagelist','train.txt'),
        zs = False,
        drop= True,
        transform=train_transform(),
        zero_shot_classes=zero_shot_classes,
    )
    retrieval_dataset = ImageList(
        root,
        os.path.join(root,'Data','CLS-LOC','train'),
        os.path.join(root,'imagelist','database.txt'),
        zs = False,
        transform=query_transform(),
        zero_shot_classes=zero_shot_classes,
    )

    query4zero_shot_dataset = ImageList(
        root,
        os.path.join(root,'Data','CLS-LOC','val'),
        os.path.join(root,'imagelist','test.txt'),
        zs = True,
        transform=query_transform(),
        zero_shot_classes=zero_shot_classes,
    )


    database4zero_shot_dataset = ImageList(
        root,
        os.path.join(root,'Data','CLS-LOC','train'),
        os.path.join(root,'imagelist','database.txt'),
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








class ImageList(Dataset):
    def __init__(self, root,data_path, image_list_path, zs, drop = False, transform =None ,zero_shot_classes=5):
        self.root = root
        class4train = 100 - zero_shot_classes
        with open(image_list_path,"r") as f:
            image_list = f.readlines()
        if 'train' in data_path:
            all_data = [os.path.join(data_path,val.strip().split(' ')[0].split('_')[0],val.strip().split(' ')[0]) for val in image_list]
        else:
            all_data = [os.path.join(data_path,val.strip().split(' ')[0]) for val in image_list]

        self.transform = transform

        all_targets = np.array([list(map(int, val.strip().split(' ')[1:])) for val in image_list])
        labels = np.argmax(all_targets, axis=1)
        zs_idx = labels >= class4train
        if zs:
            self.data = np.array(all_data)[zs_idx]
            self.targets = all_targets[zs_idx]
        else:
            self.data = np.array(all_data)[~zs_idx]
            if drop:
                self.targets = all_targets[~zs_idx][:, :class4train]
            else:
                self.targets = all_targets[~zs_idx]

        # self.n_class = n_class
    def __getitem__(self, index):

        path = os.path.join(self.root, self.data[index])
        target = self.targets[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # target = np.eye(self.n_class, dtype=np.float32)[np.array(target)]
        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()





if __name__ == '__main__':
    data_path = '/2T/dataset/ImageNet/ILSVRC'
    load_data(data_path, 32, 4, 5)
