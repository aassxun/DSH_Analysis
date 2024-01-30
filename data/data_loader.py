import torch
import os

"""
自己加的

"""


import sys
sys.path.append("/media/xsl/D/peng/fghash-test/")

#####
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import data.cifar10 as cifar10
import data.nus_wide as nuswide
import data.flickr25k as flickr25k
import data.imagenet as imagenet
import data.cub_2011 as cub2011
import data.nuswide_zs as nuswide_zs
import data.nabirds as nabirds
import data.cub_2011_for_zero_shot as cub2011_for_zero_shot
import data.cifar100zs as cifar100zs
import data.cifar10zs as cifar10zs
import data.vegfruzs as vegfruzs
import data.food101zs as food101zs
import data.stanforddogzs as stanforddogzs
import data.aircraftzs as aircraftzs
import data.nabirdszs as nabirdszs
import data.imagenetzs as imagenetzs
from data.transform import train_transform, encode_onehot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, num_query, num_train, batch_size, num_workers, num_zs,sampler=None):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10':
        query_dataloader, train_dataloader, retrieval_dataloader = cifar10.load_data(root,
                                     num_query,
                                     num_train,
                                     batch_size,
                                     num_workers,
                                     )
    elif dataset == 'nus-wide-tc10':
        query_dataloader, train_dataloader, retrieval_dataloader = nuswide.load_data(10,
                                     root,
                                     num_query,
                                     num_train,
                                     batch_size,
                                     num_workers,
                                     )
    elif dataset == 'nus-wide-tc21':
        query_dataloader, train_dataloader, retrieval_dataloader = nuswide.load_data(21,
                                     root,
                                     num_query,
                                     num_train,
                                     batch_size,
                                     num_workers
                                     )
    elif dataset == 'flickr25k':
        query_dataloader, train_dataloader, retrieval_dataloader = flickr25k.load_data(root,
                                       num_query,
                                       num_train,
                                       batch_size,
                                       num_workers,
                                       )
    elif dataset == 'imagenet':
        query_dataloader, train_dataloader, retrieval_dataloader = imagenet.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    elif dataset == 'cub-2011':

        query_dataloader, train_dataloader, retrieval_dataloader = cub2011.load_data(root,
                                    batch_size,
                                    num_workers,
                                    )
    elif dataset == 'cub-2011-for-zero-shot':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = cub2011_for_zero_shot.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader

    elif dataset == 'cifar100zs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = cifar100zs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader
    elif dataset == 'nus-widezs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = nuswide_zs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader
    elif dataset == 'vegfruzs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = vegfruzs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader

    elif dataset == 'cifar10zs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = cifar10zs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader

    elif dataset == 'food101zs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = food101zs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader

    elif dataset == 'stanforddogzs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = stanforddogzs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader

    elif dataset == 'aircraftzs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = aircraftzs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader

    elif dataset == 'nabirdszs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = nabirdszs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader

    elif dataset == 'imagenetzs':
        query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = imagenetzs.load_data(root,
                                      batch_size,
                                      num_workers,
                                      zero_shot_classes=num_zs,
                                      )
        return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader

    elif dataset == 'nabirds':
        query_dataloader, train_dataloader, retrieval_dataloader = nabirds.load_data(root,
                                      batch_size,
                                      num_workers,
                                      )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, train_dataloader, retrieval_dataloader


def sample_dataloader(dataloader, num_samples, batch_size, root, dataset):
    """
    Sample data from dataloder.

    Args
        dataloader (torch.utils.data.DataLoader): Dataloader.
        num_samples (int): Number of samples.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        sample_index (int): Sample index.
        dataset(str): Dataset name.

    Returns
        sample_dataloader (torch.utils.data.DataLoader): Sample dataloader.
    """
    data = dataloader.dataset.data
    targets = dataloader.dataset.targets

    sample_index = torch.randperm(data.shape[0])[:num_samples]
    data = data[sample_index]
    targets = targets[sample_index]
    sample = wrap_data(data, targets, batch_size, root, dataset)

    return sample, sample_index


def wrap_data(data, targets, batch_size, root, dataset):
    """
    Wrap data into dataloader.
    Args
        data (np.ndarray): Data.
        targets (np.ndarray): Targets.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        dataloader (torch.utils.data.dataloader): Data loader.
    """
    class MyDataset(Dataset):
        def __init__(self, data, targets, root, dataset):
            self.data = data
            self.targets = targets
            self.root = root
            if dataset == 'cifar100zs' or dataset == 'cifar10zs':
                self.transform = transform_train_cifar()

            else:
                self.transform = train_transform()
            self.dataset = dataset
            if dataset == 'cifar-10':
                self.onehot_targets = encode_onehot(self.targets, 10)
            else:
                self.onehot_targets = self.targets

        def __getitem__(self, index):
            if self.dataset == 'cifar10zs' :
                img = Image.fromarray(self.data[index])
                if self.transform is not None:
                    img = self.transform(img)
            elif self.dataset == 'cifar100zs':

                    r = self.data[index, :1024].reshape(32, 32)
                    g = self.data[index, 1024:2048].reshape(32, 32)
                    b = self.data[index, 2048:].reshape(32, 32)
                    img = np.dstack((r, g, b))
                    # print(img.shape)
                    img = Image.fromarray(img)
                    img = self.transform(img)
            else:
                img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
                img = self.transform(img)
            return img, self.targets[index], index

        def __len__(self):
            return self.data.shape[0]

        def get_onehot_targets(self):
            """
            Return one-hot encoding targets.
            """
            return torch.from_numpy(self.onehot_targets).float()

    dataset = MyDataset(data, targets, root, dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader



#  only for cifar100zs
# CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
def transform_train_cifar():
    # normalize = transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
    #                                  std=CIFAR100_TRAIN_STD)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
if __name__ == "__main__":
    dataset = "cifar-10"
    root ="/media/xsl/D/peng/fghash-test/dataset/cifar-10-python"
    # num_query = 1000
    # num_samples = 2000
    num_query = 10
    num_samples = 20
    batch_size = 64
    num_workers = 0
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(
            dataset,
            root,
            num_query,
            num_samples,#num_train
            batch_size,
            num_workers
        )
    print("0")
