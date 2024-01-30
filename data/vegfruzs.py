# package_paths = [
#     "../data",
#     "/home/xxx/PycharmProjects/IJCAI2022"
# ]
import sys
sys.path.append("/2T/PycharmProjects/FGhash-zero/")
# for pth in package_paths:
#     sys.path.append(pth)
import torch
import numpy as np
# import piexif
# import warnings
# warnings.filterwarnings("error", category=UserWarning)
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image, ImageFile

from data.transform import encode_onehot
from data.transform import train_transform, query_transform

def load_data(root, batch_size, num_workers,zero_shot_classes=15):
    Vegfru.init(root,zero_shot_classes=zero_shot_classes)
    query_dataset = Vegfru(root, 'query', query_transform())
    train_dataset = Vegfru(root, 'train', train_transform())
    retrieval_dataset = Vegfru(root, 'retrieval', query_transform())
    query4zero_shot_dataset = Vegfru(root, 'query4zero_shot', query_transform())
    retrieval4zero_shot_dataset = Vegfru(root, 'retrieval4zero_shot', query_transform())
    # print(len(query_dataset))
    # print(len(train_dataset))

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
    retrieval4zero_shot_dataloader = DataLoader(
        retrieval4zero_shot_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    return query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, retrieval4zero_shot_dataloader


class Vegfru(Dataset):

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = Vegfru.TRAIN_DATA
            self.targets = Vegfru.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Vegfru.QUERY_DATA
            self.targets = Vegfru.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Vegfru.RETRIEVAL_DATA
            self.targets = Vegfru.RETRIEVAL_TARGETS
        elif mode == 'query4zero_shot':
            self.data = Vegfru.QUERY_DATA4ZERO_SHOT
            self.targets = Vegfru.QUERY_TARGETS4ZERO_SHOT
        elif mode == 'retrieval4zero_shot':
            self.data = Vegfru.RETRIEVAL_DATA4ZERO_SHOT
            self.targets = Vegfru.RETRIEVAL_TARGETS4ZERO_SHOT
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')


    @staticmethod
    def init(root,zero_shot_classes = 15):
        if zero_shot_classes%3 != 0:
            raise ValueError(r'Invalid arguments: zero_shot_classes, must be a multiple of 3!')
        #This file use train data (do not combine train and val as a new train dataset).
        images_train = pd.read_csv(os.path.join(root,'vegfru_list/vegfru_train.txt'), sep=' ', names=['filepath', 'target'])
        images_val = pd.read_csv(os.path.join(root,'vegfru_list/vegfru_val.txt'), sep=' ', names=['filepath', 'target'])
        images_test = pd.read_csv(os.path.join(root,'vegfru_list/vegfru_test.txt'), sep=' ', names=['filepath', 'target'])
        train_images = []
        test_images = []
        for i in range(len(images_train)):
            train_images.append(images_train['filepath'][i])
        for i in range(len(images_test)):
            test_images.append(images_test['filepath'][i])
        # print(images_train[:10])
        label_list_train = []
        img_id_train = []
        for i in range(len(images_train)):
            label_list_train.append(int(images_train['target'][i])+1)
            img_id_train.append(i+1)

        # print(label_list_train[:10])
        images_train = []
        # print(len(train_images))
        for i in range(len(train_images)):
            images_train.append([img_id_train[i], train_images[i], label_list_train[i]])
        # print(images_train[:10])
        images_train = pd.DataFrame(images_train, columns=['img_id', 'filepath', 'target'])

        k = len(train_images)
        label_list_test = []
        img_id_test = []
        for i in range(len(test_images)):
            label_list_test.append(int(images_test['target'][i])+1)
            img_id_test.append(k+i+1)
        images_test = []
        for i in range(len(test_images)):
            images_test.append([img_id_test[i], test_images[i], label_list_test[i]])
        # print(images_test[:10])
        images_test = pd.DataFrame(images_test, columns=['img_id', 'filepath', 'target'])

        train_data = images_train
        test_data = images_test
        # zero_shot_classes = 15

        # put zero-shot classes label at the end of the list
        # [know_veg,veg_zs,know_fru,fru_zs] -->[know_veg,know_fru,veg_zs,fru_zs]
        tem_train_target = encode_onehot((train_data['target'] - 1).tolist(), 292)
        tem_test_target = encode_onehot((test_data['target'] - 1).tolist(), 292)
        train_target = np.zeros_like(tem_train_target)
        train_target[:, :200-zero_shot_classes//3*2] = tem_train_target[:, :200-zero_shot_classes//3*2]
        train_target[:,200-zero_shot_classes//3*2: 292-zero_shot_classes] = tem_train_target[:,200: 292-zero_shot_classes//3]
        train_target[:, 292-zero_shot_classes:292-zero_shot_classes//3] = tem_train_target[:,200-zero_shot_classes//3*2: 200]
        train_target[:, 292-zero_shot_classes//3:] = tem_train_target[:, 292-zero_shot_classes//3:]

        test_target = np.zeros_like(tem_test_target)
        test_target[:, :200-zero_shot_classes//3*2] = tem_test_target[:, :200-zero_shot_classes//3*2]
        test_target[:,200-zero_shot_classes//3*2: 292-zero_shot_classes] = tem_test_target[:,200: 292-zero_shot_classes//3]
        test_target[:, 292-zero_shot_classes:292-zero_shot_classes//3] = tem_test_target[:,200-zero_shot_classes//3*2: 200]
        test_target[:, 292-zero_shot_classes//3:] = tem_test_target[:, 292-zero_shot_classes//3:]



        zs_train_index = ((200-zero_shot_classes//3*2 < train_data['target']) & (train_data['target'] <= 200))|((292-zero_shot_classes//3 < train_data['target']) & (train_data['target'] <= 292))

        zs_test_index = ((200-zero_shot_classes//3*2 < test_data['target']) & (test_data['target'] <= 200))|((292-zero_shot_classes//3 < test_data['target']) & (test_data['target'] <= 292))



        Vegfru.QUERY_DATA = test_data['filepath'][~zs_test_index].to_numpy()
        Vegfru.QUERY_TARGETS = test_target[~zs_test_index,:]

        Vegfru.TRAIN_DATA = train_data['filepath'][~zs_train_index].to_numpy()
        Vegfru.TRAIN_TARGETS = train_target[~zs_train_index,:292-zero_shot_classes]

    #     data = {
    # 'path': Vegfru.TRAIN_DATA,
    # 'target': np.argmax(Vegfru.TRAIN_TARGETS, axis=1)
    #     }
    #     df = pd.DataFrame(data)
    #     df.to_csv('train_data.csv', index=False)

    #     images_train.to_csv('train_all.csv', index=False)


        Vegfru.RETRIEVAL_DATA = train_data['filepath'][~zs_train_index].to_numpy()
        Vegfru.RETRIEVAL_TARGETS = train_target[~zs_train_index,:]

        Vegfru.QUERY_DATA4ZERO_SHOT = test_data['filepath'][zs_test_index].to_numpy()
        Vegfru.QUERY_TARGETS4ZERO_SHOT = test_target[zs_test_index,:]

        Vegfru.RETRIEVAL_DATA4ZERO_SHOT = train_data['filepath'][zs_train_index].to_numpy()
        Vegfru.RETRIEVAL_TARGETS4ZERO_SHOT = train_target[zs_train_index,:]

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')
        except:
            img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGBA').convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx

# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("error", category=UserWarning)
# def main():
#     # query_dataloader, train_dataloader, retrieval_dataloader=load_data('/dataset/vegfru/', 16, 4)
#     img_list =['veg200_images/houttuynia_cordata/v_08_18_0350.jpg',
#     'fru92_images/banana/f_01_01_1031.jpg',
#     'fru92_images/carambola/f_01_05_0599.jpg',
#     'fru92_images/grape_white/f_01_10_0875.jpg',
#     'fru92_images/green_dates/f_09_03_0187.jpg',
#     'fru92_images/sugarcane/f_08_01_0863.jpg',
#     'fru92_images/sugarcane/f_08_01_1022.jpg',
#     'veg200_images/kalimeris/v_08_19_0117.jpg']
#     # piexif.remove('/dataset/vegfru/fru92_images/banana/f_01_01_1031.jpg')
#     img = Image.open('/dataset/vegfru/fru92_images/banana/f_01_01_1031.jpg').convert('RGBA').convert('RGB')

# main()
if __name__ == '__main__':
    load_data('/2T/dataset/vegfru', 16, 4)
