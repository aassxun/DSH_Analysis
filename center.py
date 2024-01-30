import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
import models.resnet as resnet
from scipy.linalg import hadamard
from loguru import logger
from models.adsh_loss import ADSH_Loss
from data.data_loader import sample_dataloader
from utils import AverageMeter
from utils.tools import compute_result, CalcTopMap
from scipy.special import comb
import torch.nn.functional as F
import random
from tqdm import tqdm
import numpy as np
import torch.nn as NN
from models.center_resnet import ResNet as center_resnet
from utils.optimizeAccel import get_hash_centers

class OurLoss(NN.Module):
    def __init__(self,  bit, num_classes, device,hash_center_root=None,epoch_change=1):
        """
        :param config: in paper, the hyper-parameter lambda is chose 0.0001
        :param bit:
        """
        super(OurLoss, self).__init__()
        self.epoch_change = epoch_change
        # self.config = config
        self.bit = bit
        l = list(range(num_classes))
        self.hash_center = self.generate_center(bit, num_classes, l, hash_center_root)
        print(f"hash center shape is {self.hash_center.shape}")
        # self.Y = torch.randn(self.config['num_train'], self.num_classes).float().to(device)
        # self.U = torch.randn(config['num_train'], bit).to(device)
        self.label_center = torch.from_numpy(
            np.eye(num_classes,dtype=np.float64)[np.array([i for i in range(num_classes)])]).to(device)


    def forward(self, u1, y, ind, U , Y,k=0):
        # u1 = u1.tanh()
        # self.U[ind, :] = u2.data
        # self.Y[ind, :] = y
        return self.cos_pair(u1, y, ind, U,Y,k)


    def cos_pair(self, u, y, ind,U,Y, k):
        if k < self.epoch_change:
            pair_loss = torch.tensor([0]).to(u.device)
        else:
            last_u = U
            last_y = Y
            pair_loss = self.moco_pairloss(u, y, last_u, last_y, ind)
        cos_loss = self.cos_eps_loss(u, y, ind)
        Q_loss = (u.abs() - 1).pow(2).mean()
        # print(cos_loss.device, pair_loss.device, Q_loss.device)
        loss = 10*cos_loss +  pair_loss + 0.0001 * Q_loss

        return loss, cos_loss, pair_loss

    def moco_pairloss(self, u, y, last_u, last_y, ind):
        u = F.normalize(u)
        last_u = F.normalize(last_u)

        last_sim = ((y.float() @ last_y.t()) > 0).float()
        last_cos = u @ last_u.t()

        loss = torch.sum(last_sim * torch.log(1 + torch.exp(1/2 *(1 - last_cos))))/torch.sum(last_sim) # only the positive pair
        return loss

    def cos_eps_loss(self, u, y, ind):
        K = self.bit
        u_norm = F.normalize(u).to(u.device)
        centers_norm = F.normalize(self.hash_center).to(u.device)
        # print(centers_norm.device,u_norm.device)
        cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_c  lass
        # print(y.dtype,self.label_center.dtype)
        s = (y @ self.label_center.t()).float() # batch x n_class
        cos_sim = K ** 0.5 * cos_sim
        p = torch.softmax(cos_sim, dim=1)
        loss = s * torch.log(p) + (1-s) * torch.log(1-p)
        loss = torch.mean(loss)
        return -loss


    def generate_center(self, bit, n_class, l,root):
        # hash_centers = np.load(self.config['center_path'])
        if os.path.exists(root):
            hash_centers = np.load(root)
            print(f"load hash center from {root}")

        else:
            hash_centers = get_hash_centers(bit, n_class)
        self.evaluate_centers(hash_centers)
        hash_centers = hash_centers[l]
        Z = torch.from_numpy(hash_centers).float()
        return Z

    def evaluate_centers(self, H):
        dist = []
        for i in range(H.shape[0]):
            for j in range(i+1, H.shape[0]):
                    TF = np.sum(H[i] != H[j])
                    dist.append(TF)
        dist = np.array(dist)
        st = dist.mean() - dist.var() + dist.min()
        print(f"mean is {dist.mean()}; min is {dist.min()}; var is {dist.var()}; max is {dist.max()}")





def train(
        test_loader,
        train_loader,
        database_loader,
        query_loader_zs,
        database_loader_zs,
        code_length,
        args,
):
    """
    Training model.

    Args
        test_loader, database_loader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        args.device(torch.args.device): GPU or CPU.
        lr(float): Learning rate.
    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    # model = alexnet.load_model(code_length).to(args.device)
    device = args.device
    args.num_train = len(train_loader.dataset)
    # args.step_continuation = 20
    print("center hash for zero shot")

    print('backbone is resnet50')
    model = resnet.resnet50(pretrained=True, num_classes=code_length)

    model.to(device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen)

    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)
    hash_center_root = os.path.join('hash_centers', f"num_classes-{args.num_classes}-bit-{code_length}.npy")

    criterion = OurLoss(code_length, args.num_classes, device, hash_center_root,args.epoch_change).to(device)


    # losses = AverageMeter()
    start = time.time()
    best_mAP = 0
    corresponding_mAP_all = 0
    corresponding_zs_mAP = 0
    corresponding_zs_mAP_all = 0



    print("start training")#*******************************************wp

    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(train_loader, args.num_samples, args.batch_size, args.root, args.dataset)
        Y = torch.randn(args.num_samples, args.num_classes).float().to(device)
        U = torch.randn(args.num_samples, code_length).to(device)

        for epoch in range(args.max_epoch):
            # epoch_start = time.time()

            # criterion.scale = (epoch // args.step_continuation + 1) ** 0.5

            model.train()
            train_loss = 0
            train_center_loss = 0
            train_pair_loss = 0
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader),ncols = 50)
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:

            # for batch, (data, targets, index) in enumerate(train_dataloader):
                # print(targets.shape)
                data, targets, index = data.to(device), targets.to(device), index.to(device)


                optimizer.zero_grad()

                u = model(data)
                U[index, :] = u.data
                Y[index, :] = targets.float()
                loss, center_loss, pair_loss = criterion(u, targets, index,U,Y, epoch)
                train_loss += loss.item()
                train_center_loss += center_loss.item()
                train_pair_loss += pair_loss.item()
                # print(pair_loss.item())
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            train_pair_loss /= len(train_dataloader)
            # print(f"pair loss is {train_pair_loss}")
            train_loss /= len(train_loader)

            logger.info('[epoch:{}/{}][loss:{:.6f}][pair_loss:{:.6f}]'.format(epoch+1, args.max_epoch, train_loss,train_pair_loss))

        scheduler.step()
        logger.info('[iter:{}/{}][iter_time:{:.2f}]'.format(it+1, args.max_iter,
                             time.time()-iter_start))
        if (it < 35 and (it + 1) % args.val_freq == 0) or (it >= 35 and (it + 1) % 1 == 0):
        # if (it + 1) % 1 == 0 :

            query_code = generate_code(model, test_loader, code_length, args.device)
            query_targets = test_loader.dataset.get_onehot_targets()
            B = generate_code(model, database_loader, code_length, args.device)
            db_label= database_loader.dataset.get_onehot_targets()

            zs_test_binary = generate_code(model, query_loader_zs, code_length, args.device)
            zs_test_label = query_loader_zs.dataset.get_onehot_targets()

            zs_db_binary = generate_code(model, database_loader_zs, code_length, args.device)
            zs_db_label = database_loader_zs.dataset.get_onehot_targets()

            db_all_binary = torch.cat((B, zs_db_binary), 0)
            db_all_label = torch.cat((db_label, zs_db_label), 0)

            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                B,
                query_targets[:,:args.num_classes].to(args.device),
                db_label[:,:args.num_classes].to(args.device),
                args.device,
                args.topk,
            )
            mAP_all = evaluate.mean_average_precision(
                query_code.to(args.device),
                db_all_binary.to(args.device),
                query_targets.to(args.device),
                db_all_label.to(args.device),
                args.device,
                args.topk,
            )
            zs_mAP_all = evaluate.mean_average_precision(
                zs_test_binary.to(args.device),
                db_all_binary.to(args.device),
                zs_test_label.to(args.device),
                db_all_label.to(args.device),
                args.device,
                args.topk,
            )
            zs_mAP = evaluate.mean_average_precision(
                zs_test_binary.to(args.device),
                zs_db_binary.to(args.device),
                zs_test_label.to(args.device),
                zs_db_label.to(args.device),
                args.device,
                args.topk,
            )


            if mAP > best_mAP:
                best_mAP = mAP
                corresponding_mAP_all = mAP_all
                corresponding_zs_mAP = zs_mAP
                corresponding_zs_mAP_all = zs_mAP_all
                ret_path = os.path.join('checkpoints', args.info, 'best_mAP',str(code_length))
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.t'))
                torch.save(B.cpu(), os.path.join(ret_path, 'database_code.t'))
                torch.save(query_targets.cpu(), os.path.join(ret_path, 'query_targets.t'))
                torch.save(db_label.cpu(), os.path.join(ret_path, 'database_targets.t'))
                torch.save(zs_test_binary.cpu(), os.path.join(ret_path, 'zs_test_binary.t'))
                torch.save(zs_db_binary.cpu(), os.path.join(ret_path, 'zs_db_binary.t'))
                torch.save(zs_test_label.cpu(), os.path.join(ret_path, 'zs_test_label.t'))
                torch.save(zs_db_label.cpu(), os.path.join(ret_path, 'zs_db_label.t'))
                torch.save(model.state_dict(), os.path.join(ret_path, 'model.pkl'))
                model = model.to(args.device)
            logger.info('[iter:{}/{}][code_length:{}][mAP:{:.5f}][mAP_all:{:.5f}][best_mAP:{:.5f}]'.format(it+1, args.max_iter, code_length, mAP,mAP_all ,best_mAP))
            logger.info('[iter:{}/{}][code_length:{}][zs_mAP:{:.5f}][zs_mAP_all:{:.5f}]'.format(it+1, args.max_iter, code_length, zs_mAP, zs_mAP_all))


    logger.info('[Training time:{:.2f}]'.format(time.time()-start))


    return best_mAP, corresponding_mAP_all, corresponding_zs_mAP, corresponding_zs_mAP_all



def generate_code(model, dataloader, code_length, device):
    # query_dataloader  wp
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """

    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length]).to(device)
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign()

    model.train()
    return code
