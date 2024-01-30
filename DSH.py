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
from tqdm import tqdm
import time
import numpy as np


class DSHLoss(torch.nn.Module):
    def __init__(self, args, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(args.num_samples, bit).float().to(args.device)
        self.Y = torch.zeros(args.num_samples, args.num_classes).float().to(args.device)

    def forward(self, u, y, ind, alpha=0.01):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = alpha * (1 - u.abs()).abs().mean()

        return loss1 + loss2



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


    device = args.device
    args.num_train = len(train_loader.dataset)
    # args.step_continuation = 20
    print("DSH for zero shot")
    model = resnet.resnet50(pretrained=args.pretrain, num_classes=code_length,with_tanh = False)

    print('backbone is resnet50')


    model.to(device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)



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
        criterion = DSHLoss(args, code_length)
#,train_dataloader.dataset.get_onehot_targets()

        for epoch in range(args.max_epoch):
            # epoch_start = time.time()

            # criterion.scale = (epoch // args.step_continuation + 1) ** 0.5

            model.train()
            # losses.reset()
            train_loss = 0
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader),ncols = 50)
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:
            # for batch, (data, targets, index) in enumerate(train_dataloader):
                # print(targets.shape)
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                optimizer.zero_grad()
                u = model(data)
              #  criterion.U[index,:] = u.data
                loss = criterion(u, targets.float(), index)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            logger.info('[epoch:{}/{}][loss:{:.6f}]'.format(epoch+1, args.max_epoch, train_loss/len(train_dataloader)))

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
