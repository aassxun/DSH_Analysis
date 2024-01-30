import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import utils.evaluate as evaluate
import models.resnet as resnet
from loguru import logger
from data.data_loader import sample_dataloader
import math
from tqdm import tqdm
import pandas as pd


class HyP(torch.nn.Module):
    def __init__(self, num_classes, num_bits,device,beta = 0.5, threshold = 0.5):
        torch.nn.Module.__init__(self)
        # torch.manual_seed(seed)
        # Initialization
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, num_bits).to(device))
        nn.init.kaiming_normal_(self.proxies, mode = 'fan_out')
        self.beta = beta
        self.threshold = threshold

    def forward(self, x = None, batch_y = None):
        P_one_hot = batch_y

        cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos = 1 - cos
        neg = F.relu(cos - self.threshold)

        P_num = len(P_one_hot.nonzero())
        N_num = len((P_one_hot == 0).nonzero())
        pos_term = torch.where(P_one_hot  ==  1, pos.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot  ==  0, neg.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / N_num
        if self.beta > 0:
            index = batch_y.sum(dim = 1) > 1
            y_ = batch_y[index].float()
            x_ = x[index]
            cos_sim = y_.mm(y_.T)
            if len((cos_sim == 0).nonzero()) == 0:
                reg_term = 0
            else:
                x_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)
                neg = self.beta * F.relu(x_sim - self.threshold)
                reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
        else:
            reg_term = 0

        return pos_term + neg_term + reg_term


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
    print('hpy2 for zero shot')
    sheet = pd.read_excel('codetable.xlsx', engine='openpyxl',header=None)

    threshold = sheet.iloc[code_length,math.ceil(math.log(args.num_classes, 2))]

    model = resnet.resnet50(pretrained=args.pretrain, num_classes=code_length,with_tanh = False)
    criterion = HyP(args.num_classes, code_length, device, beta = 0.5, threshold = threshold)
    model.to(device)
    criterion.to(device)
    if args.optim == 'SGD':
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr':args.lr}, {'params': criterion.parameters(), 'lr':args.criterion_rate}], momentum = 0.9, weight_decay = 0.0005)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    start = time.time()
    best_mAP = 0
    corresponding_mAP_all = 0
    corresponding_zs_mAP = 0
    corresponding_zs_mAP_all = 0
    '''drop_cls = 0
    ind = np.argmax(train_dataloader.dataset.targets, 1) != drop_cls
    train_dataloader.dataset.data = train_dataloader.dataset.data[ind]
    train_dataloader.dataset.targets = train_dataloader.dataset.targets[ind]
    ind = np.argmax(query_dataloader.dataset.targets, 1) != drop_cls
    query_dataloader.dataset.data = query_dataloader.dataset.data[ind]
    query_dataloader.dataset.targets = query_dataloader.dataset.targets[ind]
    args.topk = ind.sum()'''
    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(train_loader, args.num_samples, args.batch_size, args.root, args.dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)

        # Training CNN model
        model.train()
        for epoch in range(args.max_epoch):

            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:
                data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                optimizer.zero_grad()

                u = model(data)
                # loss = criterion(u,targets.float())
                loss = criterion(u,targets)

                loss.backward()
                optimizer.step()

            logger.info('[epoch:{}/{}][loss:{:.6f}]'.format(epoch+1, args.max_epoch, loss.item()))


            scheduler.step()
        # scheduler.step()

        # logger.info('[iter:{}/{}][iter_time:{:.2f}]'.format(it+1, args.max_iter, time.time()-iter_start))
        logger.info('[iter:{}/{}][iter_time:{:.2f}]'.format(it+1, args.max_iter, time.time()-iter_start))

        # Evaluation
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
