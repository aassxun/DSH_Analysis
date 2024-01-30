import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
from loguru import logger
from data.data_loader import sample_dataloader
from utils import AverageMeter
from utils.tools import compute_result, CalcTopMap
import random
import numpy as np
import models.resnet as resnet
from torch.autograd import Variable
import torch.nn as nn
from models.psldh_model import Label_net, hash_net
from tqdm import tqdm
import pickle
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
    # args.num_train = len(train_loader.dataset)
    # args.step_continuation = 20
    print("PSLDH for zero shot")


    nclass = args.num_classes
    psldh_label_code_path = os.path.join('hash_centers','psldh_label_code')
    if os.path.exists(os.path.join(psldh_label_code_path, f'{args.dataset}-code_length{code_length}-numzs{args.num_zs}.t')):
        label_code = torch.load(os.path.join(psldh_label_code_path, f'{args.dataset}-code_length{code_length}-numzs{args.num_zs}.t'))

    else:
        label_model = Label_net(nclass, code_length)
        label_model.to(device)

        optimizer_label = optim.SGD(label_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=100, gamma=0.1, last_epoch=-1)

        labels = torch.zeros((nclass, nclass)).type(torch.FloatTensor).to(device)
        for i in range(nclass):
            labels[i, i] = 1

        labels = Variable(labels)
        one_hot = Variable(torch.ones((1, nclass)).type(torch.FloatTensor).to(device))
        I = Variable(torch.eye(nclass).type(torch.FloatTensor).to(device))
        relu = nn.ReLU()
        for i in range(200):
            scheduler_l.step()
            code = label_model(labels)
            loss1 = relu((code.mm(code.t()) - code_length * I))
            loss1 = loss1.pow(2).sum() / (nclass * nclass)
            loss_b = one_hot.mm(code).pow(2).sum() / nclass
            re = (torch.sign(code) - code).pow(2).sum() / nclass
            loss_label = loss1 + 0.05 * loss_b + 0.01 * re
            optimizer_label.zero_grad()
            loss_label.backward()
            optimizer_label.step()
        label_model.eval()
        code = label_model(labels)
        label_code = torch.sign(code)

        # del label_model,optimizer_label,scheduler_l,labels,one_hot,I,relu,code,loss1,loss_b,re,loss_label

        if not os.path.exists(psldh_label_code_path):
            os.makedirs(psldh_label_code_path)
        torch.save(label_code.cpu(),
                   os.path.join(psldh_label_code_path,f'{args.dataset}-code_length{code_length}-numzs{args.num_zs}.t'))

    # label_code = label_code.to(device)
    label_code = torch.load(os.path.join(psldh_label_code_path, f'{args.dataset}-code_length{code_length}-numzs{args.num_zs}.t')).to(device)



    model = hash_net(classes=code_length,pretrained=args.pretrain)

    model.to(device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, last_epoch=-1)
    start = time.time()
    best_mAP = 0
    corresponding_mAP_all = 0
    corresponding_zs_mAP = 0
    corresponding_zs_mAP_all = 0



    print("start training")#*******************************************wp

    gamma = 0.2
    sigma = 0.2
    lamda = 0.01
    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(train_loader, args.num_samples, args.batch_size, args.root, args.dataset)


        for epoch in range(args.max_epoch):
            epoch_loss = 0.0
            epoch_loss_r = 0.0
            epoch_loss_e = 0.0

            model.train()
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader),ncols = 50)
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:
            # for batch, (data, targets, index) in enumerate(train_dataloader):
                # print(targets.shape)
                optimizer.zero_grad()
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                hash_out = model(data)
                the_batch = len(index)
                logit = hash_out.mm(label_code.t())

                our_logit = torch.exp((logit - sigma * code_length) * gamma) * targets
                mu_logit = (torch.exp(logit * gamma) * (1 - targets)).sum(1).view(-1, 1).expand(the_batch, targets.size()[1]) + our_logit
                loss = - ((torch.log(our_logit / mu_logit + 1 - targets)).sum(1) / targets.sum(1)).sum()

                Bbatch = torch.sign(hash_out)
                regterm = (Bbatch - hash_out).pow(2).sum()
                loss_all = loss / the_batch + regterm * lamda / the_batch


                loss_all.backward()
                optimizer.step()
                epoch_loss += loss_all.item()
                epoch_loss_e += loss.item() / the_batch
                epoch_loss_r += regterm.item() / the_batch

            logger.info('[epoch:{}/{}][loss:{:.6f}][loss_e:{:.6f}][loss_r:{:.6f}]'.format(epoch+1,
            args.max_epoch, epoch_loss / len(train_loader), epoch_loss_e / len(train_loader),
               epoch_loss_r / len(train_loader)))

        scheduler.step()
        logger.info('[iter:{}/{}][iter_time:{:.2f}]'.format(it+1, args.max_iter,
                             time.time()-iter_start))

        # if (it + 1) % 1 == 0 :
        if (it < 35 and (it + 1) % args.val_freq == 0) or (it >= 35 and (it + 1) % 1 == 0):
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
