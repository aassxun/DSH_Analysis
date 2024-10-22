import torch
from torch import nn
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
import models.dpah as dpah
from loguru import logger
from data.data_loader import sample_dataloader
from utils import AverageMeter
from utils.tools import compute_result, CalcTopMap
from tqdm import tqdm
import numpy as np

class SupLoss1(nn.Module):

    def __init__(self, gamma, alpha, beta):
        super(SupLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.softmax = nn.CrossEntropyLoss()

    def forward(self, yh, yc, label):
        # pdb.set_trace()
        batch_size = yh.size(0) * yh.size(1)
        #print ('*****************',type(label),'*********************')
        loss1 = self.softmax(yc, label)
        loss2 = (yh.mean(dim=1) - 0.5)**2
        loss3 = -(yh - 0.5)**2
        # loss = self.alpha * loss1 + self.gamma * loss2.sum() / (2 * yh.size(0)) + \
        #     self.beta * loss3.sum() / (2 * batch_size)
        return self.alpha * loss1, self.gamma * loss2.sum() / (2 * yh.size(0)), self.beta * loss3.sum() / (2 * batch_size)

class SVRLoss(nn.Module):

    def __init__(self, center, eta):
        super(SVRLoss, self).__init__()
        self.eta = eta
        self.center = center
        self.softmax = nn.CrossEntropyLoss()

    def forward(self, yh, yc, label):
        loss1 = self.softmax(yc, label)
        # ind1 = (yh > 1)
        # ind2 = (yh < -1)
        # ind3 = (yh > -1) & (yh < 1)
        ind1 = (yh > 0)
        ind2 = (yh < 0)
        # ipdb.set_trace()
        loss2 = (yh.mean(dim=0) - 0.0)**2
        loss3 = ((yh[ind1] - 1) ** 2).sum() + ((yh[ind2] + 1) ** 2).sum()
        #loss3 = ((yh[ind1] - 1) ** 2).sum() + ((yh[ind2] + 1) ** 2).sum() + (-(yh[ind3]).abs() + 1).sum()
        return loss1, self.center * loss2.sum() / (2 * yh.size(1)), self.eta * loss3 / (yh.size(0) * 2 * 2)


class KTLoss(nn.Module):

    def __init__(self, a):
        super(KTLoss, self).__init__()
        self.range_a = a

    def forward(self, yh):
        ind1 = (yh > self.range_a)
        ind2 = (yh < -self.range_a)
        return (((yh[ind1] - 2)**2).sum() + ((yh[ind2] + 2)**2).sum()) / (2 * 2 * yh.size(0))

class multilabel_sw_Loss(nn.Module):
    """ SW + CCC """

    def __init__(self, gamma, alpha, beta):
        super(multilabel_sw_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, yh, yc, label):
        # pdb.set_trace()
        N = yh.size(0)
        D = yh.size(1)
        batch_size = N * D
        log_p = nn.functional.log_softmax(yc, dim=1)
        # ipdb.set_trace()
        #print ('*****************',type(label),'*********************')
        loss1 = -log_p[:, -1].mean()

        loss2 = (yh.mean(dim=1) - 0.5)**2
        loss3 = -(yh - 0.5)**2
        return self.alpha * loss1, self.gamma * loss2.sum() / (2 * yh.size(0)), self.beta * loss3.sum() / (2 * batch_size)



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

    if args.max_iter == 50:
        text_step=45
    elif args.max_iter == 40:
        text_step =35
    device = args.device
    args.num_train = len(train_loader.dataset)
    # args.step_continuation = 20
    print("DPAH for zero shot")

    model = dpah.dpah(code_length=code_length,pretrained=args.pretrain,num_classes=args.num_classes)
    print('backbone is resnet50')

    model.to(device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)

    # criterion = CSQLoss(args, code_length)
    if args.dataset == 'cocozs' or args.dataset == 'nus-widezs' or args.dataset == 'flickr25kzs':
        svr_loss = multilabel_sw_Loss(1.0, 1.0, 1.0).cuda()
    else:
        svr_loss = SupLoss1(1.0, 1.0, 1.0).to(device)
    KT_loss = KTLoss(10.0)
    losses = AverageMeter()
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


        for epoch in range(args.max_epoch):
            # epoch_start = time.time()

            # criterion.scale = (epoch // args.step_continuation + 1) ** 0.5

            model.train()
            losses.reset()
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader),ncols = 50)
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:
            # for batch, (data, targets, index) in enumerate(train_dataloader):
                # print(targets.shape)
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                train_label=torch.argmax(targets, dim=1)
                optimizer.zero_grad()
                u = model(data)
                ktloss = KT_loss(u)
                yh = torch.sigmoid(u)
                yc, loss_maxlikehood = model.imgs_semantic(yh, train_label)
                cls_loss, hashloss1, hashloss2 = svr_loss(yh, yc, train_label)
                loss = loss_maxlikehood + cls_loss + hashloss1 + hashloss2 + 0.01 * ktloss
                losses.update(loss.item())
                loss.backward()
                optimizer.step()

            logger.info('[epoch:{}/{}][loss:{:.6f}]'.format(epoch+1, args.max_epoch, losses.avg))

        scheduler.step()
        logger.info('[iter:{}/{}][iter_time:{:.2f}]'.format(it+1, args.max_iter,
                             time.time()-iter_start))
        if (it < text_step and (it + 1) % args.val_freq == 0) or (it >= text_step and (it + 1) % 1 == 0):
        # if (it + 1) % 1 == 0 :

            query_code = generate_code(model, test_loader, code_length, args.device)
            query_targets = test_loader.dataset.get_onehot_targets()
            B = generate_code(model, database_loader, code_length, args.device)
            db_label= database_loader.dataset.get_onehot_targets()
            # if args.num_zs != 0:
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
            # if args.num_zs != 0:
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
            code[index, :] = torch.sign(torch.sigmoid(hash_code) - 0.5)
    model.train()
    return code
