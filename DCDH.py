import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
from scipy.linalg import hadamard
from loguru import logger
from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.dcdh as dcdh
from utils.tools import compute_result, CalcTopMap
from tqdm import tqdm
import random
import numpy as np
from torch import nn
import torch.nn.functional as F

def Log(x):
    """
    Log trick for numerical stability
    """

    lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.tensor([0.]).cuda())

    return lt
class DualClasswiseLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, inner_param=0.1, sigma=0.25, use_gpu=True):
        super(DualClasswiseLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.sigma = sigma
        self.inner_param = inner_param
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels):
        """
        Args:
            x: shape of (batch_size, feat_dim).
            labels: shape of (batch_size, ) or (batch_size, 1)
        """

        #   compute L_1 with single constraint.
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        dist_div = torch.exp(-0.5*self.sigma*distmat)/(torch.exp(-0.5*self.sigma*distmat).sum(dim=1, keepdim=True) + 1e-6)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.view(-1, 1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist_log = torch.log(dist_div+1e-6) * mask.float()
        loss = -dist_log.sum() / batch_size

        #   compute L_2 with inner constraint on class centers.
        centers_norm = F.normalize(self.centers, dim=1)
        theta_x = 0.5 * self.feat_dim * centers_norm.mm(centers_norm.t())
        mask = torch.eye(self.num_classes, self.num_classes).bool().cuda()
        theta_x.masked_fill_(mask, 0)
        loss_iner = Log(theta_x).sum() / (self.num_classes*(self.num_classes-1))

        loss_full = loss + self.inner_param * loss_iner
        return loss_full





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
    print("DCDH for zero shot")


    model = dcdh.dcdh(code_length,pretrained=args.pretrain).to(device)
    print('backbone is resnet50')
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    from thop import profile
    input = torch.randn(1, 3, 224, 224)    
    target_label = 0
    target = torch.tensor([target_label])
    one_hot = torch.nn.functional.one_hot(target, num_classes=190)
    flops, params = profile(model, inputs=(input))
    print(f'FLOPs: {flops}')


    model.to(device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)

    # criterion = DualClasswiseLoss(args, code_length)
    criterion = DualClasswiseLoss(num_classes=args.num_classes, inner_param=0.1, sigma=0.25, feat_dim=code_length, use_gpu=True)
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
                labels = torch.argmax(targets, dim=1)
                optimizer.zero_grad()
                u = model(data)
                loss_dual = criterion(u, labels)
                hash_binary = torch.sign(u)
                targets = targets.float()
                W = torch.pinverse(targets.t() @ targets) @ targets.t() @ hash_binary           # Update W
                
                eta = 0.01
                batchB = torch.sign(torch.mm(targets, W) + eta * u)  # Update B

                loss_vertex = (u - batchB).pow(2).sum() / len(data)     # quantization loss
                loss = loss_dual + eta * loss_vertex                
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
            code[index, :] = hash_code.sign()

    model.train()
    return code
