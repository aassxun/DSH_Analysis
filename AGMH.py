import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
import models.resnet as resnet

from tqdm import tqdm
from loguru import logger

from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.agmh as agmh


import torch
import torch.nn as nn
class ADSH_Loss(nn.Module):
    def __init__(self, code_length, gamma):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0]) / self.code_length * 12
        quantization_loss = ((F - B[omega, :]) ** 2).sum() / (F.shape[0] * B.shape[0]) * self.gamma / self.code_length * 12

        # loss = hash_loss + quantization_loss
        return hash_loss, quantization_loss

class Feat_Loss(nn.Module):
    def __init__(self):
        super(Feat_Loss, self).__init__()

    def forward(self, back_feat, feat_group, attn_group, device):
        top_k = len(attn_group)
        B, C, H, W = attn_group[0].shape
        batch_loss = 0

        sm_list = []
        for k in range(top_k):
            attn = attn_group[k]
            cur_max = attn.max(dim=1)[0]
            cur_flat = cur_max.reshape(B, H * W)
            cur_sm = cur_flat.softmax(dim=1)
            sm_list.append(cur_sm)

        for i in range(top_k):
            cur_sm = sm_list[i]
            for j in range(i + 1, top_k):
                com_sm = sm_list[j]
                batch_loss += (cur_sm * com_sm).sum()
        cal_num = top_k * (top_k - 1) // 2
        feat_loss = batch_loss / (cal_num * B)

        return feat_loss



def train(query_dataloader, train_loader, retrieval_dataloader,
          query_loader_zs,database_loader_zs, code_length, args):
    print("AGMH for zero-shot learning")
    if args.max_iter == 50:
        text_step=45
    elif args.max_iter == 40:
        text_step =35
    num_classes, att_size, feat_size = args.num_classes, 1, 2048
    model = agmh.agmh(code_length=code_length, num_classes=num_classes, att_size=att_size,
                      feat_size=feat_size, device=args.device, pretrained=True)
    model.to(args.device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=True)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1)
    criterion = ADSH_Loss(code_length, args.gamma)
    feat_criter = Feat_Loss()

    U = torch.zeros(args.num_samples, code_length).to(args.device)
    if args.dataset == 'imagenetzs':
        num_B = len(train_loader.dataset)
        B = torch.randn(num_B, code_length).to(args.device)
        B_tragets = train_loader.dataset.get_onehot_targets().to(args.device)
    else:
        num_retrieval = len(retrieval_dataloader.dataset)
        B = torch.randn(num_retrieval, code_length).to(args.device)
        retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()[:,:args.num_classes].to(args.device)

    # num_retrieval = len(retrieval_dataloader.dataset) #len = train data

    # B = torch.randn(num_retrieval, code_length).to(args.device)#每个初始化满足标准正态分布
    # retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()[:,:args.num_classes].to(args.device)#len = train data
    #print(len(retrieval_targets))
    cnn_losses, hash_losses, quan_losses = AverageMeter(), AverageMeter(), AverageMeter()
    feat_losses = AverageMeter()
    start = time.time()
    f_rat = 0.5
    best_mAP = 0
    corresponding_mAP_all = 0
    corresponding_zs_mAP = 0
    corresponding_zs_mAP_all = 0
    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(train_loader, args.num_samples, args.batch_size, args.root, args.dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)#len = num samples
        if args.dataset == 'imagenetzs':
            S = (train_targets @ B_tragets.t() > 0).float()
        else:
            S = (train_targets @ retrieval_targets.t() > 0).float()

        # S = (train_targets @ retrieval_targets.t() > 0).float() #num samples * train num
        # print(S[:1])
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        # print(r)
        S = S * (1 + r) - r
        # print(S[:1])
        # Training CNN model
        for epoch in range(args.max_epoch):
            cnn_losses.reset()
            hash_losses.reset()
            quan_losses.reset()
            feat_losses.reset()
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader),ncols=50)
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:
                data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                optimizer.zero_grad()
                F, back_feat, feat_group, attn_group = model(data, is_train=True)
                U[index, :] = F.data
                hash_loss, quan_loss = criterion(F, B, S[index, :], sample_index[index])
                feat_loss = feat_criter(back_feat, feat_group, attn_group, args.device)
                total_loss = hash_loss + quan_loss + feat_loss * f_rat

                cnn_losses.update(total_loss.item())
                feat_losses.update(feat_loss.item())
                hash_losses.update(hash_loss.item())
                quan_losses.update(quan_loss.item())

                total_loss.backward()
                optimizer.step()

                # print(optimizer.param_groups[0]['lr'])
            logger.info('[epoch:{}/{}][cnn_loss:{:.6f}][hash_loss:{:.6f}][quan_loss:{:.6f}][feat_loss:{:.6f}]'.format(epoch+1, args.max_epoch,
                        cnn_losses.avg, hash_losses.avg, quan_losses.avg,feat_losses.avg * f_rat))
        scheduler.step()
        # Update B
        expand_U = torch.zeros(B.shape).to(args.device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, args.gamma)

        logger.info('[iter:{}/{}][iter_time:{:.2f}]'.format(it+1, args.max_iter, time.time()-iter_start))

        if (it < text_step and (it + 1) % args.val_freq == 0) or (it >= text_step and (it + 1) % 1 == 0):
        # if (it + 1) % 1 == 0 :

            query_code = generate_code(model, query_dataloader, code_length, args.device)
            query_targets = query_dataloader.dataset.get_onehot_targets()
            print(query_targets.shape)
            if args.dataset == 'imagenetzs':
                db_code = generate_code(model, retrieval_dataloader, code_length, args.device)
            else:
                db_code = B

            db_label= retrieval_dataloader.dataset.get_onehot_targets()

            print(db_label.shape)
            print(db_label[:,:args.num_classes].shape)

            zs_test_binary = generate_code(model, query_loader_zs, code_length, args.device)
            zs_test_label = query_loader_zs.dataset.get_onehot_targets()

            zs_db_binary = generate_code(model, database_loader_zs, code_length, args.device)
            zs_db_label = database_loader_zs.dataset.get_onehot_targets()

            db_all_binary = torch.cat((db_code, zs_db_binary), 0)
            db_all_label = torch.cat((db_label, zs_db_label), 0)

            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                db_code,
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
                torch.save(db_code.cpu(), os.path.join(ret_path, 'database_code.t'))
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


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
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
        for batch, (data, targets, index) in enumerate(dataloader):
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            # hash_code, _ = model(data)
            hash_code, back_feat, feat_group, attn_group = model(data, is_train=False)
            code[index, :] = hash_code.sign()
    model.train()
    return code
