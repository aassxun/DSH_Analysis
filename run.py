import torch
import argparse
import adsh
import adsh_exchnet
import numpy as np
import random
from loguru import logger
from data.data_loader import load_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  #,4,5,6,7
import hashnet
import csq
import A_2_Net
import SEMICON_train as SEMICON_Net
from set_flag import load_args
import fish
import ortho
import center
import psldh
import hyp2
import DSH as dsh
import time
from tqdm import tqdm
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run():
    seed_everything(42)
    args = load_config()
    log_path = 'logs/' + args.arch + '-' + args.net + '/' + args.info
    if not os.path.exists(log_path):
        os.makedirs(log_path)



    #------------------------------------------------------------
    #------------------------------------------------------------
    logger.add(log_path +"/"+ '-{time:YYYY-MM-DD}.log', rotation='500 MB', level='INFO')
    logger.success(args)
    #time:YYYY-MM-DD HH
    # Load dataset
    query_dataloader, train_dataloader, retrieval_dataloader,query4zero_shot_dataloader, database4zero_shot_dataloader = load_data(
            args.dataset,
            args.root,
            args.num_query,
            args.num_samples,
            args.batch_size,
            args.num_workers,
            args.num_zs,
        )
    for i in tqdm(range(1),ncols=100,desc=f"{args.arch} is sleeping"):
        time.sleep(1)

    if args.arch == 'baseline':
        net_arch = adsh
        ### 用resnet50（分类头数量等于比特数） 直接生成hashcode
    elif args.arch == 'exchnet':
        net_arch = adsh_exchnet
    elif args.arch == 'hashnet':
        net_arch = hashnet
    elif args.arch == 'csq':
        net_arch = csq
    elif args.arch == 'a2net':
        net_arch = A_2_Net
    elif args.arch == 'semicon':
        net_arch = SEMICON_Net
    elif args.arch == 'fish':
        net_arch = fish
    elif args.arch == 'ortho':
        net_arch = ortho
    elif args.arch == 'psldh':
        net_arch = psldh
    elif args.arch == 'hyp2':
        net_arch = hyp2
    elif args.arch == 'dsh':
        net_arch = dsh
    if args.arch == 'center':
        net_arch = center
    for code_length in args.code_length:

        mAP,mAP_all,zs_mAP,zs_mAP_all = net_arch.train(query_dataloader, train_dataloader, retrieval_dataloader,
                            query4zero_shot_dataloader, database4zero_shot_dataloader,
                            code_length, args)

        logger.info('[code_length:{}][map:{:.4f}][map_all:{:.4f}][zs_map:{:.4f}][zs_map_all:{:.4f}]'.format(code_length, mAP,mAP_all,zs_mAP,zs_mAP_all))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ADSH_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--wd', default=1e-5, type=float,
                        help='Weight Decay.(default: 1e-5)')
    parser.add_argument('--optim', default='Adam', type=str,
                        help='Optimizer')
    parser.add_argument('--code-length', default='12,24,32,48', type=str,#
                        help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--arch', default='baseline',
                        help='Net arch')
    parser.add_argument('--net', default='AlexNet',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--lr-step', default='30,45', type=str,
                        help='lr decrease step.(default: 30,45)')
    parser.add_argument('--align-step', default=50, type=int,
                        help='Step of start aligning.(default: 50)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Using image net pretrain')
    parser.add_argument('--quan-loss', action='store_true',
                        help='Using quan_loss')
    parser.add_argument('--lambd-sp', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd-ch', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--momen', default=0.9, type=float,
                        help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Using SGD nesterov')
    #parser.add_argument('--cfg', default='experiments/cls_hrnet_w44_sgd_lr5e-2_wd1e-4_bs32_x100.yaml' , type=str,
    #                    help='HRNet config')
    parser.add_argument('--num-classes', default=200, type=int,
                        help='Number of classes.(default: 200)')
    parser.add_argument('--val-freq', default=5, type=int,
                        help='Number of validate frequency.(default: 10)')
    parser.add_argument('--cauchy-gamma', default=20.0, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--margin', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--num-zs', default=50, type=int,
                        help='Number of zero-shot classes(default: 50)')
    parser.add_argument('--epoch-change', default=11, type=int,
                        help='Using for center hash')
    # parser.add_argument('--pk', default=80, type=int,
    #                     help='Number of epochs.(default: 3)')
    args = parser.parse_args()
#-------------------------------------------------------------------------

    flag = "psldh" #  exchnet  csq  hashnet  a2net    center  hyp2  dsh
                    #adsh  semicon  fish   ortho psldh
    print("train {}".format(flag))
    args = load_args(args,flag)

#---------------------------------------------------------------------------------

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu}")

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))
    args.lr_step = list(map(int, args.lr_step.split(',')))

    return args


if __name__ == '__main__':

    run()
