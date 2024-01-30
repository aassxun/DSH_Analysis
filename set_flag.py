import argparse

from datetime import datetime



def load_args(args ,flag):
    # Get current date
    current_date = datetime.now()

    # Convert to string
    date_string = current_date.strftime('%m-%d')
    # args = cifar10(args, num_zs=2, num_classes=8)

    # %5
    # args = cifar100(args, num_zs=5, num_classes=95)
    # args = food101(args, num_zs=5, num_classes=96)
    # args = cub200(args, num_zs=10, num_classes=190)
    # args = vegfru(args, num_zs=15, num_classes=277)
    # args = standforddog(args, num_zs=6, num_classes=114)
    # args = aircraft(args, num_zs=5, num_classes=95)
    # args = nabirds(args, num_zs=27, num_classes=528)
    # args = nuswide(args, num_zs=11, num_classes=10)
    # args = imagenetzs(args,num_zs=5,num_classes=95)

    # %15
    args = cifar100(args, num_zs=15, num_classes=85)
    # args = food101(args, num_zs=15, num_classes=86)
    # args = cub200(args, num_zs=30, num_classes=170)
    # args = vegfru(args, num_zs=45, num_classes=247)
    # args = standforddog(args, num_zs=18, num_classes=102)
    # args = aircraft(args, num_zs=15, num_classes=85)
    # args = nabirds(args, num_zs=83, num_classes=472)
    # args = imagenetzs(args,num_zs=15,num_classes=85)

    #25%
    # args = cifar100(args, num_zs=25, num_classes=75)
    # args = food101(args, num_zs=25, num_classes=76)
    # args = cub200(args, num_zs=50, num_classes=150)
    # args = vegfru(args, num_zs=75, num_classes=217)
    # args = standforddog(args, num_zs=30, num_classes=90)
    # args = aircraft(args, num_zs=25, num_classes=75)
    # args = nabirds(args, num_zs=138, num_classes=417)
    # args = imagenetzs(args,num_zs=25,num_classes=100)

    args.optim = "SGD"

    args.num_workers = 2
    #------------
    if flag == "adsh":#baseline
        args = adsh(args)
    elif flag == "exchnet":#exchnet
        args = exchnet(args)
    elif flag == "csq":#csq
        args = csq(args)
    elif flag == "hashnet":# hasnet
        args = hashnet(args)
    elif flag == "a2net":
        args = a2net(args)
    elif flag == "semicon":
        args= semicon(args)
    elif flag == "fish":
        args = fish(args)
    elif flag == "ortho":
        args = ortho(args)
    elif flag == "center":
        args = center(args)
    elif flag == "psldh":
        args = PSLDH(args)
    elif flag == "hyp2":
        args = hyp2(args)
    elif flag == "dsh":
        args = DSH(args)
    else:
        print("wrong flag")
        exit(0)

    # args.max_iter = 2
    # args.code_length = '32'
    # args.max_epoch = 1
    args.info = "{}-{}-{}-{}-{}-{}".format(args.dataset,args.num_zs,args.num_classes,args.arch,date_string,args.code_length)
    args.info = 'test'

    return args

















def cub200(args,num_zs=10,num_classes=190):
    args.root = "/home/xsl/datasets/CUB_200_2011"
    # args.root = "/root/autodl-tmp/CUB_200_2011"
    args.pretrain = True
    args.num_samples = 2000
    args.dataset = 'cub-2011-for-zero-shot'
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.max_epoch = 30
    args.max_iter = 40
    args.val_freq = 1
    args.lr_step ='35'
    args.code_length = '12,24,32,48'#12,24,,48
    return args

def food101(args,num_zs=5,num_classes=96):
    args.root = "/home/xsl/datasets/food-101"
    # args.root = "/root/autodl-tmp/food-101"
    args.pretrain = True
    args.num_samples = 2000
    args.dataset = 'food101zs'
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.max_epoch = 30
    args.max_iter = 40
    args.val_freq = 1
    args.lr_step ='35'
    args.code_length = '12,24,32,48'#12,24,,4824,32,
    return args

def vegfru(args,num_zs=15,num_classes=277):
    args.root = "/2T/dataset/vegfru"
    # args.root = "/root/autodl-tmp/vegfru"
    args.pretrain = True
    args.num_samples = 4000
    args.dataset = 'vegfruzs'
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.max_epoch = 30
    args.max_iter = 40
    args.lr_step ='35'
    args.code_length = '12,24,32,48'#
    return args

def standforddog(args,num_zs=6,num_classes=114):
    args.root = "/2T/dataset/StanfordDog"
    args.pretrain = True
    args.num_samples = 2000
    args.dataset = 'stanforddogzs'
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.max_epoch = 30
    args.max_iter = 40
    args.lr_step ='35'
    args.code_length = '12,24,32,48'#
    return args

def nuswide(args,num_zs=11,num_classes=10):
    args.root = "/2T/dataset/NUS-WIDE"
    args.dataset = 'nus-widezs'
    args.num_samples = 2000
    args.topk = 5000
    args.num_classes = num_classes
    args.num_zs = num_zs
    print(f'dataset{args.dataset} divided into {args.num_classes} classes with {args.num_zs} zero-shot classes')
    args.pretrain = True
    args.max_epoch = 3
    args.max_iter = 50
    args.lr_step ='45'
    args.code_length = '16,32,64'
    return args

def cifar100(args,num_zs=5,num_classes=95):

    args.root = "/home/xsl/datasets/cifar-100-python"
    args.dataset = 'cifar100zs'
    args.num_samples = 2000
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.topk = 1000
    print(f'dataset{args.dataset} divided into {args.num_classes} classes with {args.num_zs} zero-shot classes')
    args.pretrain = True
    args.max_epoch = 3
    args.max_iter = 50
    args.val_freq = 1
    args.code_length = '16,32,64'
    args.lr_step ='45'
    return args

def imagenetzs(args,num_zs=5,num_classes=90):
    args.root = "/4T/ImageNet/ILSVRC"
    args.dataset = 'imagenetzs'
    args.num_samples = 13000//100*num_classes
    args.topk = 1000
    args.num_classes = num_classes
    args.num_zs = num_zs
    print(f'dataset{args.dataset} divided into {args.num_classes} classes with {args.num_zs} zero-shot classes')
    args.pretrain = True
    args.max_epoch = 1
    args.max_iter = 50
    args.lr_step ='45'
    args.code_length = '16,32,64'
    return args
#
def cifar10(args,num_zs=2,num_classes=8):

    args.root = "/home/xsl/datasets/cifar-10-python"

    args.dataset = 'cifar10zs'
    args.num_samples = 2000
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.topk = 1000
    print(f'dataset{args.dataset} divided into {args.num_classes} classes with {args.num_zs} zero-shot classes')
    args.pretrain = True
    args.max_epoch = 3
    args.max_iter = 50
    args.val_freq = 1
    args.code_length = '16,32,64'
    args.lr_step ='45'
    return args


#---------------------------------------------------------------------------



def adsh(args):
    args.momen = 0.9
    args.batch_size = 16
    args.lr = 1e-4
    args.wd = 1e-5
    args.gpu = 0
    args.arch = "baseline"
    args.net = "resnet50"
    return args
def exchnet(args):
    args.batch_size = 16
    args.lr = 1e-4
    args.wd = 1e-4
    args.momen = 0.9
    args.gpu = 0
    if args.max_iter == 40:
        args.align_step = 35
    else:
        args.align_step = 40
    args.optim = "SGD"
    args.arch = "exchnet"
    args.net = "resnet50"
    return args
def csq(args):
    args.batch_size = 16
    args.lr = 1e-3
    args.wd = 1e-4
    args.momen = 0.9
    args.arch = "csq"
    args.net = "ResNet"
    args.gpu = 0
    return args
def hashnet(args):
    args.batch_size = 16
    args.lr = 3e-4
    args.wd = 5e-4
    args.momen = 0.9
    args.arch = "hashnet"
    args.step_continuation = 15  ###
    args.net = "ResNet"
    args.gpu = 0
    return args
def a2net(args):
    args.batch_size = 16
    args.lr = 1e-4
    args.wd = 1e-4
    args.momen = 0.9
    args.optim = "SGD"
    args.arch = "a2net"
    args.net = "ResNet"
    args.gpu = 0
    return args
def semicon(args):
    args.batch_size = 16
    args.lr = 2.5e-4
    args.wd = 1e-4
    args.momen = 0.91
    # args.code_length = '16,32,64'  # 12,,32

    args.optim = "SGD"
    args.arch = "semicon"
    args.net = "ResNet"
    args.gpu = 0
    return args

def hyp2(args):

    args.batch_size = 16
    args.lr = 1e-2
    args.criterion_rate = 1e-3
    args.wd = 5e-4
    args.momen = 0.9
    # args.code_length = '16,32,64'  # 16,32,64

    args.optim = "SGD"
    args.arch = "hyp2"
    args.net = "ResNet50"
    args.gpu = 0
    return args


def PSLDH(args):

    args.batch_size = 16
    args.lr = 1e-3
    args.wd = 1e-4
    args.momen = 0.9
    # args.code_length = '12,48'  # 64

    args.optim = "SGD"
    args.arch = "psldh"
    args.net = "ResNet50"
    args.gpu = 0
    return args

def center(args):

    args.batch_size = 16
    args.lr = 1e-5
    args.wd = 1e-5
    args.momen = 0.9
    # args.code_length = '16,32,64'  #

    args.optim = 'RMSprop'
    args.arch = "center"
    if args.max_epoch ==3:
        args.epoch_change= 1
    else:
        args.epoch_change = 11
    args.net = "ResNet50"
    args.gpu = 0
    return args


def ortho(args):

    args.batch_size = 16
    args.lr = 1e-4
    args.wd = 5e-4
    args.momen = 0.9
    # args.code_length = '48'  # ,3212,24,32,
    args.optim = "SGD"
    args.arch = "ortho"

    args.net = "ResNet50"
    args.gpu = 0
    return args

def fish(args):

    args.batch_size = 16
    args.lr = 1e-2
    # args.wd = 5e-4
    args.momen = 0.9
    # args.code_length = '12,'  # ,3212,24,32,

    args.optim = "SGD"
    args.arch = "fish"
    args.lr_step ='5,15,35'
    args.net = "ResNet50"
    args.gpu = 0
    return args

def RCDH(args):

    args.batch_size = 16
    args.lr = 1e-4
    args.wd = 5e-4
    args.momen = 0.9
    # args.code_length = '16,32,64'  # ,3212,24,32,

    args.optim = "SGD"
    args.arch = "rcdh"
    args.net = "ResNet50"
    args.gpu = 0
    return args

def DSH(args):

    args.batch_size = 16
    args.lr = 0.001
    args.wd = 1e-4
    args.momen = 0.9


    args.optim = "SGD"
    args.arch = "dsh"

    args.net = "ResNet50"

    return args


