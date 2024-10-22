import argparse

from datetime import datetime



def load_args(args ,flag):
    # Get current date
    current_date = datetime.now()

    # Convert to string
    date_string = current_date.strftime('%m-%d')
    # args = cifar10(args, num_zs=2, num_classes=8)

    if args.ZS == 5:
        if args.data == 'cifar100':
            args = cifar100(args, num_zs=5, num_classes=95)
        elif args.data == 'food101':
            args = food101(args, num_zs=5, num_classes=96)
        elif args.data == 'cub200':
            args = cub200(args, num_zs=10, num_classes=190)
        elif args.data == 'vegfru':
            args = vegfru(args, num_zs=15, num_classes=277)
        elif args.data == 'standforddog':
            args = standforddog(args, num_zs=6, num_classes=114)
        elif args.data == 'aircraft':
            args = aircraft(args, num_zs=5, num_classes=95)
        elif args.data == 'nabirds':
            args = nabirds(args, num_zs=27, num_classes=528)
        elif args.data == 'nuswide':
            args = nuswide(args, num_zs=11, num_classes=10)
        elif args.data == 'coco':
            args = coco(args, num_zs=35, num_classes=45)
        elif args.data == 'flickr25k':
            args = flickr25k(args, num_zs=6, num_classes=32)
        elif args.data == 'imagenetzs':
            args = imagenetzs(args, num_zs=5, num_classes=95)

    elif args.ZS == 15:
        if args.data == 'cifar100':
            args = cifar100(args, num_zs=15, num_classes=85)
        elif args.data == 'food101':
            args = food101(args, num_zs=15, num_classes=86)
        elif args.data == 'cub200':
            args = cub200(args, num_zs=30, num_classes=170)
        elif args.data == 'vegfru':
            args = vegfru(args, num_zs=45, num_classes=247)
        elif args.data == 'standforddog':
            args = standforddog(args, num_zs=18, num_classes=102)
        elif args.data == 'aircraft':
            args = aircraft(args, num_zs=15, num_classes=85)
        elif args.data == 'nabirds':
            args = nabirds(args, num_zs=83, num_classes=472)
        elif args.data == 'nuswide':
            args = nuswide(args, num_zs=16, num_classes=5)
        elif args.data == 'coco':
            args = coco(args, num_zs=53, num_classes=27)
        elif args.data == 'flickr25k':
            args = flickr25k(args, num_zs=15, num_classes=23)
        elif args.data == 'imagenetzs':
            args = imagenetzs(args, num_zs=15, num_classes=85)

    elif args.ZS == 25:
        if args.data == 'cifar100':
            args = cifar100(args, num_zs=25, num_classes=75)
        elif args.data == 'food101':
            args = food101(args, num_zs=25, num_classes=76)
        elif args.data == 'cub200':
            args = cub200(args, num_zs=50, num_classes=150)
        elif args.data == 'vegfru':
            args = vegfru(args, num_zs=75, num_classes=217)
        elif args.data == 'standforddog':
            args = standforddog(args, num_zs=30, num_classes=90)
        elif args.data == 'aircraft':
            args = aircraft(args, num_zs=25, num_classes=75)
        elif args.data == 'nabirds':
            args = nabirds(args, num_zs=138, num_classes=417)
        elif args.data == 'imagenetzs':
            args = imagenetzs(args, num_zs=25, num_classes=75)




    print(f'dataset {args.dataset} divided into {args.num_classes} classes with {args.num_zs} zero-shot classes')
    args.optim = "SGD"
    # args.num_samples = 2000
    args.num_workers = 2
    #------------
    if flag == "adsh":#baseline
        args.momen = 0.9
        args.batch_size = 16
        args.lr = 1e-4    
        args.wd = 1e-4 
        args.gpu = 2
        args.arch = "baseline"
        args.net = "resnet50"

        #------------


    elif flag == "exchnet":#exchnet
        args.batch_size = 16
        args.lr = 1e-4            
        args.wd = 1e-4                 
        args.momen = 0.9
        if args.max_iter == 40:
            args.align_step = 35
        else:
            args.align_step = 40

        args.optim = "SGD"
        args.arch = "exchnet"
        args.net = "resnet50"




    elif flag == "csq":#csq
        args.batch_size = 16
        args.lr = 1e-3
        args.wd = 1e-4              
        args.momen = 0.9
        args.arch = "csq"      
        args.net = "ResNet"


    elif flag == "hashnet":# hasnet
        args.batch_size = 16
        args.lr = 3e-4            
        args.wd = 5e-4              
        args.momen = 0.9
        args.arch = "hashnet"
        args.step_continuation = 15  ###
        args.net = "ResNet"

    elif flag == "a2net":

        args.batch_size = 16
        args.lr = 1e-4
        args.wd = 1e-4
        args.momen = 0.9
        args.optim = "SGD"
        args.arch = "a2net"
        args.net = "ResNet"

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
    elif flag == "dpah":
        args = DPAH(args)
    elif flag == "dcdh":
        args = DCDH(args)
    elif flag == "agmh":
        args = AGMH(args)
    elif flag == "chn":
        args = CHN(args)
    elif flag == "dah":
        args = DAH(args)
    else:
        print("wrong flag")
        exit(0)

    args.info = "{}-{}-{}-{}-{}-{}-wocls".format(args.dataset,args.num_zs,args.num_classes,args.arch,date_string,args.code_length)
    print('info is '+args.info)
    return args

















def cub200(args,num_zs=10,num_classes=190):
    args.root = "/4T/dataset/CUB_200_2011"

    args.pretrain = True
    args.num_samples = 2000
    args.dataset = 'cub-2011-for-zero-shot'
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.max_epoch = 30
    args.max_iter = 40
    args.val_freq = 5
    args.lr_step ='35'
    args.code_length = '12,24,32,48'#12,24,,48
    return args

def food101(args,num_zs=5,num_classes=96):
    args.root = "/4T/dataset/food101"
    # args.root = "/root/autodl-tmp/food-101"
    args.pretrain = True
    args.num_samples = 2000
    args.dataset = 'food101zs'
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.max_epoch = 30
    args.max_iter = 40
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
    args.root = "/4T/dataset/StanfordDog"
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

def aircraft(args,num_zs=5,num_classes=95):
    args.root = "/2T/dataset/aircraft"
    # args.root = "/root/autodl-tmp/aircraft"
    args.pretrain = True
    args.num_samples = 2000
    args.dataset = 'aircraftzs'
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.max_epoch = 30
    args.max_iter = 40
    args.lr_step ='35'
    args.code_length = '12,24,32,48'#12,24,,4824,32,
    return args

def nabirds(args,num_zs=27,num_classes=528):
    args.root = "/2T/dataset/nabirds"
    # args.root = "/root/autodl-tmp/nabirds"
    args.pretrain = True
    args.num_samples = 4000
    args.dataset = 'nabirdszs'
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.max_epoch = 30
    args.max_iter = 40
    args.lr_step ='35'
    args.code_length = '12,24,32,48'#12,24,,4824,32,
    return args

def nuswide(args,num_zs=11,num_classes=10):
    args.root = "/4T/dataset/NUS-WIDE"
    args.dataset = 'nus-widezs'
    args.num_samples = 2000
    args.topk = 5000
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.pretrain = True
    args.max_epoch = 3
    args.max_iter = 50
    args.lr_step ='45'
    args.val_freq = 1
    args.code_length = '16,32,64'
    return args
def coco(args,num_zs=35,num_classes=45):
    args.root = "/4T/dataset/coco"
    args.dataset = 'cocozs'
    args.num_samples = 2000
    args.topk = 5000
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.pretrain = True
    args.max_epoch = 3
    args.max_iter = 50
    args.lr_step ='45'
    args.val_freq = 5
    args.code_length = '16,32,64'
    return args
def flickr25k(args,num_zs=35,num_classes=45):
    args.root = "/4T/dataset/flickr25k"
    args.dataset = 'flickr25kzs'
    args.num_samples = 2000
    args.topk = 5000
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.pretrain = True
    args.max_epoch = 3
    args.max_iter = 50
    args.lr_step ='45'
    args.val_freq = 5
    args.code_length = '16,32,64'
    return args

def cifar100(args,num_zs=5,num_classes=95):

    args.root = "/2T/dataset/cifar-100-python"
    args.dataset = 'cifar100zs'
    args.num_samples = 2000
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.topk = 1000
    args.pretrain = True
    args.max_epoch = 3
    args.max_iter = 50
    args.val_freq = 1
    args.code_length = '16,32,64'
    args.lr_step ='45'
    return args

def imagenetzs(args,num_zs=5,num_classes=90):
    args.root = "/4T/ImageNet/ILSVRC"
    # args.root = "/2T/dataset/ImageNet/ILSVRC"
    args.dataset = 'imagenetzs'
    args.num_samples = 13000//100*num_classes
    args.topk = 1000
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.pretrain = True
    args.max_epoch = 1
    args.max_iter = 50
    args.val_freq = 5
    args.lr_step ='45'
    args.code_length = '16,32,64'
    return args
#
def cifar10(args,num_zs=2,num_classes=8):

    args.root = "/2T/dataset/cifar-10-python"
    args.dataset = 'cifar10zs'
    args.num_samples = 2000
    args.num_classes = num_classes
    args.num_zs = num_zs
    args.topk = 1000
    args.pretrain = True
    args.max_epoch = 3
    args.max_iter = 50
    args.val_freq = 1
    args.code_length = '16,32,64'
    args.lr_step ='45'
    return args


def semicon(args):
    args.batch_size = 16
    args.lr = 5e-4
    args.wd = 1e-4
    args.momen = 0.91
    args.optim = "SGD"
    args.arch = "semicon"
    args.net = "ResNet"
    return args

def DPAH(args):
    args.batch_size = 16
    args.lr = 1e-3
    args.wd = 1e-4
    args.momen = 0.90
    args.optim = "SGD"
    args.arch = "dpah"
    args.net = "ResNet"
    return args
def DCDH(args):
    args.batch_size = 16
    args.lr = 5e-4
    args.wd = 5e-4
    args.momen = 0.90
    args.optim = "SGD"
    args.arch = "dcdh"
    args.net = "ResNet"
    return args
def AGMH(args):
    args.batch_size = 16
    args.lr = 1e-3
    args.wd = 1e-4
    args.momen = 0.90
    args.optim = "SGD"
    args.arch = "agmh"
    args.net = "ResNet"
    return args
def CHN(args):
    args.batch_size = 16
    args.lr = 1e-3
    args.wd = 1e-4
    args.momen = 0.90
    args.optim = "SGD"
    args.arch = "chn"
    args.net = "ResNet"
    return args
def DAH(args):
    args.batch_size = 16
    if args.data == 'vegfru' or args.data == 'nabirds':
        args.lr = 5e-4
    else:
        args.lr = 2.5e-4
    args.wd = 1e-4
    args.momen = 0.91
    args.optim = "SGD"
    args.arch = "dah"
    args.net = "ResNet"
    args.gpu = 1
    return args
def hyp2(args):

    args.batch_size = 16
    args.lr = 1e-2
    args.criterion_rate = 1e-3
    args.wd = 5e-4
    args.momen = 0.9
    args.optim = "SGD"
    args.arch = "hyp2"
    args.net = "ResNet50"
    return args


def PSLDH(args):

    args.batch_size = 16
    args.lr = 1e-3
    args.wd = 1e-4
    args.momen = 0.9
    args.optim = "SGD"
    args.arch = "psldh"
    args.net = "ResNet50"
    return args

def center(args):

    args.batch_size = 16
    args.lr = 1e-5
    args.wd = 1e-5
    args.momen = 0.9
    args.optim = 'RMSprop'
    args.arch = "center"
    if args.max_epoch ==3:
        args.epoch_change= 1
    else:
        args.epoch_change = 11
    args.net = "ResNet50"
    return args


def ortho(args):

    args.batch_size = 16
    args.lr = 1e-4
    args.wd = 5e-4
    args.momen = 0.9
    args.optim = "SGD"
    args.arch = "ortho"
    args.net = "ResNet50"
    return args

def fish(args):

    args.batch_size = 16
    args.lr = 1e-2
    args.momen = 0.9

    args.optim = "SGD"
    args.arch = "fish"
    args.lr_step ='5,15,35'
    args.net = "ResNet50"
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

