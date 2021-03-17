import os
import sys
import argparse, json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import CustomImageFolder, Projection_head
from optimizer.lars import LARS
import pdb


class Parser(object):
    def __init__(self):
        # model and data
        parser = argparse.ArgumentParser("My ImageNet")
        parser.add_argument('experiment',        type=str)
        parser.add_argument('--data',            type=str,           default='./data',                  help='dataset dir')
        parser.add_argument('--arch',            type=str,           default='resnet50',                help='model')
        parser.add_argument('--sim_model',       type=str,           default='./simclr_pre-train_model/resnet50-1x.pth', help='location of the simclr pre-train model')
        parser.add_argument('--mlpout',          type=int,           default=128,                       help='mlp out dimision')
        parser.add_argument('--n_classes',       type=int,           default=1000,                      help='class number')
        # training options
        parser.add_argument('--epochs',          type=int,           default=150,                       help='epochs')
        parser.add_argument('--batch_size',      type=int,           default=256,                        help='batch_size')
        parser.add_argument('--lr',              type=float,         default=0.0001,                    help='learning rate')
        parser.add_argument('--optimizer',       type=str,           default='lars',                    help='optimizer type')
        parser.add_argument('--momentum',        type=float,         default=0.9,                       help='momentum')
        parser.add_argument('--weight_decay',    type=float,         default=1e-6,                      help='weight decay' )
        parser.add_argument('--gpu',             type=str,           default=0,                         help='gpu')
        parser.add_argument('--seed',            type=int,           default=2,                         help='seed')
        # pruning setting
        parser.add_argument('--prun_percent',    type=float,         default=0.2,                       help='percent of pruning')
        parser.add_argument('--prun_epoch',      type=int,           default=10,                        help='pruning at each prun_epoch epoch')
        # function
        parser.add_argument('--print_freq',      type=int,           default=50,                        help='print freq')
        parser.add_argument('--resume',          type=str,           default='',                        help='path to latest checkpoint')
        parser.add_argument('--save_dir',        type=str,           default='./output',                help='checkpoint save path')
        parser.add_argument('--eval_dir',        type=str,           default='',                        help='checkpoint save path')

        parser.add_argument('--phead',              action='store_false', default=True,                      help='evaluate model on downstream task')
        parser.add_argument('--eval_baseline',      action='store_true',  default=False,                     help='Train and Eval the baseline')
        parser.add_argument('--deval',              action='store_true',  default=False,                     help='evaluate model on downstream task')
        parser.add_argument('--eval',               action='store_true',  default=False,                     help='evaluate model on validation set')
        parser.add_argument('--eval_cacc',          action='store_true',  default=False,                     help='evaluate contrastive acc')
        parser.add_argument('--funetune_bn_stat',   action='store_true',  help='if specified, finetune the statistics of bn during testing')
        parser.add_argument('--funetune_bn_param',  action='store_true',  help='if specified, also finetune the parameter of bn')

        self.args = parser.parse_args()
        

class Configer(Parser):
    def __init__(self):
        super(Configer, self).__init__()
        self.device = None
        
        save_dir = os.path.join(self.args.save_dir, self.args.experiment)
        if os.path.exists(save_dir) is not True:
            print("INFO: Creating Output File")
            os.system("mkdir -p {}".format(save_dir))


    def args_setup(self):
        print('-' * 80)
        print(self.args)
        print('-' * 80)
        return self.args


    def gpu_setup(self):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)  
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print('Cuda not available')
            device = torch.device("cpu")
        print("INFO: GPU ID: [{}]".format(self.args.gpu))
        self.device = device
        return device
    
    
    def seed_setup(self):
        
        print('INFO: Seeds : [{}]'.format(self.args.seed))
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        cudnn.deterministic = True
        torch.cuda.manual_seed(self.args.seed)


    def model_setup(self, project_head=True):

        print('INFO: Creating Model...')
        print("INFO: Model Name: [{}]".format(self.args.arch))
        model = models.__dict__[self.args.arch]()

        if project_head:
            print("INFO: Adding Projection Head...")
            in_dim = model.fc.in_features # 2048
            model.fc = Projection_head(in_dim, mlp_out=self.args.mlpout)
        # model.to(self.device)
        model = torch.nn.DataParallel(model).to(self.device)
        
        cudnn.benchmark = True
        return model

    
    def dataloader_setup(self):

        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.ToTensor(),
        ])

        tfs_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        print('INFO: Load Data...')

        traindir = os.path.join(self.args.data, 'train')
        valdir = os.path.join(self.args.data, 'val')

        train_datasets = CustomImageFolder(root=traindir, transform=tfs_train)
        val_datasets = CustomImageFolder(root=valdir, transform=tfs_train)

        val_train_datasets = datasets.ImageFolder(root=traindir, transform=tfs_test)
        test_datasets = datasets.ImageFolder(root=valdir, transform=tfs_test)

        print('INFO: Finish Load ImageNet!')
        print('INFO: Creating Dataloader...')

        train_loader = DataLoader(
            train_datasets,
            num_workers=4,
            batch_size=self.args.batch_size)

        val_loader = DataLoader(
            val_datasets,
            num_workers=4,
            batch_size=self.args.batch_size) 


        val_train_loader = DataLoader(
            val_train_datasets,
            num_workers=4,
            shuffle=True,
            batch_size=self.args.batch_size)

        test_loader = DataLoader(
            test_datasets,
            num_workers=4,
            shuffle=False,
            batch_size=self.args.batch_size)

        print('INFO: Batch Size: [{}]'.format(self.args.batch_size))
        return train_loader, val_loader, val_train_loader, test_loader
        

    def optimizer_setup(self, model):

        print('INFO: Creating Optimizer: [{}]  LR: [{:.8f}]   Weight Decay: [{:.8f}]'
                .format(self.args.optimizer, self.args.lr, self.args.weight_decay))
     
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'lars':
            optimizer = LARS(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=0.9)
        else:
            print("no defined optimizer")
            assert False
        return optimizer


    def criterion_setup(self):
        print('INFO: Creating Criterion...')
        criterion = nn.CrossEntropyLoss().to(self.device)
        return criterion
    

    def get_simclr_model(self):
        sim_ckpt = torch.load(self.args.sim_model)['state_dict']
        return sim_ckpt
    

    def loading_ckpt(self, model, ckpt):
        
        model_dict = model.module.state_dict()
        ori_model_keys_num = model_dict.keys().__len__()
        overlap_state_dict = {k : v for k , v in ckpt.items() if k in model_dict.keys()}
        overlap_keys_num = overlap_state_dict.keys().__len__()
    
        model_dict.update(overlap_state_dict)
        model.module.load_state_dict(model_dict)

        print("INFO: Load Original SimCLR Pre-trained Model! [{}/{}]"
              .format(overlap_keys_num, ori_model_keys_num))

