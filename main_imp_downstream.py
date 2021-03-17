'''
iterative pruning for supervised task 
with lottery tickets or pretrain tickets 
support datasets: cifar10, Fashionmnist, cifar100, svhn
'''

import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import *
from pruning_utils import *

parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### data setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset[cifar10&100, svhn, fmnist')

##################################### model setting #################################################
parser.add_argument('--arch', type=str, default='resnet50', help='model architecture[resnet18, resnet50, resnet152]')

##################################### basic setting #################################################
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=1, type=int, help='warm up epochs')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=19, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt, rewind_lt or pt_trans)')
parser.add_argument('--random_prune', action="store_true", help="whether using random pruning")
parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
parser.add_argument('--conv1', action="store_true", help="whether pruning&loading conv1")
parser.add_argument('--fc', action="store_true", help="whether loading fc")
parser.add_argument('--rewind_epoch', default=9, type=int, help='rewind checkpoint')

def main():
    best_sa = 0
    args = parser.parse_args()
    print(args)

    print('*'*50)
    print('Dataset: {}'.format(args.dataset))
    print('Model: {}'.format(args.arch))
    print('*'*50)     
    print('Pruning type: {}'.format(args.prune_type))   
    if args.random_prune:
        print('Random Unstructure Pruning')
    else:
        print('L1 Unstructure Pruning')

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset 
    model, train_loader, val_loader, test_loader = setup_model_dataset(args)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    if args.prune_type == 'lt':
        print('lottery tickets setting (rewind to random init')
        initalization = deepcopy(model.state_dict())

    elif args.prune_type == 'pt_trans':
        print('pretrain tickets with {}'.format(args.pretrained))
        pretrained_weight = torch.load(args.pretrained, map_location = torch.device('cuda:'+str(args.gpu)))
        if 'state_dict' in pretrained_weight.keys():
            pretrained_weight = pretrained_weight['state_dict']

        load_weight_pt_trans(model, pretrained_weight, args)
        initalization = deepcopy(model.state_dict())

    elif args.prune_type == 'pt':
        initalization = None
    elif args.prune_type == 'rewind_lt':
        initalization = None
    else:
        raise ValueError('Unknow pruning type')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    if args.resume:
        print('resume from checkpoint')
        checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
        best_sa = checkpoint['best_sa']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']
        start_state = checkpoint['state']

        if start_state > 0:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(model, current_mask, conv1=args.conv1)
            check_sparsity(model, conv1=args.conv1)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initalization = checkpoint['init_weight']
        print('loading state:', start_state)
        print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)

    else:
        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        start_epoch = 0
        start_state = 0

    print('######################################## Start Standard Training Iterative Pruning ########################################')
    
    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        
        check_sparsity(model, conv1=args.conv1)        
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])

            acc = train(train_loader, model, criterion, optimizer, epoch, args)

            if state == 0:
                if epoch == args.rewind_epoch-1:
                    if args.prune_type == 'rewind_lt':
                        torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1)))
                        initalization = deepcopy(model.state_dict())

            # evaluate on validation set
            tacc = test(val_loader, model, criterion, args)
            # evaluate on test set
            test_tacc = test(test_loader, model, criterion, args)

            scheduler.step()

            all_result['train'].append(acc)
            all_result['ta'].append(tacc)
            all_result['test_ta'].append(test_tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc  > best_sa
            best_sa = max(tacc, best_sa)

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': initalization
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
        
            plt.plot(all_result['train'], label='train_acc')
            plt.plot(all_result['ta'], label='val_acc')
            plt.plot(all_result['test_ta'], label='test_acc')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            plt.close()

        #report result
        check_sparsity(model, conv1=args.conv1)
        print('* best SA={}'.format(all_result['test_ta'][np.argmax(np.array(all_result['ta']))]))

        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        best_sa = 0
        start_epoch = 0

        if args.prune_type == 'pt':
            print('* loading pretrained weight')
            initalization = torch.load(os.path.join(args.save_dir, '0model_SA_best.pth.tar'), map_location = torch.device('cuda:'+str(args.gpu)))['state_dict']

        if args.random_prune:
            pruning_model_random(model, args.rate, conv1=args.conv1)
        else:
            pruning_model(model, args.rate, conv1=args.conv1)

        current_mask = extract_mask(model.state_dict())

        remove_prune(model, conv1=args.conv1)
        #rewind weight to init
        model.load_state_dict(initalization)
        prune_model_custom(model, current_mask, conv1=args.conv1)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

if __name__ == '__main__':
    main()


