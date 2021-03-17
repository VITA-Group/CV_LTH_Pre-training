'''
load lottery tickets and evaluation 
support datasets: cifar10, Fashionmnist, cifar100
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
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from pruning_utils import *

parser = argparse.ArgumentParser(description='PyTorch Evaluation Tickets')

##################################### data setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset[cifar10&100, svhn, fmnist')

##################################### model setting #################################################
parser.add_argument('--arch', type=str, default='resnet50', help='model architecture[resnet18, resnet50, resnet152]')

##################################### basic setting #################################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save_model', action="store_true", help="whether saving model")
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=1, type=int, help='warm up epochs')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight of Ticket')
parser.add_argument('--dict_key', default=None, type=str, help='key of pretrained file')
parser.add_argument('--mask_dir', default=None, type=str, help='mask direction of Ticket')
parser.add_argument('--conv1', action="store_true", help="whether prune conv1")
parser.add_argument('--load_all', action="store_true", help="whether loading all weight in pretrained model")
parser.add_argument('--reverse_mask', action="store_true", help="whether using reverse mask")

def main():

    best_sa = 0
    args = parser.parse_args()
    print(args)

    print('*'*50)
    print('Dataset: {}'.format(args.dataset))
    print('Model: {}'.format(args.arch))
    print('*'*50)     

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset 
    model, train_loader, val_loader, test_loader = setup_model_dataset(args)
    model.cuda()

    #loading tickets
    load_ticket(model, args)

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    all_result = {}
    all_result['train'] = []
    all_result['test_ta'] = []
    all_result['ta'] = []

    start_epoch = 0
    remain_weight = check_sparsity(model, conv1=args.conv1)

    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        tacc = test(val_loader, model, criterion, args)
        # evaluate on test set
        test_tacc = test(test_loader, model, criterion, args)

        scheduler.step()

        all_result['train'].append(acc)
        all_result['ta'].append(tacc)
        all_result['test_ta'].append(test_tacc)
        all_result['remain_weight'] = remain_weight

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc  > best_sa
        best_sa = max(tacc, best_sa)

        if args.save_model:

            save_checkpoint({
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_SA_best=is_best_sa, save_path=args.save_dir)

        else:
            save_checkpoint({
                'result': all_result
            }, is_SA_best=False, save_path=args.save_dir)

        plt.plot(all_result['train'], label='train_acc')
        plt.plot(all_result['ta'], label='val_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    check_sparsity(model, conv1=args.conv1)
    print('* best SA={}'.format(all_result['test_ta'][np.argmax(np.array(all_result['ta']))]))


if __name__ == '__main__':
    main()


