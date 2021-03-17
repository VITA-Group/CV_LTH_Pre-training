import utils
from utils import AverageMeter
import time
import torch
import torch.nn as nn
import pdb


def train(train_loader, model, optimizer, epoch, args, device):

    print("Current LR [{:.8f}]".format(optimizer.state_dict()['param_groups'][0]['lr']))
    losses = AverageMeter()
    
    t0 = time.time()
    for i, (inputs) in enumerate(train_loader):

        d = inputs.size()
        inputs = inputs.view(d[0] * 2, d[2], d[3], d[4]).to(device)
        features = model.train()(inputs)
        loss = utils.nt_xent(features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(float(loss.detach().cpu()), inputs.shape[0])
        # torch.cuda.empty_cache()
        
        if i % args.print_freq == 0:

            if i == 0:
                tot = time.time() - t0
                t1 = tot
            else:
                t1 = time.time() - t0 - tot
                tot = time.time() - t0

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                  'Epoch: [{}][{}/{}]  '
                  'Loss {:.4f} ({:.4f})  '
                  'Time: [{:.2f} min]  '
                  .format(epoch, i, len(train_loader), losses.val, losses.avg, t1 / 60))

    return losses.avg



def val_cacc(train_loader, model, args, device, mlp=True):

    constrastive_acc1 = AverageMeter()
    constrastive_acc3 = AverageMeter()
    constrastive_acc5 = AverageMeter()

    t0 = time.time()
    
    if mlp == False:
        origin_fc = model.module.fc
        model.module.fc = nn.Identity()

    for i, (inputs) in enumerate(train_loader):
        
        d = inputs.size()
        inputs = inputs.view(d[0] * 2, d[2], d[3], d[4]).to(device)
        with torch.no_grad():
            features = model.eval()(inputs)
        
        con_acc1, con_acc3, con_acc5, = utils.contrastive_acc(features.to('cpu'), t=0.5, topk=(1, 3, 5))

        constrastive_acc1.update(float(con_acc1), inputs.shape[0])
        constrastive_acc3.update(float(con_acc3), inputs.shape[0])
        constrastive_acc5.update(float(con_acc5), inputs.shape[0])

        if i % args.print_freq == 0:
            if i == 0:
                tot = time.time() - t0
                t1 = tot
            else:
                t1 = time.time() - t0 - tot
                tot = time.time() - t0

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                  'Iter: [{}/{}]  '
                  'Constrastive Acc {:.4f} ({:.4f})  '
                  'Time: [{:.2f} min]  '
                  .format(i, len(train_loader), constrastive_acc1.val, constrastive_acc1.avg, t1 / 60))

    if mlp == False:
        model.module.fc = origin_fc

    return constrastive_acc1.avg, constrastive_acc3.avg, constrastive_acc5.avg





def validate(train_loader, val_loader, model, device, args, finetune_bn_stat=False, funetune_bn_param=False):
    """
    Run evaluation
    """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    losses = AverageMeter()
    # train a fc on the representation
    for name, param in model.named_parameters():
        param.requires_grad = False
        # print("{} \t {}".format(name, param.shape))

    if funetune_bn_param:
        for name, param in model.named_parameters():
            if 'bn' in name:
                param.requires_grad = True
    
    previous_fc = model.module.fc
    ch = model.module.fc.fc1.in_features

    model.module.fc = nn.Linear(ch, args.n_classes)
    model.to(device)
    print("-" * 80)
    for name, param in model.named_parameters():
        print("{} \t {}".format(name, param.requires_grad))

    epochs_max = 100
    lr = 0.001

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0, momentum=0.9, nesterov=True)
    
    t0 = time.time()

    for epoch in range(epochs_max):

        print("Current lr is [{:.6f}]".format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, (sample) in enumerate(train_loader):

            x, y = sample[0].to(device), sample[1].to(device)
            if not finetune_bn_stat:
                p = model.eval()(x)
            else:
                p = model.train()(x)
            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            losses.update(float(loss.detach().cpu()), x.shape[0])

            if i % 50 == 0:

                if i == 0:
                    tot = time.time() - t0
                    t1 = tot
                else:
                    t1 = time.time() - t0 - tot
                    tot = time.time() - t0

                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                      'Test epoch: [{}][{}/{}]\t'
                      'Loss {:.4f} ({:.4f})\t'
                      'Iter Time: [{:.2f} min]\t'
                      .format(epoch, i, len(train_loader), losses.val, losses.avg, t1 / 60))
                
        

    utils.save_downstream_model(model, args)
    final_acc = eval_downstream_performance(model, val_loader, device, args)
    print("DownStream Task Top1:[{}]   Top5:[{}]".format(final_acc[0], final_acc[1]))
    # recover every thing
    model.module.fc = previous_fc
    model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    return final_acc




def eval_downstream_performance(model, val_loader, device, args):

    print("-" * 80)
    print("Final EVAL ! ")
    print("-" * 80)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, targets) in enumerate(val_loader):
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():

            outputs = model.eval()(inputs)
            loss = criterion(outputs, targets)

        outputs = outputs.float()
        loss = loss.float()    
    
        acc1, acc5 = utils.accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if i % args.print_freq == 0:
            print('Final Test: [{0}/{1}]  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Acc@1 {top1.val:.4f} ({top1.avg:.4f})  '
                  'Acc@5 {top5.val:.4f} ({top5.avg:.4f})'
                  .format(i, len(val_loader), loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg

