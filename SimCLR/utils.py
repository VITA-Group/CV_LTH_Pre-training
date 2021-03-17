import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np



class CustomImageFolder(ImageFolder):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def __getitem__(self, idx):
        
        path, target = self.samples[idx]
        img = self.loader(path)
        imgs = [self.transform(img), self.transform(img)]
        
        return torch.stack(imgs)


class Projection_head(nn.Module):
    def __init__(self, ch, mlp_out=128):
        super(Projection_head, self).__init__()

        self.in_dim = ch
        self.out_dim = mlp_out
        # projection MLP
        self.fc1 = nn.Linear(self.in_dim, ch)
        self.fc2 = nn.Linear(ch, self.out_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def save_checkpoint(epoch, model, optimizer, args, filename):

    state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
            }

    save_dir = os.path.join(args.save_dir, args.experiment)
    file_name=os.path.join(save_dir, filename)
    torch.save(state, file_name)
    print("INFO: Save Model Epoch:[{}]  Save Dir:[{}]"
          .format(epoch, file_name))


def save_downstream_model(model, args):

    state = {'state_dict': model.state_dict()}
    save_dir = os.path.join(args.save_dir, args.experiment)
    file_name=os.path.join(save_dir, 'downstream_model.pt')
    torch.save(state, file_name)
    print("INFO: Save Dir:[{}]".format(file_name))


# loss
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5):
    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return - torch.log(x.mean())


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def contrastive_acc(x, t=0.5, topk=(1,)):

    maxk = max(topk)
    batch_size = x.shape[0]
    target = torch.tensor([i for i in range(batch_size)])
    x = pair_cosine_similarity(x)
    for i in range(len(x)): x[i][i] = 0.
    
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    
    _, pred = x.topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):

    assert warmup_steps >= 0
    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
    return lr


