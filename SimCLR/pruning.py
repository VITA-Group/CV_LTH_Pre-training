import pdb
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


def prunning_and_rewind(model, sim_ckpt, px):
    
    print("INFO: Pruning Percent: [{}]".format(px))
    pruning_model(model.module, px, False)
    
    prun_model_dict = model.module.state_dict() 
    ori_num = len(prun_model_dict.keys())

    unpruned_state_dict = {k : v for k , v in sim_ckpt.items() if k in prun_model_dict.keys()}
    pruned_state_dict = {k + '_orig': v for k , v in sim_ckpt.items() if k + '_orig' in prun_model_dict.keys()}
    
    new_num = len(unpruned_state_dict.keys()) + len(pruned_state_dict.keys())

    prun_model_dict.update(unpruned_state_dict)
    prun_model_dict.update(pruned_state_dict)
    
    print("INFO: Rewind...[{}/{}]".format(new_num, ori_num))
    model.module.load_state_dict(prun_model_dict)
    check_sparsity(model.module, False)

def pruning_model(model, px, conv1=False):

    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    parameters_to_prune.append((m,'weight'))
                else:
                    print('skip conv1 for L1 unstructure global pruning')
            else:
                parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def check_sparsity(model, conv1=False):
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    sum_list = sum_list+float(m.weight.nelement())
                    zero_sum = zero_sum+float(torch.sum(m.weight == 0))    
                else:
                    print('skip conv1 for sparsity checking')
            else:
                sum_list = sum_list+float(m.weight.nelement())
                zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    print('INFO: Remain Weight [{:.4f}%] '.format(100 * (1 - zero_sum / sum_list)))

def prune_model_custom(model, mask_dict, conv1=False):

    print('start unstructured pruning with custom mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
                else:
                    print('skip conv1 for custom pruning')
            else:
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:

            if 'module' in key:
                new_key = key[len('module.'):]
            else:
                new_key = key 

            new_dict[new_key] = model_dict[key]

    return new_dict
