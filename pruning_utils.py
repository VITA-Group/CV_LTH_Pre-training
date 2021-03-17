'''
functions for pruning related operations

'''

import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

__all__ = ['pruning_model', 'prune_model_custom', 'pruning_model_random', 'remove_prune',
            'extract_mask', 'reverse_mask', 'extract_main_weight',
            'check_sparsity']

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

def pruning_model_random(model, px, conv1=False):

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
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

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

def remove_prune(model, conv1=False):

    print('remove pruning')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    prune.remove(m,'weight')
                else:
                    print('skip conv1 for remove pruning')
            else:
                prune.remove(m,'weight')

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:

            if 'module' in key:
                new_key = key[len('module.'):]
            else:
                new_key = key 

            new_dict[new_key] = copy.deepcopy(model_dict[key])

    return new_dict

def reverse_mask(mask_dict):
    new_dict = {}

    for key in mask_dict.keys():

        new_dict[key] = 1 - mask_dict[key]

    return new_dict

def extract_main_weight(model_dict, fc=False, conv1=False):

    new_dict = {}

    for key in model_dict.keys():
        if not 'mask' in key:
            if not 'normalize' in key:

                if 'module' in key:
                    new_key = key[len('module.'):]
                else:
                    new_key = key 

                new_dict[new_key] = copy.deepcopy(model_dict[key])

    delete_keys = []

    if not fc:
        for key in new_dict.keys():
            if 'fc' in key:
                delete_keys.append(key)
    if not conv1:
        delete_keys.append('conv1.weight')

    for key in delete_keys:
        print('delete key = {}'.format(key))
        del new_dict[key]

    return new_dict

def check_sparsity(model, conv1=True):
    
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

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)



