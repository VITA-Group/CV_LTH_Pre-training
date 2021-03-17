import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


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

def pruning_model(model, px, conv1=True):

    print('start unstructured pruning')
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

def prune_model_custom(model, mask_dict, conv1=True):

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

def remove_prune(model, conv1=True):
    
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

def loading_weight(model, path):

    pretrained_weight = torch.load(path)['state_dict']
    new_dict = {}
    for key in pretrained_weight.keys():
        new_dict[key[len('module.encoder_q.'):]] = pretrained_weight[key]

    model.encoder_q.load_state_dict(new_dict)
    model.encoder_k.load_state_dict(new_dict)

    print('loading pretrained weight')

def pruning_rewind(model, px, path):

    pruning_model(model.encoder_q, px, False)
    pruning_model(model.encoder_k, px, False)

    current_mask = extract_mask(model.encoder_q.state_dict())
    remove_prune(model.encoder_q, False)
    remove_prune(model.encoder_k, False)

    #rewind to pretrianed weight
    loading_weight(model, path)
    prune_model_custom(model.encoder_q, current_mask, False)
    prune_model_custom(model.encoder_k, current_mask, False)

    check_sparsity(model.encoder_q, False)
    check_sparsity(model.encoder_k, False)







