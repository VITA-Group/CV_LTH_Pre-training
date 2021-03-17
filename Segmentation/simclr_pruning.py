import pdb
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


def remove_model(model):
    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    for module_pair in parameters_to_prune:
        prune.remove(module_pair[0], module_pair[1])


def prunning_and_rewind_module(model, sim_ckpt, px):
    
    print("INFO: Rruning Percent: [{}]".format(px))
    pruning_model(model, px)
    
    prun_model_dict = model.module.state_dict() # 375
    ori_num = len(prun_model_dict.keys())

    unpruned_state_dict = {k : v for k , v in sim_ckpt.items() if k in prun_model_dict.keys()}
    pruned_state_dict = {k + '_orig': v for k , v in sim_ckpt.items() if k + '_orig' in prun_model_dict.keys()}
    
    new_num = len(unpruned_state_dict.keys()) + len(pruned_state_dict.keys())

    prun_model_dict.update(unpruned_state_dict)
    prun_model_dict.update(pruned_state_dict)
    
    print("INFO: Reload...[{}/{}]".format(new_num, ori_num))
    model.module.load_state_dict(prun_model_dict)
    see_zero_rate(model)


def prunning_and_rewind(model, epoch0_ckpt, px):
    
    print("INFO: Pruning Percent: [{}]".format(px))
    pruning_model(model, px)
    
    prun_model_dict = model.state_dict() # 375
    ori_num = len(prun_model_dict.keys())
    
    unpruned_state_dict = {k : v for k , v in epoch0_ckpt.items() if k in prun_model_dict.keys()}
    pruned_state_dict = {k + '_orig': v for k , v in epoch0_ckpt.items() if k + '_orig' in prun_model_dict.keys()}
    
    new_num = len(unpruned_state_dict.keys()) + len(pruned_state_dict.keys())

    prun_model_dict.update(unpruned_state_dict)
    prun_model_dict.update(pruned_state_dict)
    
    print("INFO: Reload...[{}/{}]".format(new_num, ori_num))
    model.load_state_dict(prun_model_dict)
    see_zero_rate(model)


def rewind_model(model, epoch0_ckpt):

    prun_model_dict = model.state_dict() # 375
    ori_num = len(prun_model_dict.keys())
    
    unpruned_state_dict = {k : v for k , v in epoch0_ckpt.items() if k in prun_model_dict.keys()}
    pruned_state_dict = {k + '_orig': v for k , v in epoch0_ckpt.items() if k + '_orig' in prun_model_dict.keys()}
    
    new_num = len(unpruned_state_dict.keys()) + len(pruned_state_dict.keys())
    
    prun_model_dict.update(unpruned_state_dict)
    prun_model_dict.update(pruned_state_dict)
    
    print("INFO: Rewind...[{}/{}]".format(new_num, ori_num))
    model.load_state_dict(prun_model_dict)
    see_zero_rate(model)



def pruning_model(model, px):

    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def see_zero_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))     
    print('INFO: Remain Weight [{:.4f}%] '.format(100 * (1 - zero_sum / sum_list)))




def simclr_pruning_model_custom_res50v1(model, mask_dict, no_conv1=True):

    module_to_prune = []
    mask_to_prune = []

    if no_conv1 == False:
        module_to_prune.append(model.conv1)
        mask_to_prune.append(mask_dict['module.conv1.weight_mask'])
    
    #module.layer1
    module_to_prune.append(model.layer1[0].conv1)
    mask_to_prune.append(mask_dict['module.layer1.0.conv1.weight_mask'])
    module_to_prune.append(model.layer1[0].conv2)
    mask_to_prune.append(mask_dict['module.layer1.0.conv2.weight_mask'])
    module_to_prune.append(model.layer1[0].conv3)
    mask_to_prune.append(mask_dict['module.layer1.0.conv3.weight_mask'])
    module_to_prune.append(model.layer1[0].downsample[0])
    mask_to_prune.append(mask_dict['module.layer1.0.downsample.0.weight_mask'])
    module_to_prune.append(model.layer1[1].conv1)
    mask_to_prune.append(mask_dict['module.layer1.1.conv1.weight_mask'])
    module_to_prune.append(model.layer1[1].conv2)
    mask_to_prune.append(mask_dict['module.layer1.1.conv2.weight_mask'])
    module_to_prune.append(model.layer1[1].conv3)
    mask_to_prune.append(mask_dict['module.layer1.1.conv3.weight_mask'])
    module_to_prune.append(model.layer1[2].conv1)
    mask_to_prune.append(mask_dict['module.layer1.2.conv1.weight_mask'])
    module_to_prune.append(model.layer1[2].conv2)
    mask_to_prune.append(mask_dict['module.layer1.2.conv2.weight_mask'])
    module_to_prune.append(model.layer1[2].conv3)
    mask_to_prune.append(mask_dict['module.layer1.2.conv3.weight_mask'])

    #module.layer2
    module_to_prune.append(model.layer2[0].conv1)
    mask_to_prune.append(mask_dict['module.layer2.0.conv1.weight_mask'])
    module_to_prune.append(model.layer2[0].conv2)
    mask_to_prune.append(mask_dict['module.layer2.0.conv2.weight_mask'])
    module_to_prune.append(model.layer2[0].conv3)
    mask_to_prune.append(mask_dict['module.layer2.0.conv3.weight_mask'])
    module_to_prune.append(model.layer2[0].downsample[0])
    mask_to_prune.append(mask_dict['module.layer2.0.downsample.0.weight_mask'])
    module_to_prune.append(model.layer2[1].conv1)
    mask_to_prune.append(mask_dict['module.layer2.1.conv1.weight_mask'])
    module_to_prune.append(model.layer2[1].conv2)
    mask_to_prune.append(mask_dict['module.layer2.1.conv2.weight_mask'])
    module_to_prune.append(model.layer2[1].conv3)
    mask_to_prune.append(mask_dict['module.layer2.1.conv3.weight_mask'])
    module_to_prune.append(model.layer2[2].conv1)
    mask_to_prune.append(mask_dict['module.layer2.2.conv1.weight_mask'])
    module_to_prune.append(model.layer2[2].conv2)
    mask_to_prune.append(mask_dict['module.layer2.2.conv2.weight_mask'])
    module_to_prune.append(model.layer2[2].conv3)
    mask_to_prune.append(mask_dict['module.layer2.2.conv3.weight_mask'])
    module_to_prune.append(model.layer2[3].conv1)
    mask_to_prune.append(mask_dict['module.layer2.3.conv1.weight_mask'])
    module_to_prune.append(model.layer2[3].conv2)
    mask_to_prune.append(mask_dict['module.layer2.3.conv2.weight_mask'])
    module_to_prune.append(model.layer2[3].conv3)
    mask_to_prune.append(mask_dict['module.layer2.3.conv3.weight_mask'])

    #module.layer3
    module_to_prune.append(model.layer3[0].conv1)
    mask_to_prune.append(mask_dict['module.layer3.0.conv1.weight_mask'])
    module_to_prune.append(model.layer3[0].conv2)
    mask_to_prune.append(mask_dict['module.layer3.0.conv2.weight_mask'])
    module_to_prune.append(model.layer3[0].conv3)
    mask_to_prune.append(mask_dict['module.layer3.0.conv3.weight_mask'])
    module_to_prune.append(model.layer3[0].downsample[0])
    mask_to_prune.append(mask_dict['module.layer3.0.downsample.0.weight_mask'])
    module_to_prune.append(model.layer3[1].conv1)
    mask_to_prune.append(mask_dict['module.layer3.1.conv1.weight_mask'])
    module_to_prune.append(model.layer3[1].conv2)
    mask_to_prune.append(mask_dict['module.layer3.1.conv2.weight_mask'])
    module_to_prune.append(model.layer3[1].conv3)
    mask_to_prune.append(mask_dict['module.layer3.1.conv3.weight_mask'])
    module_to_prune.append(model.layer3[2].conv1)
    mask_to_prune.append(mask_dict['module.layer3.2.conv1.weight_mask'])
    module_to_prune.append(model.layer3[2].conv2)
    mask_to_prune.append(mask_dict['module.layer3.2.conv2.weight_mask'])
    module_to_prune.append(model.layer3[2].conv3)
    mask_to_prune.append(mask_dict['module.layer3.2.conv3.weight_mask'])
    module_to_prune.append(model.layer3[3].conv1)
    mask_to_prune.append(mask_dict['module.layer3.3.conv1.weight_mask'])
    module_to_prune.append(model.layer3[3].conv2)
    mask_to_prune.append(mask_dict['module.layer3.3.conv2.weight_mask'])
    module_to_prune.append(model.layer3[3].conv3)
    mask_to_prune.append(mask_dict['module.layer3.3.conv3.weight_mask'])
    module_to_prune.append(model.layer3[4].conv1)
    mask_to_prune.append(mask_dict['module.layer3.4.conv1.weight_mask'])
    module_to_prune.append(model.layer3[4].conv2)
    mask_to_prune.append(mask_dict['module.layer3.4.conv2.weight_mask'])
    module_to_prune.append(model.layer3[4].conv3)
    mask_to_prune.append(mask_dict['module.layer3.4.conv3.weight_mask'])
    module_to_prune.append(model.layer3[5].conv1)
    mask_to_prune.append(mask_dict['module.layer3.5.conv1.weight_mask'])
    module_to_prune.append(model.layer3[5].conv2)
    mask_to_prune.append(mask_dict['module.layer3.5.conv2.weight_mask'])
    module_to_prune.append(model.layer3[5].conv3)
    mask_to_prune.append(mask_dict['module.layer3.5.conv3.weight_mask'])

    #module.layer4
    module_to_prune.append(model.layer4[0].conv1)
    mask_to_prune.append(mask_dict['module.layer4.0.conv1.weight_mask'])
    module_to_prune.append(model.layer4[0].conv2)
    mask_to_prune.append(mask_dict['module.layer4.0.conv2.weight_mask'])
    module_to_prune.append(model.layer4[0].conv3)
    mask_to_prune.append(mask_dict['module.layer4.0.conv3.weight_mask'])
    module_to_prune.append(model.layer4[0].downsample[0])
    mask_to_prune.append(mask_dict['module.layer4.0.downsample.0.weight_mask'])
    module_to_prune.append(model.layer4[1].conv1)
    mask_to_prune.append(mask_dict['module.layer4.1.conv1.weight_mask'])
    module_to_prune.append(model.layer4[1].conv2)
    mask_to_prune.append(mask_dict['module.layer4.1.conv2.weight_mask'])
    module_to_prune.append(model.layer4[1].conv3)
    mask_to_prune.append(mask_dict['module.layer4.1.conv3.weight_mask'])
    module_to_prune.append(model.layer4[2].conv1)
    mask_to_prune.append(mask_dict['module.layer4.2.conv1.weight_mask'])
    module_to_prune.append(model.layer4[2].conv2)
    mask_to_prune.append(mask_dict['module.layer4.2.conv2.weight_mask'])
    module_to_prune.append(model.layer4[2].conv3)
    mask_to_prune.append(mask_dict['module.layer4.2.conv3.weight_mask'])

    for ii in range(len(module_to_prune)):
        prune.CustomFromMask.apply(module_to_prune[ii], 'weight', mask=mask_to_prune[ii])


def imagenet_pruning_model_custom_res50v1(model, mask_dict, no_conv1=True ):

    module_to_prune = []
    mask_to_prune = []

    if no_conv1 == False:
        module_to_prune.append(model.conv1)
        mask_to_prune.append(mask_dict['conv1.weight_mask'])
    
    #layer1
    module_to_prune.append(model.layer1[0].conv1)
    mask_to_prune.append(mask_dict['layer1.0.conv1.weight_mask'])
    module_to_prune.append(model.layer1[0].conv2)
    mask_to_prune.append(mask_dict['layer1.0.conv2.weight_mask'])
    module_to_prune.append(model.layer1[0].conv3)
    mask_to_prune.append(mask_dict['layer1.0.conv3.weight_mask'])
    module_to_prune.append(model.layer1[0].downsample[0])
    mask_to_prune.append(mask_dict['layer1.0.downsample.0.weight_mask'])
    module_to_prune.append(model.layer1[1].conv1)
    mask_to_prune.append(mask_dict['layer1.1.conv1.weight_mask'])
    module_to_prune.append(model.layer1[1].conv2)
    mask_to_prune.append(mask_dict['layer1.1.conv2.weight_mask'])
    module_to_prune.append(model.layer1[1].conv3)
    mask_to_prune.append(mask_dict['layer1.1.conv3.weight_mask'])
    module_to_prune.append(model.layer1[2].conv1)
    mask_to_prune.append(mask_dict['layer1.2.conv1.weight_mask'])
    module_to_prune.append(model.layer1[2].conv2)
    mask_to_prune.append(mask_dict['layer1.2.conv2.weight_mask'])
    module_to_prune.append(model.layer1[2].conv3)
    mask_to_prune.append(mask_dict['layer1.2.conv3.weight_mask'])

    #layer2
    module_to_prune.append(model.layer2[0].conv1)
    mask_to_prune.append(mask_dict['layer2.0.conv1.weight_mask'])
    module_to_prune.append(model.layer2[0].conv2)
    mask_to_prune.append(mask_dict['layer2.0.conv2.weight_mask'])
    module_to_prune.append(model.layer2[0].conv3)
    mask_to_prune.append(mask_dict['layer2.0.conv3.weight_mask'])
    module_to_prune.append(model.layer2[0].downsample[0])
    mask_to_prune.append(mask_dict['layer2.0.downsample.0.weight_mask'])
    module_to_prune.append(model.layer2[1].conv1)
    mask_to_prune.append(mask_dict['layer2.1.conv1.weight_mask'])
    module_to_prune.append(model.layer2[1].conv2)
    mask_to_prune.append(mask_dict['layer2.1.conv2.weight_mask'])
    module_to_prune.append(model.layer2[1].conv3)
    mask_to_prune.append(mask_dict['layer2.1.conv3.weight_mask'])
    module_to_prune.append(model.layer2[2].conv1)
    mask_to_prune.append(mask_dict['layer2.2.conv1.weight_mask'])
    module_to_prune.append(model.layer2[2].conv2)
    mask_to_prune.append(mask_dict['layer2.2.conv2.weight_mask'])
    module_to_prune.append(model.layer2[2].conv3)
    mask_to_prune.append(mask_dict['layer2.2.conv3.weight_mask'])
    module_to_prune.append(model.layer2[3].conv1)
    mask_to_prune.append(mask_dict['layer2.3.conv1.weight_mask'])
    module_to_prune.append(model.layer2[3].conv2)
    mask_to_prune.append(mask_dict['layer2.3.conv2.weight_mask'])
    module_to_prune.append(model.layer2[3].conv3)
    mask_to_prune.append(mask_dict['layer2.3.conv3.weight_mask'])

    #layer3
    module_to_prune.append(model.layer3[0].conv1)
    mask_to_prune.append(mask_dict['layer3.0.conv1.weight_mask'])
    module_to_prune.append(model.layer3[0].conv2)
    mask_to_prune.append(mask_dict['layer3.0.conv2.weight_mask'])
    module_to_prune.append(model.layer3[0].conv3)
    mask_to_prune.append(mask_dict['layer3.0.conv3.weight_mask'])
    module_to_prune.append(model.layer3[0].downsample[0])
    mask_to_prune.append(mask_dict['layer3.0.downsample.0.weight_mask'])
    module_to_prune.append(model.layer3[1].conv1)
    mask_to_prune.append(mask_dict['layer3.1.conv1.weight_mask'])
    module_to_prune.append(model.layer3[1].conv2)
    mask_to_prune.append(mask_dict['layer3.1.conv2.weight_mask'])
    module_to_prune.append(model.layer3[1].conv3)
    mask_to_prune.append(mask_dict['layer3.1.conv3.weight_mask'])
    module_to_prune.append(model.layer3[2].conv1)
    mask_to_prune.append(mask_dict['layer3.2.conv1.weight_mask'])
    module_to_prune.append(model.layer3[2].conv2)
    mask_to_prune.append(mask_dict['layer3.2.conv2.weight_mask'])
    module_to_prune.append(model.layer3[2].conv3)
    mask_to_prune.append(mask_dict['layer3.2.conv3.weight_mask'])
    module_to_prune.append(model.layer3[3].conv1)
    mask_to_prune.append(mask_dict['layer3.3.conv1.weight_mask'])
    module_to_prune.append(model.layer3[3].conv2)
    mask_to_prune.append(mask_dict['layer3.3.conv2.weight_mask'])
    module_to_prune.append(model.layer3[3].conv3)
    mask_to_prune.append(mask_dict['layer3.3.conv3.weight_mask'])
    module_to_prune.append(model.layer3[4].conv1)
    mask_to_prune.append(mask_dict['layer3.4.conv1.weight_mask'])
    module_to_prune.append(model.layer3[4].conv2)
    mask_to_prune.append(mask_dict['layer3.4.conv2.weight_mask'])
    module_to_prune.append(model.layer3[4].conv3)
    mask_to_prune.append(mask_dict['layer3.4.conv3.weight_mask'])
    module_to_prune.append(model.layer3[5].conv1)
    mask_to_prune.append(mask_dict['layer3.5.conv1.weight_mask'])
    module_to_prune.append(model.layer3[5].conv2)
    mask_to_prune.append(mask_dict['layer3.5.conv2.weight_mask'])
    module_to_prune.append(model.layer3[5].conv3)
    mask_to_prune.append(mask_dict['layer3.5.conv3.weight_mask'])

    #layer4
    module_to_prune.append(model.layer4[0].conv1)
    mask_to_prune.append(mask_dict['layer4.0.conv1.weight_mask'])
    module_to_prune.append(model.layer4[0].conv2)
    mask_to_prune.append(mask_dict['layer4.0.conv2.weight_mask'])
    module_to_prune.append(model.layer4[0].conv3)
    mask_to_prune.append(mask_dict['layer4.0.conv3.weight_mask'])
    module_to_prune.append(model.layer4[0].downsample[0])
    mask_to_prune.append(mask_dict['layer4.0.downsample.0.weight_mask'])
    module_to_prune.append(model.layer4[1].conv1)
    mask_to_prune.append(mask_dict['layer4.1.conv1.weight_mask'])
    module_to_prune.append(model.layer4[1].conv2)
    mask_to_prune.append(mask_dict['layer4.1.conv2.weight_mask'])
    module_to_prune.append(model.layer4[1].conv3)
    mask_to_prune.append(mask_dict['layer4.1.conv3.weight_mask'])
    module_to_prune.append(model.layer4[2].conv1)
    mask_to_prune.append(mask_dict['layer4.2.conv1.weight_mask'])
    module_to_prune.append(model.layer4[2].conv2)
    mask_to_prune.append(mask_dict['layer4.2.conv2.weight_mask'])
    module_to_prune.append(model.layer4[2].conv3)
    mask_to_prune.append(mask_dict['layer4.2.conv3.weight_mask'])

    for ii in range(len(module_to_prune)):
        prune.CustomFromMask.apply(module_to_prune[ii], 'weight', mask=mask_to_prune[ii])



    

def convert_moduledict_to_dict(module_dict):
    
    new_dict = {}
    for key in module_dict.keys():
        new_key = key[7:]
        new_dict[new_key] = module_dict[key]

    return new_dict




def extract_mask(model_dict):
    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict





    

def add_orig_to_weight(model_dict):

    new_dict = {}

    mask_to_prune = []
    mask_to_prune.append('conv1.weight')
    mask_to_prune.append('layer1.0.conv1.weight')
    mask_to_prune.append('layer1.0.conv2.weight')
    mask_to_prune.append('layer1.1.conv1.weight')
    mask_to_prune.append('layer1.1.conv2.weight')
    mask_to_prune.append('layer2.0.conv1.weight')
    mask_to_prune.append('layer2.0.conv2.weight')
    mask_to_prune.append('layer2.1.conv1.weight')
    mask_to_prune.append('layer2.1.conv2.weight')
    mask_to_prune.append('layer2.0.downsample.0.weight')
    mask_to_prune.append('layer3.0.conv1.weight')
    mask_to_prune.append('layer3.0.conv2.weight')
    mask_to_prune.append('layer3.1.conv1.weight')
    mask_to_prune.append('layer3.1.conv2.weight')
    mask_to_prune.append('layer3.0.downsample.0.weight')
    mask_to_prune.append('layer4.0.conv1.weight')
    mask_to_prune.append('layer4.0.conv2.weight')
    mask_to_prune.append('layer4.1.conv1.weight')
    mask_to_prune.append('layer4.1.conv2.weight')
    mask_to_prune.append('layer4.0.downsample.0.weight')

    for key in model_dict.keys():
        if not 'fc' in key:

            if key in mask_to_prune:
                new_key = key+'_orig'
            else:
                new_key = key
            
            new_dict[new_key] = model_dict[key]
    
    return new_dict


