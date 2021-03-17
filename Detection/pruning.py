import pdb
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune




def see_zero_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))     
    print('INFO: Remain Weight [{:.4f}%] '.format(100 * (1 - zero_sum / sum_list)))



def extract_mask(model_dict):
    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict


def pruning_model(model, px, exclude_first=False, random=False):

    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))
    
    if exclude_first:
        parameters_to_prune = parameters_to_prune[1:]
        print("Exclude first conv")

    parameters_to_prune = tuple(parameters_to_prune)

    if random:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=px,
        )
    else:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )


def resume_save_epoch(epoch, model, optimizer, path):

    save_name = path + '/resume.pt'
    ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
            }
    torch.save(ckpt, save_name)
    print("-" * 100)
    print("INFO: Save epoch:[{}] model at Dir:[{}]".format(epoch, save_name))
    print("-" * 100)


def resume_begin(model, optimizer, path):

    load_name = path + '/resume.pt'
    ckpt = torch.load(load_name)
    epoch = ckpt['epoch'] + 1
    optimizer.load_state_dict(ckpt["optimizer"])
    model.load_state_dict(ckpt["model"])
    print("INFO: Resume DIR:[{}] at Epoch:[{}]".format(path, epoch))
    return epoch



def imp_pruning_yolo_resnet50(model, mask_dict):

    module_to_prune = []
    mask_to_prune = []

    # module_to_prune.append(model.conv1)
    # mask_to_prune.append(mask_dict['conv1.weight_mask'])

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


def save_dicts_and_masks(imp_num, model, name):
    
    torch.save({'imp':imp_num,
                'all_state_dict': model.state_dict(),
                'first_conv1': model._Build_Model__yolov4.backbone.conv1.state_dict(),
                'backbone': model._Build_Model__yolov4.backbone.state_dict(),
                }, name)

def recover(all_ckpt, net, if_module=True):

    print("Recover First Conv and Others")
    if if_module:
        net.module.base[0].load_state_dict(all_ckpt['first_conv1'])
        net.module.L2Norm.load_state_dict(all_ckpt['L2Norm.weight'])
        net.module.extras.load_state_dict(all_ckpt['extras'])
        net.module.loc.load_state_dict(all_ckpt['loc'])
        net.module.conf.load_state_dict(all_ckpt['conf'])
    else:
        net.base[0].load_state_dict(all_ckpt['first_conv1'])
        net.L2Norm.load_state_dict(all_ckpt['L2Norm.weight'])
        net.extras.load_state_dict(all_ckpt['extras'])
        net.loc.load_state_dict(all_ckpt['loc'])
        net.conf.load_state_dict(all_ckpt['conf'])

def convert_dict(input_dict):
    
    nk0={'conv1':'0','bn1':'1'}
    bb=np.array([3,4,6,3])
    #old_key=
    ssd_dict = {}
    for k, v in input_dict.items():
        key=k.split('.')
        nk=''
        if key[0] in nk0.keys():
            key[0]=nk0[key[0]]
            nk=".".join(key)
            ssd_dict[nk]=v
        elif key[0].startswith('layer'):
            layer=int(key[0][-1])-1
            key[0]= str(bb[:layer].sum()+int(key[1])+4)
            del key[1]
            nk=".".join(key)
            ssd_dict[nk]=v
    return ssd_dict


def get_rid_of_mask_and_orig(ckpt):
    
    clean_mask_dict = {k : v for k, v in ckpt.items() if '_mask' not in k}
    out_dict = {}
    for k, v in clean_mask_dict.items():
        if '_orig' in k:
            out_dict[k[:-5]] = v
        else:
            out_dict[k] = v
    return out_dict


