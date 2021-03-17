from args import Configer
import utils
import train
import pdb
import pruning
import warnings
import torch
import re
import os
import pdb
warnings.filterwarnings("ignore")


def main():

    configer = Configer()
    args = configer.args_setup()
    configer.seed_setup()
    device = configer.gpu_setup()
    model = configer.model_setup(project_head=args.phead)
    sim_ckpt = configer.get_simclr_model() 

    # load pre-trained model
    configer.loading_ckpt(model, sim_ckpt)
    train_loader, val_loader, val_train_loader, test_loader = configer.dataloader_setup()
    optimizer = configer.optimizer_setup(model)

    start_epoch = 1
    if args.resume != '':

        print("Begin Resume:[{}]".format(args.resume))
        mask_ckpt = torch.load(args.resume)
        mask_dict = pruning.extract_mask(mask_ckpt['state_dict'])
        pruning.prune_model_custom(model.module, mask_dict)
        model.load_state_dict(mask_ckpt['state_dict'])        
        pruning.check_sparsity(model)
        print("Change SimCLR Mask Dir:[{}]".format(args.resume))

        if 'epoch' in mask_ckpt.keys() and 'optim' in mask_ckpt.keys():
            start_epoch = mask_ckpt['epoch'] + 1
            optimizer.load_state_dict(mask_ckpt['optim'])
            print("Resume the checkpoint [{}] from epoch [{}]"
                .format(args.resume, mask_ckpt['epoch']))
            print("Finish resume!")
        else:
            print("Cannot resume start epoch and optimizer")
            assert False

    if args.resume == '':
        pruning.prunning_and_rewind(model, sim_ckpt, args.prun_percent)

    for epoch in range(start_epoch, args.epochs + 1):

        train_loss = train.train(train_loader, model, optimizer, epoch, args, device)
        val_cacc1, val_cacc3, val_cacc5 = train.val_cacc(val_loader, model, args, device, mlp=False)
        
        print("Epoch[{}] | Loss [{:.4f}]  Test Contrastive Top1: [{:.4f}]  Top3: [{:.4f}]  Top5: [{:.4f}]"
            .format(epoch, train_loss, val_cacc1, val_cacc3, val_cacc5))
        print('-' * 80)

        utils.save_checkpoint(epoch, model, optimizer, args, 'model_{}.pt'.format(epoch))
        if epoch % args.prun_epoch == 0:

            utils.save_checkpoint(epoch, model, optimizer, args, 'model_pruned{}.pt'.format(epoch))
            pruning.prunning_and_rewind(model, sim_ckpt, args.prun_percent)


if __name__ == "__main__":
    main()
