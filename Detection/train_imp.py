import logging
import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter
import config.yolov4_config as cfg
from utils import cosine_lr_scheduler
from utils.log import Logger
# from apex import amp
import torchvision.models as models
from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator
import pdb
import ap
import random
import pruning
import numpy as np

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets

def seed_setup(args):
        
    print('INFO: Seeds : [{}]'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

class Trainer(object):
    def __init__(self, weight_path, resume, gpu_id, accumulate, fp_16, args):
        init_seeds(0)
        seed_setup(args)
        self.args = args
        self.fp_16 = fp_16
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.accumulate = accumulate
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train:
            print("Using multi scales training")
        else:
            print("train img size is {}".format(cfg.TRAIN["TRAIN_IMG_SIZE"]))
        self.train_dataset = data.Build_Dataset(
            anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"]
        )
        self.epochs = (
            cfg.TRAIN["YOLO_EPOCHS"]
            if cfg.MODEL_TYPE["TYPE"] == "YOLOv4"
            else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN["BATCH_SIZE"],
            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
            shuffle=True,
            pin_memory=True,
        )

        self.yolov4 = Build_Model(weight_path=weight_path, resume=resume).to(self.device)
        self.optimizer = optim.SGD(
            self.yolov4.parameters(),
            lr=cfg.TRAIN["LR_INIT"],
            momentum=cfg.TRAIN["MOMENTUM"],
            weight_decay=cfg.TRAIN["WEIGHT_DECAY"],
        )

        self.criterion = YoloV4Loss(
            anchors=cfg.MODEL["ANCHORS"],
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],
        )

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader),
        )
        if resume:
            self.__load_resume_weights(weight_path)

    def __load_resume_weights(self, weight_path):

        last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
        chkpt = torch.load(last_weight, map_location=self.device)
        self.yolov4.load_state_dict(chkpt["model"])

        self.start_epoch = chkpt["epoch"] + 1
        if chkpt["optimizer"] is not None:
            self.optimizer.load_state_dict(chkpt["optimizer"])
            self.best_mAP = chkpt["best_mAP"]
        del chkpt


    def train(self):
        global writer
        logger.info(
            "Training start,img size is: {:d},batchsize is: {:d},work number is {:d}".format(
                cfg.TRAIN["TRAIN_IMG_SIZE"],
                cfg.TRAIN["BATCH_SIZE"],
                cfg.TRAIN["NUMBER_WORKERS"],
            )
        )
        logger.info("Train datasets number is : {}".format(len(self.train_dataset)))

        print('Loading base network...')
        if self.args.imp_type == 'imagenet':
            print("-" * 100)
            print("Use resnet50 [imagnet] weight")
            print("-" * 100)

        elif self.args.imp_type == 'moco':
            print("-" * 100)
            print("load resnet50 [moco] weight")
            print("-" * 100)
            moco_ckpt = torch.load('./weight/moco_v2_800ep_pretrain.pth.tar', map_location='cuda')['state_dict']
            resnet50_pytorch = models.resnet50(pretrained=False)
            moco_state_dict = {k[17:] : v for k , v in moco_ckpt.items() if k[17:] in resnet50_pytorch.state_dict().keys()}
            overlap_state_dict = {k : v for k , v in moco_state_dict.items() if k in self.yolov4._Build_Model__yolov4.backbone.state_dict().keys()} 
            print("MOCO Overlap[{}/{}]".format(overlap_state_dict.keys().__len__(), self.yolov4._Build_Model__yolov4.backbone.state_dict().keys().__len__()))
            ori = self.yolov4._Build_Model__yolov4.backbone.state_dict()
            ori.update(overlap_state_dict)
            self.yolov4._Build_Model__yolov4.backbone.load_state_dict(ori)

        elif self.args.imp_type == 'simclr':
            print("-" * 100)
            print("load resnet50 [SimCLR] weight")
            print("-" * 100)
            base_weights = torch.load('./weight/simclr_weight.pt', map_location='cuda')['state_dict']
            overlap_state_dict = {k : v for k , v in base_weights.items() if k in self.yolov4._Build_Model__yolov4.backbone.state_dict().keys()} 
            print("SimCLR Overlap[{}/{}]".format(overlap_state_dict.keys().__len__(), self.yolov4._Build_Model__yolov4.backbone.state_dict().keys().__len__()))
            self.yolov4._Build_Model__yolov4.backbone.load_state_dict(overlap_state_dict)

        else: assert False

        OUTDIR = 'seed{}_{}_imp_ckpt'.format(self.args.seed, self.args.imp_type)
        OUTDIRS = OUTDIR + '/model{}'.format(self.args.imp_num)
        if not os.path.exists(OUTDIRS): os.makedirs(OUTDIRS)
        
        if self.args.imp_num == 0:
            print("Begining [{}] IMP:[{}]".format(self.args.imp_type, self.args.imp_num))
            pruning.save_dicts_and_masks(imp_num=self.args.imp_num, model=self.yolov4, name='./'+ OUTDIR +'/backbone_ori.pth')
        else:
            
            print("Begining [{}] IMP:[{}]".format(self.args.imp_type, self.args.imp_num))
            load_name = './'+ OUTDIR +'/backbone_imp{}.pth'.format(self.args.imp_num)
            print("Load All CKPT Dir: {}".format(load_name))
            ###### load all others ########
            all_ckpt = torch.load(load_name, map_location="cuda")
            model_state_dict = self.yolov4.state_dict()
            overlap_all = {k : v for k , v in all_ckpt['all_state_dict'].items() if k in model_state_dict.keys()}
            model_state_dict.update(overlap_all)
            self.yolov4.load_state_dict(model_state_dict)
            ###### load conv1 ########
            self.yolov4._Build_Model__yolov4.backbone.conv1.load_state_dict(all_ckpt['first_conv1'])
            ###### pruning ######
            mask_dict = pruning.extract_mask(all_ckpt['backbone'])
            print("Load Mask Len:[{}]".format(len(mask_dict.keys())))
            pruning.imp_pruning_yolo_resnet50(self.yolov4._Build_Model__yolov4.backbone, mask_dict)
            pruning.see_zero_rate(self.yolov4._Build_Model__yolov4.backbone)
            print("-" * 100)
            print("Finish Process!")
            print("-" * 100)

            # print("Begining [{}] IMP:[{}]".format(self.args.imp_type, self.args.imp_num))
            # ###### Rewind all #######
            # load_name_rewind = './'+ OUTDIR +'/backbone_ori.pth'
            # rewind_ckpt = torch.load(load_name_rewind, map_location="cuda")
            # model_state_dict = self.yolov4.state_dict()
            # overlap_all = {k : v for k , v in rewind_ckpt['all_state_dict'].items() if k in model_state_dict.keys()}
            # model_state_dict.update(overlap_all)
            # self.yolov4.load_state_dict(model_state_dict)
            # ###### pruning ##########
            # load_name_mask = './'+ OUTDIR +'/backbone_imp{}.pth'.format(self.args.imp_num)
            # mask_ckpt = torch.load(load_name_mask, map_location="cuda")
            # mask_dict = pruning.extract_mask(mask_ckpt['backbone'])
            # print("Load Mask Len:[{}]".format(len(mask_dict.keys())))
            # pruning.imp_pruning_yolo_resnet50(self.yolov4._Build_Model__yolov4.backbone, mask_dict)
            # pruning.see_zero_rate(self.yolov4._Build_Model__yolov4.backbone)
            # print("-" * 100)
            # print("Finish Process!")
            # print("-" * 100)

        if self.args.resume_all:
            self.start_epoch = pruning.resume_begin(self.yolov4, self.optimizer, OUTDIRS)
            logger.info(" =======  Resume  Training at Epoch:[{}]  ======".format(self.start_epoch))
        else:
            logger.info(" =======  Start  Training   ======")
        start = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            
            self.yolov4.train()
            mloss = torch.zeros(4)

            for i, (imgs,label_sbbox,label_mbbox,label_lbbox,sbboxes,mbboxes,lbboxes) in enumerate(self.train_dataloader):
                self.scheduler.step(
                    len(self.train_dataloader)
                    / (cfg.TRAIN["BATCH_SIZE"])
                    * epoch
                    + i
                )
                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov4(imgs)
                loss, loss_ciou, loss_conf, loss_cls = self.criterion(p,p_d,label_sbbox,label_mbbox,
                                                                      label_lbbox,sbboxes,mbboxes,lbboxes)
                if self.fp_16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # Accumulate gradient for x batches before optimizing
                if i % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i % 10 == 0:

                    logger.info(
                        "Epoch:[{:3}/{}],step:[{:3}/{}],img_size:[{:3}],total_loss:{:.4f}|loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|lr:{:.4f}".format(
                            epoch,
                            self.epochs,
                            i,
                            len(self.train_dataloader) - 1,
                            self.train_dataset.img_size,
                            mloss[3],
                            mloss[0],
                            mloss[1],
                            mloss[2],
                            self.optimizer.param_groups[0]["lr"],
                        )
                    )
                    
                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i + 1) % 10 == 0:
                    self.train_dataset.img_size = (random.choice(range(10, 20)) * 32)
                #break
            pruning.resume_save_epoch(epoch, self.yolov4, self.optimizer, path=OUTDIRS)
            #if epoch == 5: break

        print("=========EVAL=========")
        with torch.no_grad():
            
            APs_all, inference_time = Evaluator(self.yolov4, showatt=False, result_path=OUTDIRS).APs_voc()
            ap_final, ap50, ap75 = ap.compute_all_aps(APs_all, self.train_dataset.num_classes)
            logger.info("AP:[{}] AP50:[{}] AP75:[{}]".format(ap_final,ap50,ap75))
            logger.info("inference time: {:.2f} ms".format(inference_time))

        end = time.time()
        logger.info("  ===cost time:{:.4f}s".format(end - start))
        logger.info("syd {} AP:[{:.4f}] AP50:[{:.4f}] AP75:[{:.4f}]".format(self.args.imp_num, ap_final,ap50,ap75))
        
        print('-' * 100)
        print("Finish Training, Begin Pruning...")
        print('-' * 100)
        pruning.pruning_model(self.yolov4._Build_Model__yolov4.backbone, 0.2, exclude_first=True)
        pruning.see_zero_rate(self.yolov4._Build_Model__yolov4.backbone)
        save_name = './' + OUTDIR + '/backbone_imp{}.pth'.format(self.args.imp_num + 1)
        pruning.save_dicts_and_masks(imp_num=self.args.imp_num, model=self.yolov4, name=save_name)
        print("Save in {}".format(save_name))
        print('-' * 100)


if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_all",action="store_true",default=False,help="resume training flag")
    parser.add_argument('--imp_type', default='imagenet', type=str, help='imagenet, moco, simclr')     
    parser.add_argument('--mask_dir', default='', type=str, help='imagenet, moco, simclr') 
    parser.add_argument('--imp_num', default=-1, type=int, help='mask number')
    parser.add_argument('--seed', default=1, type=int, help='Seed')  
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weight/mobilenetv2.pth",
        help="weight file path",
    )  # weight/darknet53_448.weights
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training flag",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="whither use GPU(0) or CPU(-1)",
    )
    parser.add_argument("--log_path", type=str, default="log/", help="log path")
    parser.add_argument(
        "--accumulate",
        type=int,
        default=2,
        help="batches to accumulate before optimizing",
    )
    parser.add_argument(
        "--fp_16",
        type=bool,
        default=False,
        help="whither to use fp16 precision",
    )

    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + "/event")
    logger = Logger(
        log_file_name=opt.log_path + "/log.txt",
        log_level=logging.DEBUG,
        logger_name="YOLOv4",
    ).get_log()

    Trainer(
        weight_path=opt.weight_path,
        resume=opt.resume,
        gpu_id=opt.gpu_id,
        accumulate=opt.accumulate,
        fp_16=opt.fp_16,
        args=opt

    ).train()
