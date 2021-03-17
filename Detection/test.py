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

        print('Loading eval network dir [{}]'.format(self.args.model_dir))
        ckpt = torch.load(self.args.model_dir)
        model_state_dict = self.yolov4.state_dict()
        overlap = {k : v for k, v in ckpt.items() if k in model_state_dict.keys()}
        model_state_dict.update(overlap)
        self.yolov4.load_state_dict(model_state_dict)
        print("load weight:[{}/{}]".format(overlap.keys().__len__(), self.yolov4.state_dict().keys().__len__()))

        OUTDIRS = 'mobilenetv2_eval'
        if not os.path.exists(OUTDIRS): os.makedirs(OUTDIRS)
        print("=========EVAL=========")
        with torch.no_grad():
            
            # APs_all, inference_time = Evaluator(self.yolov4, showatt=False, result_path=OUTDIRS).APs_voc()
            # print("mAP:{}".format(APs_all / self.train_dataset.num_classes))
            ap_final, ap50, ap75 = ap.compute_all_aps(APs_all, self.train_dataset.num_classes)
            logger.info("AP:[{}] AP50:[{}] AP75:[{}]".format(ap_final,ap50,ap75))
            logger.info("inference time: {:.2f} ms".format(inference_time))

        logger.info("syd {} AP:[{:.4f}] AP50:[{:.4f}] AP75:[{:.4f}]".format(self.args.imp_num, ap_final,ap50,ap75))
        

if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()

    parser.add_argument("--resume_all",action="store_true",default=False,help="resume training flag")
    parser.add_argument('--imp_type', default='imagenet', type=str, help='imagenet, moco, simclr')     
    parser.add_argument('--model_dir', default='', type=str, help='imagenet, moco, simclr') 
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
