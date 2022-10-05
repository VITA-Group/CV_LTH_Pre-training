python -u main_eval_downstream.py --dataset cifar100 --arch resnet50 --save_dir cifar100_25sample_1  --dict_key state_dict  --mask_dir  /fs/vulcan-projects/sailon_root/Sonaal/fsl/CV_LTH_Pre-training/imagenet/1model_best.pth.tar  --save_model --subratio 0.25

python -u main_eval_downstream.py --dataset cifar100 --arch resnet50 --save_dir cifar100_25sample_2  --dict_key state_dict  --mask_dir  /fs/vulcan-projects/sailon_root/Sonaal/fsl/CV_LTH_Pre-training/imagenet/2model_best.pth.tar  --save_model --subratio 0.25

python -u main_eval_downstream.py --dataset cifar100 --arch resnet50 --save_dir cifar100_25sample_3  --dict_key state_dict  --mask_dir  /fs/vulcan-projects/sailon_root/Sonaal/fsl/CV_LTH_Pre-training/imagenet/3model_best.pth.tar  --save_model --subratio 0.25

python -u main_eval_downstream.py --dataset cifar100 --arch resnet50 --save_dir cifar100_50sample_1  --dict_key state_dict  --mask_dir  /fs/vulcan-projects/sailon_root/Sonaal/fsl/CV_LTH_Pre-training/imagenet/1model_best.pth.tar  --save_model --subratio 0.5

python -u main_eval_downstream.py --dataset cifar100 --arch resnet50 --save_dir cifar100_50sample_2  --dict_key state_dict  --mask_dir  /fs/vulcan-projects/sailon_root/Sonaal/fsl/CV_LTH_Pre-training/imagenet/2model_best.pth.tar  --save_model --subratio 0.5

python -u main_eval_downstream.py --dataset cifar100 --arch resnet50 --save_dir cifar100_50sample_3  --dict_key state_dict  --mask_dir  /fs/vulcan-projects/sailon_root/Sonaal/fsl/CV_LTH_Pre-training/imagenet/3model_best.pth.tar  --save_model --subratio 0.5
