GPU=$1
SEED=10

python -u main_transfer_moco.py \
--model deeplabv3plus_resnet50 \
--year 2012 \
--crop_val \
--batch_size 4 \
--gpu_id ${GPU} \
--random_seed ${SEED} \
--mask_dir ../Maskall/img2-mask/mask-1.pt \
seed${SEED}_moco_01