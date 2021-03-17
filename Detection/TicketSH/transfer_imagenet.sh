GPU=$1
SEED=10
TYPE=imagenet
MASKROOT=./Maskall/img1-mask

# i=1
# echo syd ----------
# echo syd resume imagenet transfer ${i}
# echo syd ----------
# CUDA_VISIBLE_DEVICES=${GPU} \
# python -u train_transfer.py \
# --transfer_type ${TYPE} \
# --mask_dir ${MASKROOT}/mask-${i}.pt \
# --mask_num ${i} \
# --seed ${SEED} \
# --resume_all

for i in {1..18}
do
    echo syd ----------
    echo syd imagenet transfer ${i}
    echo syd ----------
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u train_transfer.py \
    --transfer_type ${TYPE} \
    --mask_dir ${MASKROOT}/mask-${i}.pt \
    --mask_num ${i} \
    --seed ${SEED}
done