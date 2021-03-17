GPU=$1
SEED=10
TYPE=simclr
MASKROOT=./Maskall/simclr1-mask

# i=1
# echo syd ----------
# echo syd resume simclr transfer ${i}
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
    echo syd simclr transfer ${i}
    echo syd ----------
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u train_transfer.py \
    --transfer_type ${TYPE} \
    --mask_dir ${MASKROOT}/mask-${i}.pt \
    --mask_num ${i} \
    --seed ${SEED}
done