GPU=$1
i=$2
SEED=10
TYPE=imagenet
MASKROOT=./Maskall/img1-mask
# imagenet
echo syd ----------
echo syd imagenet transfer ${i}
echo syd ----------
CUDA_VISIBLE_DEVICES=${GPU} \
python -u train_transfer.py \
--transfer_type ${TYPE} \
--mask_dir ${MASKROOT}/mask-${i}.pt \
--mask_num ${i} \
--seed ${SEED}
# simclr
TYPE=simclr
MASKROOT=./Maskall/simclr1-mask
echo syd ----------
echo syd simclr transfer ${i}
echo syd ----------
CUDA_VISIBLE_DEVICES=${GPU} \
python -u train_transfer.py \
--transfer_type ${TYPE} \
--mask_dir ${MASKROOT}/mask-${i}.pt \
--mask_num ${i} \
--seed ${SEED}
# moco
TYPE=moco
MASKROOT=./Maskall/moco1-mask
echo syd ----------
echo syd moco transfer ${i}
echo syd ----------
CUDA_VISIBLE_DEVICES=${GPU} \
python -u train_transfer.py \
--transfer_type ${TYPE} \
--mask_dir ${MASKROOT}/mask-${i}.pt \
--mask_num ${i} \
--seed ${SEED}
