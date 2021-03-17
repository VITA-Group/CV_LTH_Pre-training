GPU=$1
SEED=10
TYPE=imagenet

# i=1
# echo syd ----------
# echo syd resume imagenet imp ${i}
# echo syd ----------
# CUDA_VISIBLE_DEVICES=${GPU} \
# python -u train_imp.py \
# --imp_type ${TYPE} \
# --imp_num ${i} \
# --seed ${SEED} \
# --resume_all

i=7
echo syd ----------
echo syd imagenet imp ${i}
echo syd ----------
CUDA_VISIBLE_DEVICES=${GPU} \
python -u train_imp_eval.py \
--imp_type ${TYPE} \
--imp_num ${i} \
--seed ${SEED} \
--resume_all
