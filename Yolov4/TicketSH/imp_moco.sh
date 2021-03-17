GPU=$1
SEED=10
TYPE=moco

# i=1
# echo syd ----------
# echo syd resume moco imp ${i}
# echo syd ----------
# CUDA_VISIBLE_DEVICES=${GPU} \
# python -u train_imp.py \
# --imp_type ${TYPE} \
# --imp_num ${i} \
# --seed ${SEED} \
# --resume_all

for i in {0..18}
do
    echo syd ----------
    echo syd moco imp ${i}
    echo syd ----------
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u train_imp.py \
    --imp_type ${TYPE} \
    --imp_num ${i} \
    --seed ${SEED}
done