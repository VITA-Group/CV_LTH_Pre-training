GPU=$1
SEED=10
TYPE=moco
for i in {3..4}
do
    echo syd ----------
    echo syd moco random ${i}
    echo syd ----------
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u train_random.py \
    --imp_type ${TYPE} \
    --imp_num ${i} \
    --seed ${SEED}
done