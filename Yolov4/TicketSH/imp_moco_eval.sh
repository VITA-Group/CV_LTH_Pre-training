GPU=$1
SEED=10
TYPE=moco

i=5
echo syd ----------
echo syd moco imp ${i}
echo syd ----------
CUDA_VISIBLE_DEVICES=${GPU} \
python -u train_imp_eval.py \
--imp_type ${TYPE} \
--imp_num ${i} \
--seed ${SEED} \
--resume_all