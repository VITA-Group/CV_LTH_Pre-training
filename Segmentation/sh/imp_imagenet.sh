GPU=$1
SEED=10
for i in {0..18}
do
    python -u main_imp_imagenet.py \
    --model deeplabv3plus_resnet50 \
    --year 2012 \
    --crop_val \
    --gpu_id ${GPU} \
    --random_seed ${SEED} \
    --batch_size 4 \
    --imp_num ${i} \
    seed${SEED}_imp_imagenet_${i}
done