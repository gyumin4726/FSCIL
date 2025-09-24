GPUS=1
work_dir=work_dirs/Mini-ImageNet
bash tools/dist_train.sh configs/mini_imagenet/mini_imagenet_base.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
#RESUME_LATEST=true bash tools/dist_train.sh configs/mini_imagenet/mini_imagenet_base.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
bash tools/run_fscil.sh configs/mini_imagenet/mini_imagenet_inc.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic