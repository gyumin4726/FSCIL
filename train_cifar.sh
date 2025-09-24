GPUS=1
work_dir=work_dirs/CIFAR-100
bash tools/dist_train.sh configs/cifar/cifar_base.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
#RESUME_LATEST=true bash tools/dist_train.sh configs/cifar/cifar_base.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
bash tools/run_fscil.sh configs/cifar/cifar_inc.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic