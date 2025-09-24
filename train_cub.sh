GPUS=1
work_dir=work_dirs/CUB-200-2011
bash tools/dist_train.sh configs/cub/cub_base.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
bash tools/run_fscil.sh configs/cub/cub_inc.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic