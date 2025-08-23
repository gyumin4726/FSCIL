GPUS=1
work_dir=work_dirs/vmamba_base_mambafscil_miniimagenet
bash tools/dist_train.sh configs/mini_imagenet/vmamba_base_etf_bs512_500e_miniimagenet_mambafscil.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
#RESUME_LATEST=true bash tools/dist_train.sh configs/mini_imagenet/vmamba_base_etf_bs512_500e_miniimagenet_mambafscil.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
bash tools/run_fscil.sh configs/mini_imagenet/vmamba_base_etf_bs512_500e_miniimagenet_eval_mambafscil.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic