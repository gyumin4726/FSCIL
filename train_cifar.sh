GPUS=1
work_dir=work_dirs/vmamba_base_mambafscil_cifar
bash tools/dist_train.sh configs/cifar/vmamba_base_etf_bs512_200e_cifar_mambafscil.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
#RESUME_LATEST=true bash tools/dist_train.sh configs/cifar/vmamba_base_etf_bs512_200e_cifar_mambafscil.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
bash tools/run_fscil.sh configs/cifar/vmamba_base_etf_bs512_200e_cifar_eval_mambafscil.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic