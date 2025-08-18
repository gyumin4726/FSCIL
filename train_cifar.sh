GPUS=1
work_dir=work_dirs/vmamba_base_mambafscil_cifar
# latest.pth를 자동으로 찾아서 base 세션에서 이어서 학습
RESUME_LATEST=true bash tools/dist_train.sh configs/cifar/vmamba_base_etf_bs512_200e_cifar_mambafscil.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
# latest.pth를 자동으로 찾아서 incremental 세션에서 이어서 학습
RESUME_LATEST=true bash tools/run_fscil.sh configs/cifar/vmamba_base_etf_bs512_200e_cifar_eval_mambafscil.py ${work_dir} "" $GPUS --seed 0 --deterministic 