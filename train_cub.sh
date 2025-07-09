#!/bin/bash

# Enhanced Mamba-FSCIL training with improved skip connections
echo "=== Enhanced Mamba-FSCIL Training Pipeline ==="
echo "New Features:"
echo "  ✓ Multi-scale skip connections from ResNet layers"
echo "  ✓ Attention-weighted skip connection fusion"
echo "  ✓ Progressive feature fusion"
echo "  ✓ Backward compatible with existing pipeline"
echo ""

GPUS=1
work_dir=work_dirs/mamba_fscil/cub_resnet18_mambafscil

echo "Stage 1: Training base session with enhanced skip connections..."
bash tools/dist_train.sh configs/cub/resnet18_etf_bs512_80e_cub_mambafscil.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic

echo ""
echo "Stage 2: Evaluating incremental learning with enhanced skip connections..."
bash tools/run_fscil.sh configs/cub/resnet18_etf_bs512_80e_cub_eval_mambafscil.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic

echo ""
echo "=== Training Complete ==="
echo "Enhanced features automatically enabled in existing configs!"
