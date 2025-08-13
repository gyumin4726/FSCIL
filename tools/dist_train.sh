#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((29500 + $RANDOM % 29))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# latest.pth 파일을 자동으로 찾아서 이어서 학습하는 옵션
RESUME_LATEST=${RESUME_LATEST:-false}

# latest.pth 파일 찾기 함수
find_latest_checkpoint() {
    local work_dir=""
    local args=("$@")
    
    # --work-dir 옵션에서 work_dir 추출
    for i in "${!args[@]}"; do
        if [[ ${args[i]} == --work-dir ]]; then
            if [[ $((i+1)) -lt ${#args[@]} ]]; then
                work_dir="${args[i+1]}"
                break
            fi
        elif [[ ${args[i]} == --work-dir=* ]]; then
            work_dir="${args[i]#--work-dir=}"
            break
        fi
    done
    
    echo "DEBUG: Found work_dir: $work_dir" >&2
    
    if [ -z "$work_dir" ]; then
        echo "DEBUG: No work_dir found" >&2
        echo ""
        return
    fi
    
    local latest_file=""
    
    # work_dir/fscil 디렉토리에서 latest.pth 찾기
    if [ -d "$work_dir/fscil" ]; then
        latest_file=$(find "$work_dir/fscil" -name "latest.pth" -type f 2>/dev/null | head -1)
        echo "DEBUG: Looking in $work_dir/fscil for latest.pth" >&2
    fi
    
    # work_dir에서 직접 latest.pth 찾기
    if [ -z "$latest_file" ] && [ -f "$work_dir/latest.pth" ]; then
        latest_file="$work_dir/latest.pth"
        echo "DEBUG: Found latest.pth directly in $work_dir" >&2
    fi
    
    echo "DEBUG: Final latest_file: $latest_file" >&2
    echo "$latest_file"
}

# RESUME_LATEST가 true이면 latest.pth 찾기
if [ "$RESUME_LATEST" = "true" ]; then
    echo "DEBUG: RESUME_LATEST is true, searching for latest.pth..." >&2
    echo "DEBUG: Arguments: $@" >&2
    LATEST_CKPT=$(find_latest_checkpoint "$@")
    if [ -n "$LATEST_CKPT" ]; then
        echo "Found latest checkpoint: $LATEST_CKPT"
        echo "Resuming training from latest checkpoint..."
        # --resume-from 옵션 추가
        set -- "$@" --resume-from "$LATEST_CKPT"
        echo "DEBUG: Added --resume-from $LATEST_CKPT" >&2
    else
        echo "Warning: RESUME_LATEST=true but no latest.pth found"
        echo "Proceeding without checkpoint..."
    fi
fi

if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=$NNODES \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
else
  echo "Using launch mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
fi