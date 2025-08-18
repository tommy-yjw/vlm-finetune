#!/bin/bash

# --- 配置 ---
export BASE_MODEL="/data/oceanus_ctr/j-yanjiangwei-jk/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
export OUTPUT_DIR="./output/qwen_lora_llm_full_vit"
export DS_CONFIG="./configs/deepspeed_config.json"

# --- GPU与并行配置 ---
NUM_GPUS=1
NUM_WORKERS=16 # 根据你的CPU核心数和内存大小调整

# --- 训练参数 ---
EPOCHS=4
BATCH_SIZE_PER_GPU=1
GRAD_ACCUM_STEPS=16
MIN_PIXELS=200704 # 示例值，根据需要调整
MAX_PIXELS=4005632 # 示例值，根据需要调整
TRAIN_SPLIT_RATIO=0.9 # 训练集划分比例，0.0到1.0之间

# --- 启动命令 ---
# 检查是否需要恢复训练
RESUME_FLAG=""
if [ -n "$1" ] && [ -d "$1" ]; then
    echo "发现检查点目录，将尝试从 $1 恢复训练。"
    RESUME_FLAG="--resume_from_checkpoint $1"
else
    echo "未发现检查点目录，将从头开始训练。"
fi

deepspeed --num_gpus=${NUM_GPUS} train.py \
    --deepspeed ${DS_CONFIG} \
    --model_name_or_path ${BASE_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    ${RESUME_FLAG} \
    \
    --tuning_strategy "lora_llm_full_vit" \
    \
    --dataset_config_path ./configs/dataset_config.json \
    --dataset_names 360_aigc_layout fix_direction CGL PKU\
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --train_split_ratio ${TRAIN_SPLIT_RATIO} \
    \
    --epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    \
    --learning_rate_llm 5e-6 \
    --learning_rate_vit 1e-5 \
    --learning_rate_projector 1e-5 \
    \
    --max_seq_length 2048 \
    \
    --temperature 0.3 \
    --top_p 0.9 \
    --top_k 3 \
    \
    --use_wandb --wandb_project "qwen-vl-finetune" --wandb_run_name "full-finetune-run" \
    --log_to_file
