#!/bin/bash

# --- 配置 ---
export BASE_MODEL="Qwen/Qwen2.5-7B-VL-Chat"
export OUTPUT_DIR="./output/qwen_full_finetune"
export DS_CONFIG="./configs/deepspeed_config.json"

# --- GPU与并行配置 ---
NUM_GPUS=4
NUM_WORKERS=8 # 根据你的CPU核心数和内存大小调整

# --- 训练参数 ---
EPOCHS=3
BATCH_SIZE_PER_GPU=1
GRAD_ACCUM_STEPS=16
MIN_PIXELS=50000 # 示例值，根据需要调整
MAX_PIXELS=1000000 # 示例值，根据需要调整
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

deepspeed --num_gpus=${NUM_GPUS} train_grpo.py \
    --deepspeed ${DS_CONFIG} \
    --model_name_or_path ${BASE_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    ${RESUME_FLAG} \
    \
    --tuning_strategy "qlora" \
    \
    --dataset_config_path ./configs/dataset_config.json \
    --dataset_names my_dataset_1 my_dataset_2 \
    --min_pixels ${MIN_PIXELS} \
    --max_pixels ${MAX_PIXELS} \
    --train_split_ratio ${TRAIN_SPLIT_RATIO} \
    \
    --epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    \
    --learning_rate_lora 1e-6 \
    \
    --max_seq_length 2048 \
    --save_steps 500 \
    --eval_steps 250 \
    \
    --temperature 0.7 \
    --top_p 0.9 \
    \
    --eval_script_registry_path ./configs/eval_script_registry.json \
    --eval_dataset_scripts \
        "my_dataset_1:bbox_mbr_pass_rate,bbox_per_object" \
        "my_dataset_2:vqa_template" \
    \
    --use_wandb --wandb_project "qwen-vl-grpo" --wandb_run_name "grpo-run" \
    --log_to_file
