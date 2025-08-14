#!/bin/bash

# --- 配置 ---
export BASE_MODEL="Qwen/Qwen2.5-7B-VL-Chat"
export OUTPUT_DIR="./output/qwen_qlora_multi_dataset"
export DS_CONFIG="./configs/deepspeed_config.json"
export DATASET1_JSON="./data/dataset1/data.json"
export DATASET1_IMG_ROOT="./data/dataset1/images"
export DATASET2_JSON="./data/dataset2/data.json"
export DATASET2_IMG_ROOT="./data/dataset2/images"

# --- GPU与并行配置 ---
NUM_GPUS=2
NUM_WORKERS=4 # 根据你的CPU核心数和内存大小调整

# --- 训练参数 ---
EPOCHS=5
BATCH_SIZE_PER_GPU=4
GRAD_ACCUM_STEPS=4

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
    --tuning_strategy "qlora" \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    \
    --datasets ${DATASET1_JSON} ${DATASET1_IMG_ROOT} \
    --datasets ${DATASET2_JSON} ${DATASET2_IMG_ROOT} \
    --use_data_augmentation \
    --min_pixels 100000 \
    --max_pixels 4000000 \
    \
    --epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --num_workers ${NUM_WORKERS} \
    \
    --learning_rate_llm 2e-4 \
    --learning_rate_vit 1e-4 \
    --learning_rate_projector 1e-4 \
    \
    --max_seq_length 2048 \
    --save_steps 500 \
    --eval_steps 250 \
    \
    --use_wandb --wandb_project "qwen-vl-finetune" --wandb_run_name "qlora-multi-dataset-run" \
    --log_to_file