#!/bin/bash

# --- 配置 ---
export BASE_MODEL="Qwen/Qwen2.5-7B-VL-Chat"
export OUTPUT_DIR="./output/qwen_qlora_advanced_run"
export DS_CONFIG="./configs/deepspeed_config.json"
export DATASET1_JSON="./data/dataset1/data.json"
export DATASET1_IMG_ROOT="./data/dataset1/images"
export RESUME_CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-500" # 示例：从step 500恢复

# --- GPU与并行配置 ---
NUM_GPUS=4
NUM_WORKERS=8 # 根据你的CPU核心数和内存大小调整

# --- 训练参数 ---
EPOCHS=5
BATCH_SIZE_PER_GPU=2
GRAD_ACCUM_STEPS=8

# --- 启动命令 ---
# 检查是否需要恢复训练
RESUME_FLAG=""
if [ -d "$RESUME_CHECKPOINT_PATH" ]; then
    echo "发现检查点目录，将尝试从 $RESUME_CHECKPOINT_PATH 恢复训练。"
    RESUME_FLAG="--resume_from_checkpoint ${RESUME_CHECKPOINT_PATH}"
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
    --lora_rank 128 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    \
    --datasets ${DATASET1_JSON} ${DATASET1_IMG_ROOT} \
    --use_data_augmentation \
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
    --use_wandb --wandb_project "qwen-vl-advanced" --wandb_run_name "qlora-aug-run" \
    --log_to_file
