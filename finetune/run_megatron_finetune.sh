#!/bin/bash

# 设置默认参数
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-7B-VL-Chat"
DATASET_NAMES="your_dataset_name" # 请替换为您的实际数据集名称，例如 "coco_caption"
TUNING_STRATEGY="qlora" # full, lora_llm_full_vit, vit_only, partial_llm_full_vit, qlora
OUTPUT_DIR="./output/megatron_finetune"
EPOCHS=3
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
LORA_RANK=64
LORA_ALPHA=16
LORA_DROPOUT=0.05
LOG_TO_FILE=true
WANDB_PROJECT="qwen_vl_finetune"
WANDB_RUN_NAME="megatron-qlora-$(date +%Y%m%d-%H%M%S)"

# 如果需要使用wandb，请取消注释以下行
# USE_WANDB=true

# 解析命令行参数，覆盖默认值
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpus_per_node)
        GPUS_PER_NODE="$2"
        shift # past argument
        shift # past value
        ;;
        --nnodes)
        NNODES="$2"
        shift # past argument
        shift # past value
        ;;
        --node_rank)
        NODE_RANK="$2"
        shift # past argument
        shift # past value
        ;;
        --master_addr)
        MASTER_ADDR="$2"
        shift # past argument
        shift # past value
        ;;
        --master_port)
        MASTER_PORT="$2"
        shift # past argument
        shift # past value
        ;;
        --model_name_or_path)
        MODEL_NAME_OR_PATH="$2"
        shift
        shift
        ;;
        --dataset_names)
        DATASET_NAMES="$2"
        shift
        shift
        ;;
        --tuning_strategy)
        TUNING_STRATEGY="$2"
        shift
        shift
        ;;
        --output_dir)
        OUTPUT_DIR="$2"
        shift
        shift
        ;;
        --epochs)
        EPOCHS="$2"
        shift
        shift
        ;;
        --global_batch_size)
        GLOBAL_BATCH_SIZE="$2"
        shift
        shift
        ;;
        --micro_batch_size)
        MICRO_BATCH_SIZE="$2"
        shift
        shift
        ;;
        --max_seq_length)
        MAX_SEQ_LENGTH="$2"
        shift
        shift
        ;;
        --lora_rank)
        LORA_RANK="$2"
        shift
        shift
        ;;
        --lora_alpha)
        LORA_ALPHA="$2"
        shift
        shift
        ;;
        --lora_dropout)
        LORA_DROPOUT="$2"
        shift
        shift
        ;;
        --log_to_file)
        LOG_TO_FILE=true
        shift
        ;;
        --no_log_to_file)
        LOG_TO_FILE=false
        shift
        ;;
        --use_wandb)
        USE_WANDB=true
        shift
        ;;
        --no_use_wandb)
        USE_WANDB=false
        shift
        ;;
        --wandb_project)
        WANDB_PROJECT="$2"
        shift
        shift
        ;;
        --wandb_run_name)
        WANDB_RUN_NAME="$2"
        shift
        shift
        ;;
        *)    # unknown option
        echo "未知参数: $1"
        exit 1
        ;;
    esac
done

# 确保输出目录存在
mkdir -p ${OUTPUT_DIR}

# 构建 Megatron 训练命令
CMD="python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    qwen_finetune_project/train_with_megatron.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_names ${DATASET_NAMES} \
    --tuning_strategy ${TUNING_STRATEGY} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --save ${OUTPUT_DIR} \
    --log-interval 10 \
    --eval-iters 10 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num_workers 4 \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-warmup-iters 100 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --fp16 \
    --train-iters 100000 # 这是一个占位符，实际会在脚本中根据epochs和数据集大小计算
"

# 添加日志相关参数
if [ "$LOG_TO_FILE" = true ]; then
    CMD="${CMD} --log_to_file"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="${CMD} --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${WANDB_RUN_NAME}"
fi

echo "启动训练命令:"
echo "${CMD}"

# 执行训练命令
eval ${CMD}
