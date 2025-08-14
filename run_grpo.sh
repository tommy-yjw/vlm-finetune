#!/bin/bash

#===================================================================================================
# GRPO (Group Relative Policy Optimization) 训练脚本
#
# 该脚本用于在SFT训练后的LoRA/QLoRA适配器上进行GRPO训练，以进一步优化模型的特定能力。
#===================================================================================================

# --- 1. 环境变量和路径配置 ---

# 模型路径 (Hugging Face Hub 或本地路径)
# 确保这是您完成SFT训练的基础模型
MODEL_PATH="Qwen/Qwen1.5-7B-Chat"

# SFT训练后得到的LoRA适配器路径
# 这是GRPO训练的起点
LORA_ADAPTER_PATH="./output/qwen1.5-7b-chat-qlora/sft_checkpoint-300"

# 包含奖励函数的Python脚本路径
REWARD_FUNCTION_PATH="./reward_function.py"

# 训练输出目录
OUTPUT_DIR="./output/qwen1.5-7b-chat-grpo"

# DeepSpeed配置文件路径
DS_CONFIG_PATH="./configs/deepspeed_config.json"

# 数据集配置文件路径
# GRPO通常使用与SFT阶段不同的、更侧重于特定能力的指令数据集
DATASET_CONFIG_PATH="./data/dataset1/grpo_dataset.json"

# --- 2. 训练超参数 ---

# GRPO特定参数
K_SAMPLES=4                # 每个prompt生成多少个候选回复
GRPO_BETA=1.0              # GRPO损失的温度系数
USE_KL_PENALTY=true        # 是否使用KL散度惩罚来稳定训练
KL_BETA=0.1                # KL散度惩罚的系数

# 通用训练参数
EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-5
SEQ_LENGTH=1024
SAVE_STEPS=100
LOG_STEPS=10

# --- 3. DeepSpeed启动命令 ---

# 使用deepspeed启动训练
# --master_port 随机指定一个未被占用的端口
deepspeed --include localhost:0,1 --master_port 29501 train_grpo.py \
    --model_name_or_path $MODEL_PATH \
    --lora_adapter_path $LORA_ADAPTER_PATH \
    --reward_function_path $REWARD_FUNCTION_PATH \
    --data_config_path $DATASET_CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --deepspeed $DS_CONFIG_PATH \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $SEQ_LENGTH \
    --save_steps $SAVE_STEPS \
    --log_steps $LOG_STEPS \
    --k_samples $K_SAMPLES \
    --grpo_beta $GRPO_BETA \
    --use_kl_penalty $USE_KL_PENALTY \
    --kl_beta $KL_BETA \
    --tuning_strategy 'qlora' \
    --bf16

# --- 脚本说明 ---
#
# 1.  **环境准备**:
#     - 确保已安装所有依赖 (`pip install -r requirements.txt`)。
#     - 确保 `deepspeed` 已正确配置。
#
# 2.  **路径配置**:
#     - `MODEL_PATH`: 基础大模型的路径。
#     - `LORA_ADAPTER_PATH`: **必须**提供一个经过SFT微调的LoRA/QLoRA适配器路径。
#     - `REWARD_MODEL_PATH`: 指向您实现了`get_reward_function`的脚本。
#     - `DATASET_CONFIG_PATH`: GRPO训练所用的数据集配置文件。
#
# 3.  **GRPO参数**:
#     - `K_SAMPLES`: 控制每个prompt生成多少个响应进行比较，是GRPO的核心。增加此值会提高样本多样性，但也会显著增加计算开销。
#     - `GRPO_BETA`: 控制策略更新的强度。值越高，策略更新越激进。
#     - `USE_KL_PENALTY` & `KL_BETA`: 强烈建议开启KL惩罚，以防止模型在优化过程中遗忘SFT阶段学到的知识。`KL_BETA`控制惩罚强度。
#
# 4.  **运行**:
#     - 在终端中直接运行 `./run_grpo.sh`。
#     - 根据您的GPU数量修改 `deepspeed --include` 参数。