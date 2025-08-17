# GRPO 调参攻略

本文档为 `train_grpo.py` 脚本提供了一份详细的调参指南，旨在帮助您根据不同的任务和硬件配置，高效地进行 GRPO (Generative Reward Post-Optimization) 训练。

## 目录

1.  [核心概念](#核心概念)
2.  [参数详解](#参数详解)
    *   [模型与路径 (Model & Paths)](#模型与路径)
    *   [数据集 (Datasets)](#数据集)
    *   [GRPO 核心超参数 (GRPO Hyperparams)](#grpo-核心超参数)
    *   [KL 散度惩罚 (KL Penalty)](#kl-散度惩罚)
    *   [微调策略 (Tuning Strategy)](#微调策略)
    *   [日志与保存 (Logging & Saving)](#日志与保存)
    *   [DeepSpeed 配置](#deepspeed-配置)
    *   [vLLM Rollout 配置](#vllm-rollout-配置)
3.  [调参建议](#调参建议)
    *   [显存优化](#显存优化)
    *   [训练稳定性](#训练稳定性)
    *   [效果优化](#效果优化)
4.  [启动命令示例](#启动命令示例)

---

## 核心概念

GRPO 是一种基于强化学习的对齐算法，其核心思想是：

1.  **Rollout (采样)**: 对于每个训练样本（prompt + image），使用当前模型生成 `k` 个候选回复 (responses)。
2.  **Reward (奖励)**: 使用一个外部的奖励函数 (Reward Function) 为这 `k` 个候选回复打分。这个奖励函数需要您根据具体任务目标来定义，例如，可以是基于 VQA 的准确率、目标检测的 IoU，或是美学评分等。
3.  **Optimization (优化)**: 根据奖励分数，构建 GRPO 损失函数，该损失函数会鼓励模型生成更高奖励的回复，同时惩罚低奖励的回复。通过反向传播更新模型参数。

与 DPO 等偏好学习算法不同，GRPO 直接使用绝对的奖励分数进行优化，更适合那些能够量化评估生成质量的场景。

---

## 参数详解

### 模型与路径

*   `--model_name_or_path`
    *   **作用**: 指定基础模型的路径或 Hugging Face Hub 上的模型名称。
    *   **建议**: 通常使用 `Qwen/Qwen2.5-7B-VL-Chat` 或您已经进行过 SFT 的模型路径。

*   `--output_dir`
    *   **作用**: 指定训练过程中模型权重、日志和其他输出文件的保存目录。
    *   **建议**: 为每次实验设置一个独立的、有意义的目录名称。

*   `--resume_from_checkpoint`
    *   **作用**: 从指定的 checkpoint 目录恢复训练，而不是从头开始。
    *   **建议**: 在训练意外中断后使用，确保路径指向一个有效的 DeepSpeed checkpoint 目录（例如 `output_dir/step-100`）。

### 数据集

*   `--dataset_config_path`
    *   **作用**: 指向数据集注册的 JSON 配置文件路径。
    *   **建议**: 保持默认值 `./configs/dataset_config.json` 即可，除非您修改了其位置。

*   `--dataset_names`
    *   **作用**: 指定要用于训练的一个或多个数据集的名称（在配置文件中注册的名称）。
    *   **建议**: 根据您的任务选择一个或多个相关的数据集。

*   `--train_split_ratio`
    *   **作用**: 设定训练集和验证集的划分比例。
    *   **建议**: 通常设置为 `0.9` 或 `0.95`。如果没有验证需求，可以设置为 `1.0`。

### GRPO 核心超参数

*   `--epochs`
    *   **作用**: 训练的总轮数。
    *   **建议**: RL 训练通常对数据更敏感，建议从 `1-3` 个 epoch 开始尝试。

*   `--per_device_train_batch_size`
    *   **作用**: 每个 GPU 上的批处理大小。
    *   **建议**: **关键显存参数**。在不超出显存的前提下，尽可能设置为最大。通常从 `1` 或 `2` 开始。

*   `--gradient_accumulation_steps`
    *   **作用**: 梯度累积步数。`global_batch_size = per_device_train_batch_size * num_gpus * gradient_accumulation_steps`。
    *   **建议**: 如果因为显存限制无法增大 `per_device_train_batch_size`，可以通过增大此参数来模拟更大的全局批量，从而稳定训练。通常设置为 `4`, `8`, `16`。

*   `--learning_rate_lora`
    *   **作用**: LoRA 层的学习率。
    *   **建议**: RL 训练的学习率通常比 SFT 更小，以避免模型遗忘通用能力。建议从 `1e-6` 到 `5e-6` 开始尝试。

*   `--k_samples`
    *   **作用**: 每个 prompt 生成的候选样本数量。
    *   **建议**: `k` 值越大，探索的空间越广，但显存和计算开销也越大。通常设置为 `4` 或 `8`。如果显存紧张，可以适当减小到 `2`。

*   `--rollout_max_new_tokens`
    *   **作用**: 在 Rollout 阶段生成回复的最大长度。
    *   **建议**: 根据您的任务回复的典型长度来设置。设置过长会增加计算开销。

*   `--temperature` & `--top_p`
    *   **作用**: 控制采样过程中的随机性。`temperature` 越高，随机性越强；`top_p` 越小，生成结果越集中于高概率词。
    *   **建议**: 推荐使用 `temperature=0.7`, `top_p=0.9` 的组合来增加生成的多样性，为奖励模型提供更丰富的样本。

### KL 散度惩罚

*   `--use_kl_penalty`
    *   **作用**: 是否启用 KL 散度惩罚。该惩罚项用于约束训练后的模型与原始参考模型之间的分布差异，防止模型在优化奖励的同时“走偏”，丧失通用能力。
    *   **建议**: **强烈建议开启**，特别是当奖励函数比较“刁钻”或训练数据较少时，可以有效防止“奖励过拟合” (reward hacking)。

*   `--kl_beta`
    *   **作用**: KL 散度惩罚项的系数。
    *   **建议**: 这是一个需要调试的关键参数。通常从 `0.1` 或 `0.05` 开始。如果发现模型生成能力下降（例如，回复变得不流利），可以适当减小 `kl_beta`；如果奖励提升不明显，可以适当增大。

### 微调策略

*   `--tuning_strategy`
    *   **作用**: 选择微调方式：`qlora` (推荐), `lora`, `full`。
    *   **建议**: 对于大多数用户，**`qlora`** 是最佳选择，它在效果和显存效率之间取得了很好的平衡。

*   `--lora_rank`, `--lora_alpha`, `--lora_dropout`
    *   **作用**: LoRA 的相关参数。
    *   **建议**: 使用 `lora_rank=64`, `lora_alpha=16`, `lora_dropout=0.05` 的默认组合通常效果不错。`lora_alpha` 通常设置为 `lora_rank` 的 1/4 或 1/2。

### 日志与保存

*   `--save_steps` & `--eval_steps`
    *   **作用**: 每隔多少步保存一次 checkpoint 和运行一次评估。
    *   **建议**: 根据您的数据集大小和训练时长来设置。例如，每 `100` 或 `200` 步保存/评估一次。

*   `--use_wandb`
    *   **作用**: 使用 `wandb` 来记录和可视化训练过程。
    *   **建议**: **强烈建议开启**，便于监控损失、奖励、KL 散度等关键指标的变化。

### DeepSpeed 配置

*   `--deepspeed`
    *   **作用**: 指定 DeepSpeed 的配置文件路径。
    *   **建议**: 根据您的微调策略选择合适的配置文件，例如 `zero2.json` 或 `zero3.json`。对于 QLoRA，通常使用 `zero2.json`。

### vLLM Rollout 配置

*   `--use_vllm_rollout`
    *   **作用**: 使用 vLLM 进行 Rollout 采样。vLLM 是一个高效的推理引擎，可以**显著加速**采样过程，从而提升训练的整体速度。
    *   **建议**: 如果您的环境已安装 vLLM 并且硬件支持，**强烈建议开启**。

*   `--vllm_tp`
    *   **作用**: vLLM 的张量并行度（Tensor Parallelism），即使用多少个 GPU 来共同运行 vLLM 推理。
    *   **建议**: 根据您的 GPU 数量设置，通常设置为 `1`（单卡）或 `2`、`4`（多卡）。

---

## 调参建议

### 显存优化

如果遇到 OOM (Out of Memory) 错误，请按以下顺序调整：

1.  减小 `--per_device_train_batch_size` (例如，减为 `1`)。
2.  减小 `--lora_rank` (例如，从 `64` 减为 `32`)。
3.  减小 `--k_samples` (例如，从 `4` 减为 `2`)。
4.  减小 `--max_seq_length`。
5.  使用 DeepSpeed ZeRO Stage 3 (`zero3.json`)。

### 训练稳定性

*   **监控奖励 (Reward)**: 确保奖励值的均值在训练过程中是上升趋势。如果奖励停滞或下降，可能需要调整学习率或奖励函数本身。
*   **监控 KL 散度**: 如果开启了 KL 惩罚，监控 `kl_div` 指标。如果该值持续增大，说明模型正在偏离原始模型，可能需要增大 `--kl_beta`。
*   **使用较小的学习率**: RL 训练对学习率敏感，`1e-6` 是一个安全的起点。

### 效果优化

1.  **高质量的奖励函数**: 这是 GRPO 成功的关键。奖励函数需要准确、稳定地反映您的优化目标。在投入大量计算资源进行 GRPO 训练前，请充分测试您的奖励函数。
2.  **平衡探索与利用**: 调整 `--temperature` 和 `--top_p` 来控制生成样本的多样性。多样性太低可能导致模型陷入局部最优；多样性太高可能导致训练不稳定。
3.  **合适的 KL 约束**: `--kl_beta` 是平衡“学习新知识”和“不忘记旧知识”的关键。多组实验找到最佳值。
4.  **从 SFT 模型开始**: 通常，先对模型进行 SFT (Supervised Fine-Tuning)，使其具备基本的能力，然后再进行 GRPO 训练，效果会更好。

---

## 启动命令示例

这是一个使用 QLoRA、vLLM Rollout 和 KL 惩罚的典型启动脚本 (`run_grpo.sh`) 示例：

```bash
#!/bin/bash

# DeepSpeed 配置
DEEPSPEED_CONFIG=./configs/zero2.json
# 模型路径 (可以是基础模型或 SFT 后的模型)
MODEL_PATH="Qwen/Qwen2.5-7B-VL-Chat"
# 输出目录
OUTPUT_DIR="./output/grpo_qwen_vl_run_1"

# vLLM 使用的 GPU 数量
VLLM_TP=1 

# NCCL & CUDA 配置
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 启动命令
deepspeed --master_port 12345 train_grpo.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --dataset_config_path ./configs/dataset_config.json \
    --dataset_names "your_custom_dataset" \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate_lora 2e-6 \
    --max_seq_length 2048 \
    --k_samples 4 \
    --rollout_max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --tuning_strategy "qlora" \
    --lora_rank 64 \
    --lora_alpha 16 \
    --use_kl_penalty \
    --kl_beta 0.1 \
    --save_steps 100 \
    --eval_steps 100 \
    --use_vllm_rollout \
    --vllm_tp ${VLLM_TP} \
    --use_wandb \
    --wandb_project "grpo_tuning" \
    --log_to_file
```

希望这份攻略能帮助您顺利完成 GRPO 训练！
