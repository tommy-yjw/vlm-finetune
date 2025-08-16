# SFT (Supervised Fine-Tuning) 调参攻略

本文档为 `train.py` 脚本提供了一份详细的监督微调 (SFT) 指南，旨在帮助您根据不同的任务需求、硬件配置和微调策略，高效地训练 Qwen-VL 模型。

## 目录

1.  [核心概念：SFT 与不同微调策略](#核心概念)
2.  [参数详解](#参数详解)
    *   [模型与路径](#模型与路径)
    *   [数据集配置](#数据集配置)
    *   [训练超参数](#训练超参数)
    *   [微调策略 (Tuning Strategy)](#微调策略)
    *   [日志、保存与评估](#日志保存与评估)
    *   [DeepSpeed 配置](#deepspeed-配置)
3.  [调参建议](#调参建议)
    *   [显存优化](#显存优化)
    *   [选择合适的微调策略](#选择合适的微调策略)
    *   [学习率设置](#学习率设置)
4.  [启动命令示例](#启动命令示例)
    *   [示例 1: QLoRA (推荐)](#示例-1-qlora-推荐)
    *   [示例 2: LoRA LLM + Full ViT](#示例-2-lora-llm--full-vit)
    *   [示例 3: Full Fine-Tuning (全参数微调)](#示例-3-full-fine-tuning-全参数微调)

---

## 核心概念

SFT (Supervised Fine-Tuning) 是指使用成对的 “(输入, 理想输出)” 数据来微调预训练模型，使其适应特定任务的格式和要求。对于多模态模型 Qwen-VL，输入通常是 “(图片, 指令文本)”，输出是模型应生成的文本回复。

`train.py` 脚本支持多种微调策略，让您可以根据计算资源和任务需求，灵活地选择要训练的模型部分：

*   **QLoRA**: **（推荐）** 对 LLM 部分使用 4-bit 量化和 LoRA 微调，同时可以全参数微调 ViT。这是**显存效率最高**的策略。
*   **LoRA LLM + Full ViT**: 对 LLM 部分使用 LoRA 微调，同时全参数微调 ViT 和 Projector。在性能和效率之间取得了很好的平衡。
*   **Partial LLM + Full ViT**: 冻结 LLM 的底层部分，只微调高层部分、ViT 和 Projector。适合在保持模型通用能力的同时，注入特定知识。
*   **ViT Only**: 只微调 ViT 和 Projector，完全冻结 LLM。适合纯粹的视觉能力提升任务。
*   **Full Fine-Tuning**: 微调模型的所有参数。效果最好，但需要巨大的计算资源。

---

## 参数详解

### 模型与路径

*   `--model_name_or_path`: 基础模型路径，例如 `Qwen/Qwen2.5-7B-VL-Chat`。
*   `--output_dir`: 训练输出（ checkpoints, logs）的保存目录。
*   `--resume_from_checkpoint`: 从中断的 checkpoint 继续训练。

### 数据集配置

*   `--dataset_config_path`: 数据集注册的 JSON 配置文件。
*   `--dataset_names`: 要使用的已注册数据集名称列表。
*   `--train_split_ratio`: 训练集/验证集划分比例，默认为 `0.9`。
*   `--use_data_augmentation`: 是否对训练图片启用视觉数据增强（例如随机翻转、裁剪等）。可以提升模型的视觉鲁棒性。

### 训练超参数

*   `--epochs`: 训练轮数。通常 SFT 设置为 `1-3` 个 epoch 即可。
*   `--per_device_train_batch_size`: **关键显存参数**。每个 GPU 的批处理大小，通常从 `1` 或 `2` 开始尝试。
*   `--gradient_accumulation_steps`: 梯度累积步数。用于在不增加显存的情况下模拟更大的全局批量。
*   `--learning_rate_llm`, `--learning_rate_vit`, `--learning_rate_projector`: 分别为模型的三个主要部分（语言模型、视觉编码器、投影层）设置不同的学习率。这是进行精细化调优的关键。
*   `--weight_decay`: 权重衰减，一种正则化手段，防止过拟合。
*   `--warmup_steps`: 学习率预热步数，有助于训练初期的稳定。
*   `--max_seq_length`: 输入序列的最大长度。越长显存消耗越大。

### 微调策略

*   `--tuning_strategy`: **核心参数**，选择上述五种微调策略之一。
*   `--llm_frozen_layers`: 配合 `partial_llm_full_vit` 策略使用，指定要冻结的 LLM 层数（从第 0 层开始）。
*   `--lora_rank`, `--lora_alpha`, `--lora_dropout`: LoRA/QLoRA 的相关参数。
    *   `lora_rank`: LoRA 矩阵的秩，值越大，可调参数越多，拟合能力越强，但显存消耗也越大。常见值为 `32`, `64`, `128`。
    *   `lora_alpha`: LoRA 的缩放因子。通常设置为 `rank` 的 1/4 或 1/2 (例如 `rank=64, alpha=16`)。
    *   `lora_dropout`: LoRA 层的 dropout 概率。

### 日志、保存与评估

*   `--save_steps`, `--eval_steps`: 每隔多少步保存 checkpoint 和运行评估。
*   `--eval_dataset_scripts`: 指定用于评估的脚本和数据集。格式为 `'dataset_name:script_name1,script_name2'`。
*   `--use_wandb`, `--wandb_project`, `--wandb_run_name`: **强烈建议开启** WandB 来可视化和追踪训练过程。

### DeepSpeed 配置

*   `--deepspeed`: 指定 DeepSpeed 配置文件路径。
    *   对于 QLoRA/LoRA，通常使用 `zero2.json`。
    *   对于 Full Fine-Tuning，为了最大化显存优化，建议使用 `zero3.json`。

---

## 调参建议

### 显存优化

遇到 OOM (Out of Memory) 错误时，按以下顺序调整：

1.  **切换到 `qlora` 策略**：这是最有效的节省显存的方法。
2.  减小 `--per_device_train_batch_size` (例如，减为 `1`)。
3.  增大 `--gradient_accumulation_steps` 以补偿 batch size 的减小。
4.  减小 `--lora_rank` (例如，从 `64` 减为 `32`)。
5.  减小 `--max_seq_length`。
6.  如果进行全参数微调，确保使用 DeepSpeed ZeRO Stage 3 (`zero3.json`)。

### 选择合适的微调策略

*   **资源有限/快速迭代**: **首选 `qlora`**。它以极低的资源消耗，通常能达到全量微调 90% 以上的效果。
*   **提升视觉理解**: 如果任务侧重于更精细的视觉理解（如小目标检测、OCR），选择 `lora_llm_full_vit` 或 `partial_llm_full_vit`，确保 ViT 被充分训练。
*   **注入领域知识**: 如果任务需要模型学习大量新的文本知识，`partial_llm_full_vit` 或 `full` 策略更合适。
*   **追求最佳性能且资源充足**: 使用 `full` (全参数微调)。

### 学习率设置

*   **基本原则**: 越接近模型输入的模块（ViT），学习率可以设置得越高；越接近模型输出的模块（LLM），学习率应设置得越低，以防止灾难性遗忘。
*   **推荐范围**:
    *   `--learning_rate_vit`: `1e-4` to `2e-4`
    *   `--learning_rate_projector`: `1e-4` to `2e-4`
    *   `--learning_rate_llm`:
        *   对于 LoRA/QLoRA: `1e-4` to `2e-4`
        *   对于 Full/Partial: `1e-5` to `5e-5` (全量微调时学习率要更小)

---

## 启动命令示例

### 示例 1: QLoRA (推荐)

这是最常用、资源最友好的配置。

```bash
#!/bin/bash

DEEPSPEED_CONFIG=./configs/zero2.json
MODEL_PATH="Qwen/Qwen2.5-7B-VL-Chat"
OUTPUT_DIR="./output/sft_qwen_vl_qlora"

deepspeed --master_port 12346 train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --tuning_strategy "qlora" \
    --dataset_names "your_sft_dataset" \
    --epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate_llm 1e-4 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --max_seq_length 2048 \
    --save_steps 200 \
    --eval_steps 200 \
    --use_wandb
```

### 示例 2: LoRA LLM + Full ViT

当需要重点提升视觉能力时，这是一个很好的选择。

```bash
#!/bin/bash

DEEPSPEED_CONFIG=./configs/zero2.json
MODEL_PATH="Qwen/Qwen2.5-7B-VL-Chat"
OUTPUT_DIR="./output/sft_qwen_vl_lora_llm_full_vit"

deepspeed --master_port 12346 train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --tuning_strategy "lora_llm_full_vit" \
    --dataset_names "your_sft_dataset" \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate_llm 1e-4 \
    --learning_rate_vit 2e-4 \
    --learning_rate_projector 2e-4 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --max_seq_length 2048 \
    --save_steps 200 \
    --eval_steps 200 \
    --use_wandb
```

### 示例 3: Full Fine-Tuning (全参数微调)

需要大量显存（例如 4-8 张 A100/H100 80G），但效果最好。

```bash
#!/bin/bash

DEEPSPEED_CONFIG=./configs/zero3.json # 使用 ZeRO-3
MODEL_PATH="Qwen/Qwen2.5-7B-VL-Chat"
OUTPUT_DIR="./output/sft_qwen_vl_full"

deepspeed --master_port 12346 train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --tuning_strategy "full" \
    --dataset_names "your_sft_dataset" \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate_llm 2e-5 \
    --learning_rate_vit 1e-4 \
    --learning_rate_projector 1e-4 \
    --max_seq_length 2048 \
    --save_steps 200 \
    --eval_steps 200 \
    --use_wandb
