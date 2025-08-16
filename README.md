# Qwen2.5-VL 多功能微调框架

本项目是一个为 Qwen2.5-VL 多模态大模型设计的多功能微调框架。它集成了多种主流的微调策略，并提供从数据处理、模型训练、模型评估到最终推理的全流程支持。

## ✨ 功能特性

- **多种微调策略**：支持全参数微调、LoRA、QLoRA，并允许仅微调视觉（ViT）或语言模型（LLM）部分。
- **高级训练算法**：集成 GRPO (Group Reward Policy Optimization) 算法，可通过自定义奖励函数进行策略优化。
- **分布式训练**：深度集成 DeepSpeed (ZeRO-2/3)，支持多机多卡高效训练。
- **多数据集支持**：可同时加载和训练多个不同来源的数据集。
- **灵活的评估**：支持训练过程中和训练后进行评估，并可轻松扩展自定义评估脚本。
- **详细日志**：支持 Weights & Biases (wandb) 和本地日志记录，方便监控训练过程。

## 📂 目录结构

- `train.py`: SFT（监督微调）主训练脚本。
- `train_grpo.py`: GRPO 算法训练脚本。
- `inference.py`: 推理脚本，用于与微调后的模型进行交互。
- `evaluation/`: 评估脚本目录。
  - `eval_sft.py`: SFT 模型的评估脚本。
  - `eval_vqa_template.py`: 自定义评估逻辑的模板。
- `datasets/`: 数据集处理相关脚本。
- `configs/`: 配置文件目录（如 DeepSpeed 配置）。
- `run_*.sh`: 各种微调策略的启动脚本。
- `data/`: 示例数据集目录结构。
- `output/`: 训练输出目录，存放模型权重和日志。

## 🚀 快速开始

### 1. 环境准备

首先，克隆项目并安装所需的依赖库：

```bash
git clone <your-repo-url>
cd qwen_finetune_project
pip install -r requirements.txt
```

### 2. 数据准备

将您的数据集按以下结构存放：

```
data/
└── your_dataset_name/
    ├── data.json       # 标注文件
    └── images/         # 图片文件夹
        ├── image1.jpg
        └── image2.png
```

`data.json` 应为包含对话和图片路径的 JSON 格式。

### 3. 开始训练

根据您的需求选择一个启动脚本，例如，使用 QLoRA 进行微调：

```bash
bash run_qlora_advanced.sh
```

在运行前，请根据实际情况修改脚本内的 `BASE_MODEL_PATH`、`DATA_PATH` 和其他训练参数。

### 4. 模型评估

训练完成后，您可以使用评估脚本来测试模型性能。例如，评估一个 SFT 模型：

```bash
bash run_eval_sft.sh ./output/your_model_checkpoint
```

### 5. 模型推理

使用 `inference.py` 脚本与您的模型进行交互：

```bash
python inference.py \
  --model_path ./output/your_model_checkpoint/final_model \
  --image_path ./path/to/your/image.jpg \
  --prompt "描述一下这张图片"
```

## 📚 详细指南

本项目提供了更详细的微调指南，帮助您深入了解不同训练策略的配置和原理：

- **[SFT 微调指南](./SFT_tuning_guide.md)**：涵盖了全参数、LoRA、QLoRA 等多种微调方法的详细说明。
- **[GRPO 训练指南](./GRPO_tuning_guide.md)**：介绍了如何使用 GRPO 算法和自定义奖励函数来优化模型。

## 🛠️ 脚本说明

- `run_full_finetune.sh`: 全参数微调。
- `run_lora_llm_full_vit.sh`: LoRA 微调 LLM，全参数微调 ViT。
- `run_qlora_advanced.sh`: 使用 QLoRA 进行高效微调。
- `run_qlora_multi_dataset.sh`: 使用 QLoRA 同时在多个数据集上微调。
- `run_grpo.sh`: 启动 GRPO 训练。
- `run_eval_sft.sh`: 运行 SFT 模型评估。

---

如有任何问题，欢迎提交 issue 或参与讨论。
