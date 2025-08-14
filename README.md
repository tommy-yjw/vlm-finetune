# Qwen2.5-VL 微调项目

本项目用于对 Qwen2.5-VL 多模态大模型进行多种微调策略（全参数、LoRA、QLoRA、视觉部分、部分 LLM 等），支持多数据集、DeepSpeed 加速、wandb 日志和自定义评估。

## 目录结构
- `train.py`：主训练脚本，支持多种微调策略和参数配置。
- `train_grpo.py`：GRPO 算法训练脚本，用于基于奖励函数进行模型优化。
- `inference.py`：推理脚本，加载微调后的模型进行图片问答。
- `datasets/custom_dataset.py`：自定义数据集类，支持像素过滤和多图数据处理。
- `evaluation/eval_vqa_template.py`：评估模板，可扩展自定义评估逻辑。
- `configs/deepspeed_config.json`：DeepSpeed 配置文件。
- `core/grpo_utils.py`：GRPO 算法核心工具函数，包含损失计算和 log_probs 计算。
- `reward_function.py`：奖励函数定义脚本，可自定义奖励逻辑。
- `run_full_finetune.sh`、`run_lora_llm_full_vit.sh`、`run_partial_llm_full_vit.sh`、`run_qlora_advanced.sh`、`run_qlora_multi_dataset.sh`、`run_vit_only.sh`、`run_qlora_windows.bat`：多种微调策略的启动脚本。
- `run_grpo.sh`：GRPO 训练启动脚本。
- `requirements.txt`：依赖库列表。
- `output/`：模型输出和日志目录。
- `data/`：数据集目录，需包含 `data.json` 和图片文件夹。

## 安装依赖
```bash
pip install -r requirements.txt
```

## 数据准备
- 数据集需为 JSON 格式，包含图片路径和对话内容，图片存放于指定文件夹。
- 示例：`data/dataset1/data.json`、`data/dataset1/images/`

## 训练方法
- **常规微调**：选择合适的脚本（如 `run_full_finetune.sh`），根据实际 GPU 数量和数据路径修改参数。
- **GRPO 训练**：使用 `run_grpo.sh` 脚本，需指定预训练的 LoRA 适配器路径和自定义的奖励函数脚本路径。
- 支持恢复训练：脚本第一个参数为 checkpoint 目录时自动恢复。
- 训练日志可通过 wandb 或本地文件记录。

## 推理方法
```bash
python inference.py --model_path ./output/xxx/final_model --image_path ./test.jpg --prompt "图片里有什么？"
```

## 评估扩展
- 可在训练参数中通过 `--eval_scripts` 指定自定义评估脚本。
- 参考 `evaluation/eval_vqa_template.py` 实现评估逻辑。

## 主要依赖
- transformers
- torch
- deepspeed
- peft
- Pillow
- torchvision
- wandb

## 说明
- 推荐使用 Linux/Mac 环境，Windows 可用 `.bat` 脚本。
- GPU 资源和 batch size 请根据实际情况调整。
- 支持多数据集、像素过滤和视觉数据增强。

如有问题欢迎反馈或交流。
