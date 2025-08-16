# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen-VL Finetuning script integrated with Megatron-LM.

This script is adapted from a DeepSpeed-based training script to use Megatron-LM.
It supports various fine-tuning strategies like LoRA, QLoRA, and partial freezing
for the Qwen-VL model.

Compatibility Notes:
- This script is primarily designed for Data Parallelism (DDP) and Tensor Parallelism (TP).
  Pipeline Parallelism (PP) may require significant model modifications to work correctly.
- The optimizer is simplified to use a single learning rate (`--lr`) as defined by
  Megatron. The original script's grouped learning rates for LLM, ViT, and Projector
  are not used in this version but the arguments are kept for reference.
- In-training evaluation has been simplified. For comprehensive evaluation, it is
  recommended to run evaluation scripts on the final saved model checkpoint.
"""

import argparse
import os
import json
import importlib.util
import logging
import time
from typing import List, Dict, Set

import torch
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler

# Megatron Core Imports
from megatron.core import mpu
from megatron.initialize import initialize_megatron
from megatron.training import get_args, setup_model_and_optimizer, train
from megatron.utils import get_timers, report_memory

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from datasets.dataset_registry import get_dataset, register_dataset, load_datasets_from_config # 导入数据集注册模块和加载函数

# 尝试导入 wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingLogger:
    """
    一个统一的日志记录器，支持WandB和本地文件日志。
    只在主进程 (local_rank == 0) 上进行操作。
    """
    def __init__(self, args):
        self.args = args
        self.is_main_process = args.local_rank == 0
        
        if not self.is_main_process:
            return

        # 初始化 WandB
        if args.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb 未安装，将禁用 wandb 日志。请运行 'pip install wandb'。")
                self.args.use_wandb = False
            else:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    config=vars(args)
                )
                logger.info(f"WandB 已初始化。项目: {args.wandb_project}, 运行: {args.wandb_run_name}")

        # 初始化本地日志文件
        if args.log_to_file:
            self.log_file_path = os.path.join(args.output_dir, "training_log.jsonl")
            # 如果是从检查点恢复，则以追加模式打开
            mode = 'a' if args.resume_from_checkpoint else 'w'
            self.log_file = open(self.log_file_path, mode, encoding='utf-8')
            logger.info(f"本地日志将记录到: {self.log_file_path}")

    def log(self, data: Dict, step: int):
        if not self.is_main_process:
            return
            
        log_data = {'step': step, **data}

        if self.args.use_wandb:
            wandb.log(log_data, step=step)

        if self.args.log_to_file:
            self.log_file.write(json.dumps(log_data) + '\n')
            self.log_file.flush()

    def close(self):
        if not self.is_main_process:
            return
        
        if self.args.use_wandb:
            wandb.finish()
        
        if self.args.log_to_file and hasattr(self, 'log_file'):
            self.log_file.close()


def extra_args_provider(parser):
    """为Megatron的参数解析器添加额外的自定义参数"""
    group = parser.add_argument_group(title='Qwen-VL Finetune Arguments')

    # --- 模型与路径 ---
    group.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-VL-Chat", help="预训练模型路径")
    # Note: Megatron's --save argument is used for the output directory.
    # Note: Megatron's --load argument is used to resume from a checkpoint.
    
    # --- 数据集配置 ---
    group.add_argument("--dataset_config_path", type=str, default="./configs/dataset_config.json", help="数据集注册配置文件的路径")
    group.add_argument("--dataset_names", type=str, nargs='+', required=True, help="要使用的已注册数据集的名称列表")
    group.add_argument("--min_pixels", type=int, default=None, help="图片的最小像素数 (宽 * 高)")
    group.add_argument("--max_pixels", type=int, default=None, help="图片的最大像素数 (宽 * 高)")
    group.add_argument("--train_split_ratio", type=float, default=0.9, help="训练集和验证集的划分比例 (0.0到1.0之间)")
    group.add_argument("--use_data_augmentation", action='store_true', help="对训练图片使用视觉数据增强")

    # --- 训练超参数 ---
    group.add_argument("--epochs", type=int, default=3, help="训练的总轮数 (将被转换为 train-iters)")
    group.add_argument("--max_seq_length", type=int, default=2048, help="输入序列最大长度")
    group.add_argument("--num_workers", type=int, default=4, help="DataLoader使用的工作进程数")

    # --- 微调策略 ---
    group.add_argument("--tuning_strategy", type=str, required=True,
                        choices=['full', 'lora_llm_full_vit', 'vit_only', 'partial_llm_full_vit', 'qlora'],
                        help="微调策略。新增 'qlora' 选项。")
    group.add_argument("--llm_frozen_layers", type=int, default=0, help="当使用 'partial' 策略时，要冻结的LLM层数。")
    group.add_argument("--lora_rank", type=int, default=64, help="LoRA/QLoRA的秩")
    group.add_argument("--lora_alpha", type=int, default=16, help="LoRA/QLoRA的alpha")
    group.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA/QLoRA的dropout")

    # --- 日志 ---
    group.add_argument("--use_wandb", action='store_true', help="启用WandB")
    group.add_argument("--wandb_project", type=str, default="qwen_vl_finetune", help="WandB项目名称")
    group.add_argument("--wandb_run_name", type=str, default=f"run-{int(time.time())}", help="WandB运行名称")
    group.add_argument("--log_to_file", action='store_true', help="将日志记录到本地文件")

    # Set some sane defaults for arguments that are not typically part of Megatron's defaults.
    parser.set_defaults(
        lr_decay_style='cosine',
        fp16=True,
        # Set a default for train_iters to be calculated later if not provided.
        train_iters=None,
    )
    return parser

def setup_model_parameters(model: torch.nn.Module, args: argparse.Namespace):
    """根据微调策略设置模型参数的 requires_grad 属性"""
    if args.tuning_strategy == 'qlora':
        logger.info(f"应用QLoRA策略 (rank={args.lora_rank}, alpha={args.lora_alpha})...")
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    for param in model.parameters():
        param.requires_grad = False
    
    strategy = args.tuning_strategy
    logger.info(f"应用微调策略: {strategy}")

    if strategy in ['lora_llm_full_vit', 'vit_only', 'partial_llm_full_vit', 'dynamic_freeze_llm_full_vit']:
        logger.info("正在解冻 ViT (vision_tower) 参数...")
        for name, param in model.named_parameters():
            if "vision_tower" in name:
                param.requires_grad = True
    
    logger.info("正在解冻 Projector (multi_modal_projector) 参数...")
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True

    if strategy == 'full':
        logger.info("正在解冻所有模型参数进行全参数微调...")
        for param in model.parameters():
            param.requires_grad = True
    
    elif strategy == 'lora_llm_full_vit':
        logger.info(f"为LLM应用LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif strategy == 'vit_only':
        logger.info("LLM部分已完全冻结，只训练ViT和Projector。")

    elif strategy in ['partial_llm_full_vit', 'dynamic_freeze_llm_full_vit']:
        num_frozen = args.llm_frozen_layers
        logger.info(f"正在解冻LLM的最后 {len(model.model.layers) - num_frozen} 层...")
        if num_frozen < 0 or num_frozen >= len(model.model.layers):
            raise ValueError(f"llm_frozen_layers 必须在 [0, {len(model.model.layers) - 1}] 范围内")
        
        for i, layer in enumerate(model.model.layers):
            if i >= num_frozen:
                for param in layer.parameters():
                    param.requires_grad = True
        
        logger.info("正在解冻LLM的 lm_head 和 final_layernorm...")
        for param in model.lm_head.parameters():
            param.requires_grad = True
        for param in model.model.norm.parameters():
            param.requires_grad = True
            
    return model

def model_provider(pre_process=True, post_process=True):
    """
    构建并返回模型。
    这是Megatron在 `setup_model_and_optimizer` 中调用的回调函数。
    """
    args = get_args()
    
    logger.info(f"正在从 {args.model_name_or_path} 加载模型...")
    
    quantization_config = None
    torch_dtype = torch.float16
    if args.tuning_strategy == 'qlora':
        logger.info("为QLoRA准备4-bit量化配置...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
        use_flash_attention_2=True,
    )
    
    return setup_model_parameters(model, args)

def forward_step(data_iterator, model):
    """
    Megatron的单步前向传播函数。
    """
    args = get_args()
    timers = get_timers()

    # 从迭代器获取数据
    timers('batch-generator').start()
    try:
        batch = next(data_iterator)
    except StopIteration:
        return None, None # 训练结束的信号
    timers('batch-generator').stop()

    # 将数据移动到正确的设备
    batch = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    # 模型前向传播
    outputs = model(**batch)
    
    # 定义损失函数
    def loss_func(loss_tensor):
        # 在这里，`loss_tensor` 就是 `outputs.loss`
        # Megatron的并行化和梯度计算会自动处理这个损失
        return loss_tensor, {'loss': loss_tensor}

    return outputs.loss, loss_func

def build_data_iterator(dataset, num_workers):
    """
    将 PyTorch Dataset 包装成 Megatron 所需的无限数据迭代器。
    """
    args = get_args()
    
    # Megatron 要求使用其自身的数据并行设置
    sampler = DistributedSampler(
        dataset,
        num_replicas=mpu.get_data_parallel_world_size(),
        rank=mpu.get_data_parallel_rank(),
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # 确保所有批次大小一致
    )

    # 包装成无限迭代器
    while True:
        for batch in dataloader:
            yield batch

def main():
    """Main training program."""
    # Initalize Megatron
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'micro_batch_size': 1, # Default, will be overridden by args
                                       'global_batch_size': 8, # Default, will be overridden by args
                                       'no_load_optim': True,
                                       'no_load_rng': True})
    
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # The TrainingLogger now needs to be initialized after Megatron's initialization
    # to get the correct rank information.
    training_logger = TrainingLogger(args)
    
    # 设置模型、优化器和学习率调度器
    # 注意：分组学习率在此简化，统一使用 args.lr
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    
    # Load and register datasets from config
    load_datasets_from_config(args.dataset_config_path) # This registers dataset classes

    logger.info(f"正在加载处理器...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Prepare datasets for training
    all_train_datasets = []
    
    for dataset_name in args.dataset_names:
        full_dataset = get_dataset(
            name=dataset_name,
            processor=processor,
            max_length=args.max_seq_length,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels
        )
        
        if not full_dataset:
            logger.error(f"无法加载数据集 '{dataset_name}'，请检查其是否已注册。")
            exit(1)

        # Split each dataset into train and eval parts
        dataset_len = len(full_dataset)
        train_len = int(args.train_split_ratio * dataset_len)
        eval_len = dataset_len - train_len # eval_len is still calculated but not used for in-training eval

        if eval_len > 0:
            train_part, _ = torch.utils.data.random_split(full_dataset, [train_len, eval_len])
            all_train_datasets.append(train_part)
            logger.info(f"数据集 '{dataset_name}' 已划分为训练集 ({train_len} 条) 和验证集 ({eval_len} 条)。")
        else:
            all_train_datasets.append(full_dataset)
            logger.info(f"数据集 '{dataset_name}' 未进行验证集划分，所有数据将用于训练。")

    if not all_train_datasets:
        logger.error("没有成功加载任何训练数据集，请检查数据集配置。")
        exit(1)

    combined_train_dataset = ConcatDataset(all_train_datasets)
    logger.info(f"总训练数据集大小: {len(combined_train_dataset)} 条。")

    # --- 启动训练 ---
    logger.info("***** 开始使用 Megatron 进行训练 *****")

    # 根据 epoch 计算 train_iters
    if args.train_iters is None:
        num_train_samples = len(combined_train_dataset)
        # global_batch_size = micro_batch_size * data_parallel_size * gradient_accumulation_steps
        # Megatron's get_args() should already have global_batch_size calculated
        if not hasattr(args, 'global_batch_size'):
             args.global_batch_size = args.micro_batch_size * mpu.get_data_parallel_world_size() * args.gradient_accumulation_steps
        
        iters_per_epoch = num_train_samples // args.global_batch_size
        args.train_iters = args.epochs * iters_per_epoch
        logger.info(f"根据 Epochs ({args.epochs}) 和数据集大小 ({num_train_samples}) 计算出的总训练步数: {args.train_iters}")

    # 创建训练数据迭代器
    train_data_iterator = build_data_iterator(combined_train_dataset, args.num_workers)
    
    # TODO: 创建验证数据迭代器和评估回调
    valid_data_iterator = None

    # 调用Megatron的训练函数
    iteration = train(
        forward_step_func=forward_step,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data_iterator=train_data_iterator,
        valid_data_iterator=valid_data_iterator
    )

    logger.info(f"训练完成，最终步数: {iteration}")

    # --- 保存最终模型 ---
    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        if args.rank == 0:
            logger.info("正在保存最终模型...")
            final_save_path = os.path.join(args.output_dir, "final_model")
            
            # 获取未包装的模型
            unwrapped_model = model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module

            if hasattr(unwrapped_model, "save_pretrained"):
                 unwrapped_model.save_pretrained(final_save_path)
            else:
                # 适用于PEFT模型
                unwrapped_model.base_model.base_model.save_pretrained(final_save_path)
            
            processor.save_pretrained(final_save_path)
            logger.info(f"最终模型已保存到 {final_save_path}")

    training_logger.close()

if __name__ == "__main__":
    main()
