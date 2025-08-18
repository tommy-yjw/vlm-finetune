import argparse
import os
import json
import importlib.util
import logging
import time
from typing import List, Dict, Set

import torch
import deepspeed
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import sys
sys.path.append('/data/oceanus_ctr/j-yanjiangwei-jk/vlm-finetune')
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 高级微调框架")

    # --- 模型与路径 ---
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-VL-Chat", help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, required=True, help="模型检查点和日志的输出目录")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定的DeepSpeed检查点目录恢复训练")
    
    # --- 数据集配置 ---
    parser.add_argument("--dataset_config_path", type=str, default="./configs/dataset_config.json", help="数据集注册配置文件的路径")
    parser.add_argument("--dataset_names", type=str, nargs='+', required=True, help="要使用的已注册数据集的名称列表")
    parser.add_argument("--min_pixels", type=int, default=None, help="图片的最小像素数 (宽 * 高)")
    parser.add_argument("--max_pixels", type=int, default=None, help="图片的最大像素数 (宽 * 高)")
    parser.add_argument("--train_split_ratio", type=float, default=0.9, help="训练集和验证集的划分比例 (0.0到1.0之间)")
    parser.add_argument("--use_data_augmentation", action='store_true', help="对训练图片使用视觉数据增强")

    # --- 训练超参数 ---
    parser.add_argument("--epochs", type=int, default=3, help="训练的总轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个GPU的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate_llm", type=float, default=1e-5, help="LLM部分的学习率")
    parser.add_argument("--learning_rate_vit", type=float, default=1e-4, help="ViT部分的学习率")
    parser.add_argument("--learning_rate_projector", type=float, default=1e-4, help="Projector部分的学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="输入序列最大长度")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader使用的工作进程数")

    # --- 生成参数 (用于评估时的模型推理) ---
    parser.add_argument("--temperature", type=float, default=1.0, help="生成时的温度")
    parser.add_argument("--top_p", type=float, default=1.0, help="生成时的top_p")
    parser.add_argument("--top_k", type=int, default=50, help="生成时的top_k")

    # --- 微调策略 ---
    parser.add_argument("--tuning_strategy", type=str, required=True,
                        choices=['full', 'lora_llm_full_vit', 'vit_only', 'partial_llm_full_vit', 'qlora'],
                        help="微调策略。新增 'qlora' 选项。")
    parser.add_argument("--llm_frozen_layers", type=int, default=0, help="当使用 'partial' 策略时，要冻结的LLM层数。")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA/QLoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA/QLoRA的alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA/QLoRA的dropout")

    # --- 保存、评估与日志 ---
    parser.add_argument("--save_steps", type=int, default=None, help="每N步保存一次检查点。如果未设置，则使用 --save_interval_epochs。")
    parser.add_argument("--save_interval_epochs", type=float, default=1.0, help="每N个epoch保存一次检查点。")
    parser.add_argument("--eval_steps", type=int, default=None, help="每N步进行一次评估。如果未设置，则使用 --eval_interval_epochs。")
    parser.add_argument("--eval_interval_epochs", type=float, default=1.0, help="每N个epoch进行一次评估。")
    parser.add_argument("--eval_script_registry_path", type=str, default="./configs/eval_script_registry.json", help="评估脚本注册配置文件的路径")
    parser.add_argument("--eval_dataset_scripts", type=str, nargs='*', default=[],
                        help="指定要评估的数据集及其对应的评估脚本。格式: 'dataset_name:script_name1,script_name2,...' (脚本名称在 eval_script_registry.json 中定义)")
    parser.add_argument("--use_eval_function", action='store_true', help="是否使用评估函数进行评估，如果为False，则只计算验证集损失。")
    parser.add_argument("--use_wandb", action='store_true', help="启用WandB")
    parser.add_argument("--wandb_project", type=str, default="qwen_vl_finetune", help="WandB项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=f"run-{int(time.time())}", help="WandB运行名称")
    parser.add_argument("--log_to_file", action='store_true', help="将日志记录到本地文件")

    # --- DeepSpeed ---
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed配置文件的路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="由DeepSpeed自动管理的本地进程排名")

    args = parser.parse_args()
    return args

def load_evaluation_modules(eval_scripts_config: List[List[str]]) -> Dict:
    """动态加载评估脚本作为模块"""
    if not eval_scripts_config: return {}
    eval_modules = {}
    for name, path in eval_scripts_config:
        if not os.path.exists(path):
            logger.warning(f"评估脚本路径不存在: {path}，跳过。")
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, 'evaluate'):
            logger.warning(f"评估脚本 {path} 中未找到 'evaluate' 函数，跳过。")
            continue
        eval_modules[name] = module
        logger.info(f"成功加载评估模块 '{name}' 从 {path}")
    return eval_modules

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

def create_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    """根据不同模块创建具有不同学习率的优化器"""
    param_groups = []
    vit_params, projector_params, llm_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_tower" in name:
            vit_params.append(param)
        elif "multi_modal_projector" in name:
            projector_params.append(param)
        else:
            llm_params.append(param)

    if vit_params:
        param_groups.append({"params": vit_params, "lr": args.learning_rate_vit})
    if projector_params:
        param_groups.append({"params": projector_params, "lr": args.learning_rate_projector})
    if llm_params:
        param_groups.append({"params": llm_params, "lr": args.learning_rate_llm})
    
    logger.info(f"优化器分组: ViT({len(vit_params)} params, lr={args.learning_rate_vit}), "
                f"Projector({len(projector_params)} params, lr={args.learning_rate_projector}), "
                f"LLM({len(llm_params)} params, lr={args.learning_rate_llm})")

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    deepspeed.init_distributed()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    training_logger = TrainingLogger(args)
    
    # Load and register datasets from config
    load_datasets_from_config(args.dataset_config_path) # This registers dataset classes

    logger.info(f"正在从 {args.model_name_or_path} 加载模型和处理器...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        # quantization_config=quantization_config,
        trust_remote_code=True,
        use_flash_attention_2=True,
    )
    
    model = setup_model_parameters(model, args)

    # Load evaluation script registry
    eval_registry = {}
    if os.path.exists(args.eval_script_registry_path):
        with open(args.eval_script_registry_path, 'r', encoding='utf-8') as f:
            eval_registry = json.load(f).get("eval_scripts", {})
    else:
        logger.warning(f"评估脚本注册文件未找到: {args.eval_script_registry_path}")

    # Parse eval_dataset_scripts argument
    eval_config_from_args = {} # {dataset_name: [[script_name, script_path], ...]}
    all_eval_scripts_to_load = set() # Collect all unique eval scripts paths
    for entry in args.eval_dataset_scripts:
        parts = entry.split(':')
        if len(parts) != 2:
            logger.warning(f"无效的 --eval_dataset_scripts 格式: {entry}. 预期格式为 'dataset_name:script_name1,script_name2,...'. 跳过。")
            continue
        
        dataset_name, script_names_str = parts
        script_names = script_names_str.split(',')
        
        scripts_for_dataset = []
        for script_name in script_names:
            script_path = eval_registry.get(script_name)
            if script_path:
                scripts_for_dataset.append([script_name, script_path])
                all_eval_scripts_to_load.add((script_name, script_path))
            else:
                logger.warning(f"在注册表中未找到评估脚本 '{script_name}'。")
        
        if scripts_for_dataset:
            eval_config_from_args[dataset_name] = scripts_for_dataset

    # Load all unique evaluation modules once
    loaded_eval_modules = load_evaluation_modules(list(all_eval_scripts_to_load))
    if not loaded_eval_modules and args.eval_steps is not None and args.eval_steps > 0:
        logger.warning("已设置 eval_steps > 0 但没有加载任何评估脚本，将只记录验证集损失。")

    # Prepare datasets for training and evaluation
    all_train_datasets = []
    eval_datasets_by_name = {} # Store evaluation datasets by their original name
    
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
        eval_len = dataset_len - train_len

        if eval_len > 0:
            train_part, eval_part = torch.utils.data.random_split(full_dataset, [train_len, eval_len])
            all_train_datasets.append(train_part)
            
            # Only add to eval_datasets_by_name if it's specified for evaluation in args
            if dataset_name in eval_config_from_args:
                eval_datasets_by_name[dataset_name] = eval_part
                logger.info(f"数据集 '{dataset_name}' 已划分为训练集 ({train_len} 条) 和验证集 ({eval_len} 条)，并准备进行特定评估。")
            else:
                logger.info(f"数据集 '{dataset_name}' 已划分为训练集 ({train_len} 条) 和验证集 ({eval_len} 条)。")
        else:
            all_train_datasets.append(full_dataset)
            logger.info(f"数据集 '{dataset_name}' 未进行验证集划分，所有数据将用于训练。")

    if not all_train_datasets:
        logger.error("没有成功加载任何训练数据集，请检查数据集配置。")
        exit(1)

    combined_train_dataset = ConcatDataset(all_train_datasets)
    logger.info(f"总训练数据集大小: {len(combined_train_dataset)} 条。")

    optimizer = create_optimizer(model, args)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, optimizer=optimizer, config=args.deepspeed
    )
    
    train_sampler = DistributedSampler(combined_train_dataset, num_replicas=model_engine.world_size, rank=model_engine.rank)
    train_dataloader = DataLoader(
        combined_train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 根据 epoch 间隔计算实际的 save_steps 和 eval_steps
    steps_per_epoch = len(train_dataloader)
    if args.save_steps is None:
        args.save_steps = max(1, int(steps_per_epoch * args.save_interval_epochs))
        logger.info(f"保存步数 (save_steps) 已设置为每 {args.save_interval_epochs} 个 epoch 保存一次，即 {args.save_steps} 步。")
    if args.eval_steps is None:
        args.eval_steps = max(1, int(steps_per_epoch * args.eval_interval_epochs))
        logger.info(f"评估步数 (eval_steps) 已设置为每 {args.eval_interval_epochs} 个 epoch 评估一次，即 {args.eval_steps} 步。")

    # Prepare individual evaluation dataloaders for specified datasets
    eval_dataloaders_for_scripts = {}
    for ds_name, eval_part_dataset in eval_datasets_by_name.items():
        eval_sampler = DistributedSampler(eval_part_dataset, num_replicas=model_engine.world_size, rank=model_engine.rank, shuffle=False)
        eval_dataloader = DataLoader(
            eval_part_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=eval_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        eval_dataloaders_for_scripts[ds_name] = eval_dataloader

    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"正在从检查点 {args.resume_from_checkpoint} 恢复训练...")
        _, client_state = model_engine.load_checkpoint(args.resume_from_checkpoint)
        if client_state:
            global_step = client_state.get('global_steps', 0)
            start_epoch = client_state.get('epoch', 0)
            logger.info(f"已恢复到 Step: {global_step}, Epoch: {start_epoch}")
        else:
            logger.warning("未能从检查点加载 client_state，将从头开始计数。")

    logger.info("***** 开始训练 *****")
    for epoch in range(start_epoch, args.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            if not batch: continue
            batch = {k: v.to(model_engine.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            global_step += 1
            
            if args.local_rank == 0:
                training_logger.log({
                    'loss': loss.item(),
                    'learning_rate': model_engine.get_lr()[0]
                }, step=global_step)
                if global_step % 10 == 0:
                     logger.info(f"Epoch: {epoch+1}, Step: {global_step}, Loss: {loss.item():.4f}")

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                if args.local_rank == 0:
                    logger.info(f"***** 正在评估 (Step: {global_step}) *****")
                    model_engine.eval()
                    
                    if args.use_eval_function:
                        # Run per-dataset evaluation scripts
                        for ds_name, eval_scripts_for_ds in eval_config_from_args.items():
                            eval_dataloader_for_ds = eval_dataloaders_for_scripts.get(ds_name)
                            
                            if eval_dataloader_for_ds and eval_scripts_for_ds:
                                logger.info(f"----- 正在评估数据集 '{ds_name}' (Step: {global_step}) -----")
                                for eval_script_name, eval_script_path in eval_scripts_for_ds:
                                    eval_module = loaded_eval_modules.get(eval_script_name)
                                    if eval_module:
                                        logger.info(f"正在运行评估脚本: {eval_script_name} on dataset '{ds_name}'")
                                        results = eval_module.evaluate(
                                            model_engine, 
                                            processor, 
                                            eval_dataloader_for_ds, 
                                            args.output_dir, 
                                            args.local_rank,
                                            temperature=args.temperature,
                                            top_p=args.top_p,
                                            top_k=args.top_k
                                        )
                                        training_logger.log({f'eval_results_{ds_name}_{eval_script_name}': results}, step=global_step)
                                        logger.info(f"评估脚本 '{eval_script_name}' on dataset '{ds_name}' 结果 (Step: {global_step}): {results}")
                                    else:
                                        logger.warning(f"评估脚本 '{eval_script_name}' 未成功加载，跳过。")
                            elif args.local_rank == 0: # Changed from local_rank to args.local_rank for consistency
                                logger.info(f"数据集 '{ds_name}' 没有指定评估脚本或没有评估数据，跳过其特定评估。")
                    else:
                        # Calculate validation loss
                        total_eval_loss = 0.0
                        num_eval_batches = 0
                        if eval_dataloaders_for_scripts:
                            logger.info(f"----- 正在计算验证集损失 (Step: {global_step}) -----")
                            for ds_name, eval_dataloader_for_ds in eval_dataloaders_for_scripts.items():
                                for eval_batch in eval_dataloader_for_ds:
                                    if not eval_batch: continue
                                    eval_batch = {k: v.to(model_engine.device) for k, v in eval_batch.items() if isinstance(v, torch.Tensor)}
                                    with torch.no_grad():
                                        eval_outputs = model_engine(**eval_batch)
                                        eval_loss = eval_outputs.loss
                                    total_eval_loss += eval_loss.item()
                                    num_eval_batches += 1
                            
                            if num_eval_batches > 0:
                                average_eval_loss = total_eval_loss / num_eval_batches
                                training_logger.log({'eval_loss': average_eval_loss}, step=global_step)
                                logger.info(f"平均验证损失 (Step: {global_step}): {average_eval_loss:.4f}")
                            else:
                                logger.warning("没有可用的验证数据来计算损失。")
                        else:
                            logger.warning("没有配置任何验证数据集，跳过验证损失计算。")

                    model_engine.train() # 切换回训练模式

            if global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model_engine.save_checkpoint(checkpoint_dir)

    if args.local_rank == 0:
        logger.info("训练完成，正在保存最终模型...")
        final_save_path = os.path.join(args.output_dir, "final_model")
        unwrapped_model = model_engine.module
        if hasattr(unwrapped_model, "save_pretrained"):
             unwrapped_model.save_pretrained(final_save_path)
        else:
            unwrapped_model.base_model.base_model.save_pretrained(final_save_path)
        processor.save_pretrained(final_save_path)
        logger.info(f"最终模型已保存到 {final_save_path}")
    
    training_logger.close()

if __name__ == "__main__":
    main()
