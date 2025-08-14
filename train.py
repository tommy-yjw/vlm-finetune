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
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 动态导入，因为 custom_dataset.py 现在有两个类
from datasets.custom_dataset import QwenVLDataset, QwenVLDatasetWithAug

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
    parser.add_argument("--datasets", nargs=2, action='append', metavar=('JSON_PATH', 'IMAGE_ROOT'), required=True, help="指定初始数据集")
    parser.add_argument("--dynamic_dataset_dir", type=str, default=None, help="动态加载数据集的目录")
    parser.add_argument("--min_pixels", type=int, default=None, help="图片的最小像素数 (宽 * 高)")
    parser.add_argument("--max_pixels", type=int, default=None, help="图片的最大像素数 (宽 * 高)")
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

    # --- 微调策略 ---
    parser.add_argument("--tuning_strategy", type=str, required=True,
                        choices=['full', 'lora_llm_full_vit', 'vit_only', 'partial_llm_full_vit', 'qlora'],
                        help="微调策略。新增 'qlora' 选项。")
    parser.add_argument("--llm_frozen_layers", type=int, default=0, help="当使用 'partial' 策略时，要冻结的LLM层数。")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA/QLoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA/QLoRA的alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA/QLoRA的dropout")

    # --- 保存、评估与日志 ---
    parser.add_argument("--save_steps", type=int, default=500, help="每N步保存一次检查点")
    parser.add_argument("--eval_steps", type=int, default=250, help="每N步进行一次评估")
    parser.add_argument("--eval_scripts", nargs=2, action='append', metavar=('NAME', 'SCRIPT_PATH'), help="指定评估脚本")
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

def get_dataset_class(use_augmentation: bool):
    """根据是否使用数据增强返回相应的数据集类"""
    if use_augmentation:
        logger.info("使用带数据增强的数据集: QwenVLDatasetWithAug")
        return QwenVLDatasetWithAug
    else:
        logger.info("使用标准数据集: QwenVLDataset")
        return QwenVLDataset

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    deepspeed.init_distributed()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    training_logger = TrainingLogger(args)
    
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
        use_flash_attention_2=True,
    )
    
    model = setup_model_parameters(model, args)

    DatasetClass = get_dataset_class(args.use_data_augmentation)
    initial_datasets = [
        DatasetClass(jp, ir, processor, args.max_seq_length, args.min_pixels, args.max_pixels) 
        for jp, ir in args.datasets
    ]
    train_dataset = ConcatDataset(initial_datasets)
    
    optimizer = create_optimizer(model, args)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, optimizer=optimizer, config=args.deepspeed
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=model_engine.world_size, rank=model_engine.rank)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

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
