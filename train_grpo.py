import argparse
import os
import json
import logging
import time
import importlib.util
from typing import List, Dict, Tuple, Optional

import torch
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ---- Optional: vLLM for rollout (single-node, multi-GPU inference) ----
try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal import MultiModalData
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

# ---- Your dataset (expects text prompts + images + optional image_paths) ----
from datasets.custom_dataset import QwenVLDataset  # prompt-only for RL is common

# ---- GRPO utils (token log-probs + loss) ----
from core.grpo_utils import calculate_log_probs, compute_grpo_loss

# ---- wandb (optional) ----
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ------------------------ Logging ------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GRPOTrainingLogger:
    def __init__(self, args):
        self.args = args
        self.is_main = (args.local_rank in (-1, 0)) or (getattr(args, 'global_rank', 0) == 0)
        if not self.is_main:
            return

        if args.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb 未安装，将禁用 wandb 日志。请运行 'pip install wandb'.")
                self.args.use_wandb = False
            else:
                wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

        if args.log_to_file:
            os.makedirs(args.output_dir, exist_ok=True)
            self.log_file_path = os.path.join(args.output_dir, "grpo_training_log.jsonl")
            mode = 'a' if args.resume_from_checkpoint else 'w'
            self.log_file = open(self.log_file_path, mode, encoding='utf-8')

    def log(self, data: Dict, step: int):
        if not self.is_main:
            return
        if self.args.use_wandb:
            wandb.log({**data, 'step': step}, step=step)
        if self.args.log_to_file:
            self.log_file.write(json.dumps({'step': step, **data}, ensure_ascii=False) + '\n')
            self.log_file.flush()

    def close(self):
        if not self.is_main:
            return
        if self.args.use_wandb:
            wandb.finish()
        if self.args.log_to_file and hasattr(self, 'log_file'):
            self.log_file.close()


# ------------------------ Argparse ------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="GRPO for Qwen2.5-VL with optional vLLM rollout (single-node multi-GPU)")

    # Model & paths
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-VL-Chat')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    # Datasets: list of (json_path, image_root)
    parser.add_argument('--datasets', nargs=2, action='append', metavar=('JSON_PATH', 'IMAGE_ROOT'), required=True)

    # GRPO hyperparams
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate_lora', type=float, default=1e-6)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--k_samples', type=int, default=4, help='candidates per prompt')
    parser.add_argument('--rollout_max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)

    # KL penalty
    parser.add_argument('--use_kl_penalty', action='store_true')
    parser.add_argument('--kl_beta', type=float, default=0.1)

    # Tuning strategy
    parser.add_argument('--tuning_strategy', type=str, default='qlora', choices=['qlora', 'lora', 'full'])
    parser.add_argument('--lora_adapter_path', type=str, default=None)
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Logging & saving
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--save_total_limit', type=int, default=5, help='max number of checkpoints to keep')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='qwen_vl_grpo')
    parser.add_argument('--wandb_run_name', type=str, default=f"run-{int(time.time())}")
    parser.add_argument('--log_to_file', action='store_true')

    # DeepSpeed
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=-1)

    # vLLM rollout options
    parser.add_argument('--use_vllm_rollout', action='store_true', help='use vLLM for rollout generation')
    parser.add_argument('--vllm_tp', type=int, default=1, help='tensor parallel GPUs for vLLM')
    parser.add_argument('--vllm_max_model_len', type=int, default=4096)
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9)

    args = parser.parse_args()
    return args


# ------------------------ Reward function loader ------------------------

def load_reward_function(script_path: str):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"奖励函数脚本未找到: {script_path}")
    spec = importlib.util.spec_from_file_location("reward_function_module", script_path)
    reward_function_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reward_function_module)
    if not hasattr(reward_function_module, 'get_reward_function'):
        raise AttributeError(f"脚本 {script_path} 中未找到 get_reward_function()")
    return reward_function_module.get_reward_function()


# ------------------------ Rollout Engines ------------------------

class HFEngine:
    """Use training model_engine to do rollout (reference backend)."""
    def __init__(self, model_engine, processor, max_new_tokens: int, temperature: float, top_p: float):
        self.model_engine = model_engine
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def generate(self, prompts: List[str], images: List, device: torch.device) -> List[str]:
        inputs = self.processor(text=prompts, images=images, return_tensors='pt', padding=True).to(device)
        prompt_len = inputs['input_ids'].shape[1]
        gen_ids = self.model_engine.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=self.top_p,
            temperature=self.temperature
        )
        return self.processor.batch_decode(gen_ids[:, prompt_len:], skip_special_tokens=True)


class VLLMEngine:
    """Use vLLM for fast single-node multi-GPU rollout. Requires image paths or PILs."""
    def __init__(self, args, processor):
        if not VLLM_AVAILABLE:
            raise RuntimeError('vLLM 未安装，无法使用 --use_vllm_rollout')
        self.processor = processor
        self.sampling = SamplingParams(
            max_tokens=args.rollout_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # tensor_parallel_size uses multiple GPUs on a single node
        self.llm = LLM(
            model=args.model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=args.vllm_tp,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )

    def _to_mm_data(self, image_paths: List[str]):
        # vLLM MultiModalData expects dict with modality lists
        return [MultiModalData(image_paths=[p]) for p in image_paths]

    def generate(self, prompts: List[str], image_paths: List[str]) -> List[str]:
        # vLLM multimodal: pass list of dicts with prompt + mm data
        requests = []
        mm_inputs = self._to_mm_data(image_paths)
        for p, mm in zip(prompts, mm_inputs):
            requests.append({
                'prompt': p,
                'multi_modal_data': mm,
            })
        outputs = self.llm.generate(requests, self.sampling)
        # outputs is a list of RequestOutput
        return [o.outputs[0].text for o in outputs]


# ------------------------ Utilities ------------------------

def cleanup_old_checkpoints(base_dir: str, keep_last: int):
    if keep_last <= 0:
        return
    if not os.path.isdir(base_dir):
        return
    # gather tags like step-XXXX
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('step-')]
    def _key(s):
        try:
            return int(s.split('-')[-1])
        except Exception:
            return -1
    subdirs.sort(key=_key)
    while len(subdirs) > keep_last:
        d = subdirs.pop(0)
        path = os.path.join(base_dir, d)
        try:
            import shutil
            shutil.rmtree(path)
            logger.info(f"Removed old checkpoint: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")


# ------------------------ Main ------------------------

def main():
    args = parse_arguments()

    # Distributed init (DeepSpeed single-node multi-GPU)
    deepspeed.init_distributed()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        args.global_rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        args.global_rank = 0
        args.world_size = 1

    # (Optional) reward function
    # Keep optional to allow dry-run without a script
    reward_fn = None
    if hasattr(args, 'reward_function_path') and args.__dict__.get('reward_function_path'):
        reward_fn = load_reward_function(args.reward_function_path)

    # Processor
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Model (training)
    quant_cfg = None
    if args.tuning_strategy == 'qlora':
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        trust_remote_code=True,
        use_flash_attention_2=True,
    )

    # Reference model (for KL)
    ref_model = None
    if args.use_kl_penalty:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attention_2=True,
        )
        ref_model.eval()

    # Apply (Q)LoRA or full finetune
    if args.tuning_strategy in ['qlora', 'lora']:
        if args.tuning_strategy == 'qlora':
            model = prepare_model_for_kbit_training(model)
        if args.lora_adapter_path:
            logger.info(f"Loading LoRA adapter from {args.lora_adapter_path}")
            model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)
            if ref_model is not None:
                ref_model = PeftModel.from_pretrained(ref_model, args.lora_adapter_path)
        else:
            lora_cfg = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    else:
        logger.info("Using full-parameter finetuning (no LoRA/QLoRA)")

    # Dataset (use the first pair for now; extend to multiple if needed)
    json_path, image_root = args.datasets[0]
    dataset = QwenVLDataset(json_path, image_root, processor, args.max_seq_length)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, sampler=sampler, num_workers=args.num_workers)

    # DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

    # Prepare reference model device
    if ref_model is not None:
        ref_model = ref_model.to(model_engine.device)

    # Rollout engine (HF default; vLLM optional)
    rollout_engine = None
    if args.use_vllm_rollout:
        if not VLLM_AVAILABLE:
            raise RuntimeError('设置了 --use_vllm_rollout 但未安装 vLLM')
        if args.global_rank == 0:
            logger.info(f"Using vLLM rollout with TP={args.vllm_tp}")
        rollout_engine = VLLMEngine(args, processor)
    else:
        rollout_engine = HFEngine(model_engine, processor, args.rollout_max_new_tokens, args.temperature, args.top_p)

    # Logger
    tlogger = GRPOTrainingLogger(args)
    global_step = 0

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            model_engine.train()

            prompts: List[str] = batch['text']
            # Images: tensor (for HF) + optional 'image_paths' (for vLLM)
            images = batch['image'].to(model_engine.device)
            image_paths: Optional[List[str]] = batch.get('image_paths') if isinstance(batch, dict) else None

            # 1) Rollout K samples per prompt
            all_responses_per_k: List[List[str]] = []
            all_log_probs_per_k: List[torch.Tensor] = []
            all_ref_log_probs_per_k: List[torch.Tensor] = []

            with torch.no_grad():
                for _k in range(args.k_samples):
                    if args.use_vllm_rollout:
                        if image_paths is None:
                            raise ValueError('vLLM 回滚需要数据集返回 image_paths（原始图片路径）以支持多模态。请在 QwenVLDataset.__getitem__ 中加入该字段。')
                        responses = rollout_engine.generate(prompts, image_paths)
                        # For logprobs, still compute with HF model locally
                        inputs = processor(text=prompts, images=images, return_tensors='pt', padding=True).to(model_engine.device)
                        prompt_len = inputs['input_ids'].shape[1]
                        # Re-tokenize concatenated prompt+response for log-prob calc
                        # Build full ids by re-encoding prompt+response with processor
                        full_texts = [p + r for p, r in zip(prompts, responses)]
                        full_inputs = processor(text=full_texts, images=[None]*len(full_texts), return_tensors='pt', padding=True).to(model_engine.device)
                        log_probs = calculate_log_probs(model_engine.module, full_inputs['input_ids'], prompt_len=None, attention_mask=full_inputs.get('attention_mask'))
                        if ref_model is not None:
                            ref_log_probs = calculate_log_probs(ref_model, full_inputs['input_ids'], prompt_len=None, attention_mask=full_inputs.get('attention_mask'))
                    else:
                        # HF engine: generate & compute log_probs on same tokens
                        inputs = processor(text=prompts, images=images, return_tensors='pt', padding=True).to(model_engine.device)
                        prompt_len = inputs['input_ids'].shape[1]
                        gen_ids = model_engine.generate(
                            **inputs,
                            max_new_tokens=args.rollout_max_new_tokens,
                            do_sample=True,
                            top_p=args.top_p,
                            temperature=args.temperature,
                        )
                        responses = processor.batch_decode(gen_ids[:, prompt_len:], skip_special_tokens=True)
                        log_probs = calculate_log_probs(model_engine.module, gen_ids, prompt_len, inputs['attention_mask'])
                        if ref_model is not None:
                            ref_log_probs = calculate_log_probs(ref_model, gen_ids, prompt_len, inputs['attention_mask'])

                    all_responses_per_k.append(responses)
                    all_log_probs_per_k.append(log_probs)
                    if ref_model is not None:
                        all_ref_log_probs_per_k.append(ref_log_probs)

            # 2) Rewards
            # transpose from [K][B] -> [B][K]
            transposed_responses = list(zip(*all_responses_per_k))  # B * [K]
            rewards: List[List[float]] = []
            for i in range(len(prompts)):
                if reward_fn is None:
                    # fallback: length-based dummy reward
                    rs = [float(len(r)) for r in transposed_responses[i]]
                else:
                    # reward_fn(prompts_i_repeated, responses_i_list, images_i_repeated)
                    img_for_i = [images[i]] * args.k_samples
                    rs = reward_fn([prompts[i]] * args.k_samples, list(transposed_responses[i]), img_for_i)
                rewards.append(rs)

            rewards_tensor = torch.tensor(rewards, device=model_engine.device, dtype=torch.float32)  # [B, K]
            log_probs_tensor = torch.stack(all_log_probs_per_k, dim=1)  # [B, K]

            # 3) GRPO loss
            # flatten across K
            B, K = rewards_tensor.shape
            log_probs_all = log_probs_tensor.reshape(B * K)
            rewards_all = rewards_tensor.reshape(B * K)

            loss_grpo = compute_grpo_loss(
                log_probs_all_samples=log_probs_all,
                rewards_all_samples=rewards_all,
                k_samples=K,
                beta=getattr(args, 'kl_beta', 0.0),  # allow function to use if needed
            )

            # Optional KL penalty against ref model
            kl_div_val = torch.tensor(0.0, device=model_engine.device)
            if ref_model is not None and len(all_ref_log_probs_per_k) == len(all_log_probs_per_k):
                ref_log_probs_tensor = torch.stack(all_ref_log_probs_per_k, dim=1)  # [B, K]
                kl_div_val = (log_probs_tensor.detach() - ref_log_probs_tensor.detach()).mean()
                loss = loss_grpo + args.kl_beta * kl_div_val
            else:
                loss = loss_grpo

            # 4) Backward & step (DeepSpeed)
            model_engine.backward(loss)
            model_engine.step()

            # 5) Logging
            if args.global_rank == 0:
                payload = {
                    'loss': float(loss.detach().item()),
                    'loss_grpo': float(loss_grpo.detach().item()),
                    'reward_mean': float(rewards_tensor.mean().item()),
                }
                if ref_model is not None:
                    payload['kl_div'] = float(kl_div_val.detach().item())
                tlogger.log(payload, global_step)
                logger.info(f"Epoch {epoch} | Step {global_step} | loss={payload['loss']:.4f} | grpo={payload['loss_grpo']:.4f} | reward={payload['reward_mean']:.4f} " + (f"| kl={payload['kl_div']:.4f}" if 'kl_div' in payload else ''))

            # 6) Periodic checkpoint saving with tag
            if (global_step % args.save_steps == 0) and (args.global_rank == 0):
                tag = f"step-{global_step}"
                model_engine.save_checkpoint(args.output_dir, tag=tag)
                logger.info(f"Checkpoint saved: {os.path.join(args.output_dir, tag)}")
                cleanup_old_checkpoints(args.output_dir, args.save_total_limit)

            global_step += 1

    # final save
    if args.global_rank == 0:
        tag = f"step-{global_step}-final"
        model_engine.save_checkpoint(args.output_dir, tag=tag)
        logger.info(f"Final checkpoint saved: {os.path.join(args.output_dir, tag)}")

    tlogger.close()


if __name__ == '__main__':
    main()
