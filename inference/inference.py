import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import json
import os
import importlib.util # 新增导入
from typing import List, Dict # 新增导入

from datasets.dataset_registry import get_dataset, load_datasets_from_config # 新增导入

def main_inference(model_path, image_path, prompt, device, temperature: float, top_p: float, top_k: int):
    """
    使用微调后的Qwen-VL模型进行单样本推理。
    """
    print(f"正在从 {model_path} 加载模型...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("模型加载完成。")

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {image_path}")
        return
    except Exception as e:
        print(f"错误：加载图片 {image_path} 失败: {e}")
        return

    query = [{'role': 'user', 'content': f'<image>\n{prompt}'}]
    
    text = processor.tokenizer.from_list_format(query)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    print("\n正在生成回答...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True if temperature > 0.0 else False, # 根据temperature设置do_sample
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    input_token_len = inputs["input_ids"].shape[1]
    response_ids = generated_ids[:, input_token_len:]
    
    response = processor.batch_decode(response_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*20 + " 模型回答 " + "="*20)
    print(response.strip())
    print("="*50 + "\n")

def evaluate(model_path: str, dataset_names: List[str], eval_script_registry_path: str, eval_dataset_scripts: List[str], output_dir: str, device: str, temperature: float, top_p: float, top_k: int):
    """
    使用微调后的Qwen-VL模型在数据集上进行评估。
    数据集通过 dataset_registry 加载，评估逻辑通过 eval_script_registry 动态加载。
    """
    print(f"正在从 {model_path} 加载模型进行评估...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("模型加载完成。")

    # 加载并注册数据集配置
    # 假设 dataset_config.json 位于 configs/ 目录下
    load_datasets_from_config("./configs/dataset_config.json") 

    # 加载评估脚本注册表
    eval_registry = {}
    if os.path.exists(eval_script_registry_path):
        with open(eval_script_registry_path, 'r', encoding='utf-8') as f:
            eval_registry = json.load(f).get("eval_scripts", {})
    else:
        print(f"警告：评估脚本注册文件未找到: {eval_script_registry_path}")

    # 解析 eval_dataset_scripts 参数
    eval_config_from_args = {} # {dataset_name: [[script_name, script_path], ...]}
    all_eval_scripts_to_load = set() # 收集所有唯一的评估脚本路径
    for entry in eval_dataset_scripts:
        parts = entry.split(':')
        if len(parts) != 2:
            print(f"警告：无效的 --eval_dataset_scripts 格式: {entry}. 预期格式为 'dataset_name:script_name1,script_name2,...'. 跳过。")
            continue
        
        dataset_name, script_names_str = parts
        script_names = script_names_str.split(',')
        
        scripts_for_dataset = []
        for script_name in script_names:
            script_path = eval_registry.get(script_name)
            if script_path:
                # 调整路径，如果注册表中的路径是相对于项目根目录的完整路径
                # 例如 "qwen_finetune_project/evaluation/eval_bbox_mbr_and_pass_rate.py"
                if script_path.startswith("qwen_finetune_project/"):
                    script_path = script_path.replace("qwen_finetune_project/", "")
                scripts_for_dataset.append([script_name, script_path])
                all_eval_scripts_to_load.add((script_name, script_path))
            else:
                print(f"警告：在注册表中未找到评估脚本 '{script_name}'。")
        
        if scripts_for_dataset:
            eval_config_from_args[dataset_name] = scripts_for_dataset

    # 一次性加载所有唯一的评估模块
    loaded_eval_modules = load_evaluation_modules(list(all_eval_scripts_to_load))
    if not loaded_eval_modules and eval_dataset_scripts:
        print("警告：已指定评估数据集但没有加载任何评估脚本。")

    # 准备评估数据集
    eval_datasets_by_name = {}
    for dataset_name in dataset_names:
        full_dataset = get_dataset(
            name=dataset_name,
            processor=processor,
            max_length=2048, # 评估时假设一个默认的最大长度
            min_pixels=None, 
            max_pixels=None  
        )
        if full_dataset:
            eval_datasets_by_name[dataset_name] = full_dataset
            print(f"成功加载评估数据集 '{dataset_name}'，包含 {len(full_dataset)} 条数据。")
        else:
            print(f"错误：无法加载数据集 '{dataset_name}'，请检查其是否已注册。")

    # 准备每个评估数据集的 DataLoader
    eval_dataloaders_for_scripts = {}
    for ds_name, eval_dataset in eval_datasets_by_name.items():
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1, # 评估时通常使用批次大小1
            shuffle=False,
            num_workers=0, # 可根据需要调整
            pin_memory=True
        )
        eval_dataloaders_for_scripts[ds_name] = eval_dataloader

    # 运行每个数据集的评估脚本
    for ds_name, eval_scripts_for_ds in eval_config_from_args.items():
        eval_dataloader_for_ds = eval_dataloaders_for_scripts.get(ds_name)
        
        if eval_dataloader_for_ds and eval_scripts_for_ds:
            print(f"----- 正在评估数据集 '{ds_name}' -----")
            for eval_script_name, eval_script_path in eval_scripts_for_ds:
                eval_module = loaded_eval_modules.get(eval_script_name)
                if eval_module:
                    print(f"正在运行评估脚本: {eval_script_name} on dataset '{ds_name}'")
                    # 传递模型、处理器、dataloader、输出目录和生成参数
                    results = eval_module.evaluate(
                        model, 
                        processor, 
                        eval_dataloader_for_ds, 
                        output_dir, 
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )
                    print(f"评估脚本 '{eval_script_name}' on dataset '{ds_name}' 结果: {results}")
                else:
                    print(f"警告：评估脚本 '{eval_script_name}' 未成功加载，跳过。")
        else:
            print(f"数据集 '{ds_name}' 没有指定评估脚本或没有评估数据，跳过其特定评估。")

    print(f"所有评估任务完成。结果已保存到 {output_dir} (如果评估脚本有保存的话)。")

def load_evaluation_modules(eval_scripts_config: List[List[str]]) -> Dict:
    """动态加载评估脚本作为模块 (从 train.py 复制)"""
    if not eval_scripts_config: return {}
    eval_modules = {}
    for name, path in eval_scripts_config:
        if not os.path.exists(path):
            print(f"警告：评估脚本路径不存在: {path}，跳过。")
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, 'evaluate'):
            print(f"警告：评估脚本 {path} 中未找到 'evaluate' 函数，跳过。")
            continue
        eval_modules[name] = module
        print(f"成功加载评估模块 '{name}' 从 {path}")
    return eval_modules

def main():
    parser = argparse.ArgumentParser(description="使用微调后的Qwen-VL模型进行推理或评估")
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "evaluate"],
                        help="运行模式: 'inference' (单样本推理) 或 'evaluate' (数据集评估)")
    parser.add_argument("--model_path", type=str, required=True, help="微调后模型的路径 (例如, ./output/qwen_lora_finetune/final_model)")
    parser.add_argument("--device", type=str, default="cuda", help="运行推理或评估的设备 (cuda or cpu)")

    # 生成参数 (适用于推理和评估)
    parser.add_argument("--temperature", type=float, default=1.0, help="生成时的温度")
    parser.add_argument("--top_p", type=float, default=1.0, help="生成时的top_p")
    parser.add_argument("--top_k", type=int, default=50, help="生成时的top_k")

    # 推理模式特有参数
    parser.add_argument("--image_path", type=str, help="要进行推理的图片路径 (仅在inference模式下需要)")
    parser.add_argument("--prompt", type=str, help="向模型提出的问题或指令 (仅在inference模式下需要)")

    # 评估模式特有参数
    parser.add_argument("--dataset_names", type=str, nargs='+', help="要评估的已注册数据集的名称列表 (仅在evaluate模式下需要)")
    parser.add_argument("--eval_script_registry_path", type=str, default="./configs/eval_script_registry.json", help="评估脚本注册配置文件的路径 (仅在evaluate模式下需要)")
    parser.add_argument("--eval_dataset_scripts", type=str, nargs='*', default=[],
                        help="指定要评估的数据集及其对应的评估脚本。格式: 'dataset_name:script_name1,script_name2,...' (脚本名称在 eval_script_registry.json 中定义, 仅在evaluate模式下需要)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_output", help="评估结果保存的目录 (仅在evaluate模式下需要)")

    args = parser.parse_args()

    if args.mode == "inference":
        if not args.image_path or not args.prompt:
            parser.error("--image_path 和 --prompt 在 inference 模式下是必需的。")
        main_inference(args.model_path, args.image_path, args.prompt, args.device, args.temperature, args.top_p, args.top_k)
    elif args.mode == "evaluate":
        if not args.dataset_names or not args.eval_dataset_scripts:
            parser.error("--dataset_names 和 --eval_dataset_scripts 在 evaluate 模式下是必需的。")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        evaluate(args.model_path, args.dataset_names, args.eval_script_registry_path, args.eval_dataset_scripts, args.output_dir, args.device, args.temperature, args.top_p, args.top_k)

if __name__ == "__main__":
    main()
