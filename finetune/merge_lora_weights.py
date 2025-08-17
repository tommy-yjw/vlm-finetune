import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

def main():
    parser = argparse.ArgumentParser(description="合并LoRA/QLoRA权重到基础模型")
    parser.add_argument("--base_model_path", type=str, required=True, help="原始基础模型的路径或Hub名称 (e.g., Qwen/Qwen2.5-7B-VL-Chat)")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="训练好的LoRA/QLoRA适配器路径 (e.g., ./output/qwen_qlora/final_model)")
    parser.add_argument("--output_path", type=str, required=True, help="合并后模型的保存路径")
    args = parser.parse_args()

    print(f"正在从 {args.base_model_path} 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"正在从 {args.lora_adapter_path} 加载LoRA适配器...")
    model_to_merge = PeftModel.from_pretrained(base_model, args.lora_adapter_path)

    print("正在合并权重...")
    merged_model = model_to_merge.merge_and_unload()
    print("权重合并完成。")

    print(f"正在将合并后的模型保存到 {args.output_path}...")
    merged_model.save_pretrained(args.output_path)
    processor = AutoProcessor.from_pretrained(args.lora_adapter_path)
    processor.save_pretrained(args.output_path)
    
    print("模型保存成功！")

if __name__ == "__main__":
    main()
