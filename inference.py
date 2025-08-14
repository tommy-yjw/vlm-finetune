import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

def main():
    parser = argparse.ArgumentParser(description="使用微调后的Qwen-VL模型进行推理")
    parser.add_argument("--model_path", type=str, required=True, help="微调后模型的路径 (例如, ./output/qwen_lora_finetune/final_model)")
    parser.add_argument("--image_path", type=str, required=True, help="要进行推理的图片路径")
    parser.add_argument("--prompt", type=str, required=True, help="向模型提出的问题或指令")
    parser.add_argument("--device", type=str, default="cuda", help="运行推理的设备 (cuda or cpu)")
    args = parser.parse_args()

    print(f"正在从 {args.model_path} 加载模型...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("模型加载完成。")

    try:
        image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {args.image_path}")
        return

    query = [{'role': 'user', 'content': f'<image>\n{args.prompt}'}]
    
    text = processor.tokenizer.from_list_format(query)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(args.device)

    print("\n正在生成回答...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    input_token_len = inputs["input_ids"].shape[1]
    response_ids = generated_ids[:, input_token_len:]
    
    response = processor.batch_decode(response_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*20 + " 模型回答 " + "="*20)
    print(response.strip())
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
