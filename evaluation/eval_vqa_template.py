import torch
import logging
import os
import json
from tqdm import tqdm
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

logger = logging.getLogger(__name__)

def evaluate(model_engine, processor, eval_dataloader, output_dir, local_rank, temperature=1.0, top_p=1.0, top_k=50):
    """
    评估函数模板。
    
    :param model_engine: DeepSpeed训练后的模型引擎。
    :param processor: Hugging Face处理器。
    :param eval_dataloader: 用于评估的数据加载器。
    :param output_dir: 评估结果的输出目录。
    :param local_rank: 当前进程的本地排名。
    :param temperature: 生成时的温度。
    :param top_p: 生成时的top_p。
    :param top_k: 生成时的top_k。
    :return: 一个包含评估指标的字典。
    """
    if local_rank == 0:
        logger.info("开始执行VQA评估任务 (模板)...")
    
    model_engine.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating (Rank {local_rank})", disable=(local_rank != 0)):
            if not batch:
                if local_rank == 0:
                    logger.warning("Skipping empty batch in evaluation.")
                continue

            batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            pixel_values = batch.pop("pixel_values")
            input_ids = batch.pop("input_ids")
            attention_mask = batch.pop("attention_mask")
            original_conversations = batch.pop("original_conversations", None)

            if original_conversations is None:
                if local_rank == 0:
                    logger.warning("Original conversations not found in batch. Cannot perform GT comparison.")
                continue

            # 假设GT答案在原始对话的GPT回复中，并且是纯文本
            gt_answer = original_conversations[0][1]["value"].strip().lower() # Assuming batch_size=1 and GPT response is at index 1

            if not gt_answer:
                if local_rank == 0:
                    logger.warning(f"Ground Truth answer is empty for a sample. Skipping comparison for this item.")
                continue

            # 生成模型输出
            generated_ids = model_engine.module.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=50, # VQA答案通常较短
                do_sample=True if temperature != 1.0 or top_p != 1.0 or top_k != 0 else False,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=1,
            )
            
            generated_text = processor.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip().lower()
            
            total_samples += 1
            if generated_text == gt_answer:
                total_correct += 1

    # Collect results from all processes
    if model_engine.world_size > 1:
        total_correct_tensor = torch.tensor(total_correct, device=model_engine.device)
        total_samples_tensor = torch.tensor(total_samples, device=model_engine.device)

        torch.distributed.all_reduce(total_correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)

        total_correct = total_correct_tensor.item()
        total_samples = total_samples_tensor.item()

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    
    results = {"accuracy": accuracy}

    if local_rank == 0:
        logger.info("VQA评估任务完成。")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        
        # Save results to a JSON file
        results_file = os.path.join(output_dir, f"eval_vqa_template_results_step_{model_engine.global_steps}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"评估结果已保存到: {results_file}")
    
    return results

if __name__ == "__main__":
    # This block is for standalone testing and needs to be adapted if run directly.
    class MockModelEngine:
        def __init__(self):
            self.device = torch.device("cpu")
            self.world_size = 1
            self.rank = 0
            self.module = None # In a real scenario, this would be the actual model
            self.global_steps = 0 # Mock global steps

        def eval(self):
            pass

        def generate(self, input_ids, pixel_values, attention_mask, pad_token_id, max_new_tokens, do_sample, num_beams):
            # Mock generation for testing
            # This is a very simplified mock. Real generation is complex.
            return torch.zeros((1, input_ids.shape[1] + 10), dtype=torch.long) # Dummy output

    class MockProcessor:
        def __init__(self):
            class MockTokenizer:
                def __init__(self):
                    self.eos_token_id = 0
                def batch_decode(self, ids, skip_special_tokens):
                    return ["mock answer"]

            self.tokenizer = MockTokenizer()
        
        def batch_decode(self, ids, skip_special_tokens=True):
            return ["mock answer"]

    class MockDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                "pixel_values": torch.randn(1, 3, 224, 224),
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10),
                "original_conversations": [item["conversations"]]
            }

    mock_eval_data_for_test = [
        {
            "image_id": "img_001",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is this?"},
                {"from": "gpt", "value": "mock answer"}
            ]
        }
    ]
    mock_dataset = MockDataset(mock_eval_data_for_test)
    mock_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=1)

    mock_model_engine = MockModelEngine()
    mock_processor = MockProcessor()
    mock_output_dir = "./mock_eval_output_vqa"
    os.makedirs(mock_output_dir, exist_ok=True)

    # For standalone testing, you would call:
    # results = evaluate(mock_model_engine, mock_processor, mock_dataloader, mock_output_dir, local_rank=0)
    # print(results)
    pass
