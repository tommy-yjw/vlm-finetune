import torch
import logging
import os
import json
from tqdm import tqdm
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from bbox_utils import extract_json_from_text, parse_bbox_from_data, calculate_iou, calculate_overlap
import concurrent.futures

logger = logging.getLogger(__name__)

def process_sample(pred_text, gt_conversation):
    """
    处理单个样本，计算mIoU和重合度。
    """
    try:
        gt_gpt_value_str = gt_conversation[1]["value"]
        gt_extracted_json = extract_json_from_text(gt_gpt_value_str)
        gt_data_list = parse_bbox_from_data(gt_extracted_json)
        gt_bboxes = [item["bbox"] for item in gt_data_list if "bbox" in item]

        if not gt_bboxes:
            return None

        pred_extracted_json = extract_json_from_text(pred_text)
        pred_data_list = parse_bbox_from_data(pred_extracted_json)
        pred_bboxes = [item["bbox"] for item in pred_data_list if "bbox" in item]

        if not pred_bboxes:
            return None

        image_ious = []
        for pred_box in pred_bboxes:
            max_iou_for_pred = 0.0
            for gt_box in gt_bboxes:
                iou = calculate_iou(pred_box, gt_box)
                max_iou_for_pred = max(max_iou_for_pred, iou)
            image_ious.append(max_iou_for_pred)
        
        miou_per_image = sum(image_ious) / len(image_ious) if image_ious else 0.0
        overlap_per_image = calculate_overlap(pred_data_list)

        return {"miou": miou_per_image, "overlap": overlap_per_image}
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return None

def evaluate(model_engine, processor, eval_dataloader, output_dir, local_rank, temperature=1.0, top_p=1.0, top_k=50):
    """
    评估函数，用于评估模型在bounding box任务上的表现。
    计算每个预测框与GT框的最高IoU，并计算预测框之间的重合度。
    
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
        logger.info("开始执行Bounding Box (Per Object)评估任务...")
    
    model_engine.eval()
    all_predictions = []
    all_ground_truths = []

    # --- 阶段1: 模型推理 ---
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Generating Predictions (Rank {local_rank})", disable=(local_rank != 0)):
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

            generated_ids = model_engine.module.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True if temperature != 1.0 or top_p != 1.0 or top_k != 0 else False,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=1,
            )
            
            generated_text = processor.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            all_predictions.append(generated_text)
            all_ground_truths.append(original_conversations[0])

    # --- 阶段2: 并发后处理 ---
    all_mious = []
    all_overlaps = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_data = {executor.submit(process_sample, pred, gt): (pred, gt) for pred, gt in zip(all_predictions, all_ground_truths)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(all_predictions), desc=f"Processing (Rank {local_rank})", disable=(local_rank != 0)):
            result = future.result()
            if result:
                all_mious.append(result["miou"])
                all_overlaps.append(result["overlap"])

    # --- 阶段3: 结果聚合 ---
    if model_engine.world_size > 1:
        mious_tensor = torch.tensor(all_mious, device=model_engine.device)
        overlaps_tensor = torch.tensor(all_overlaps, device=model_engine.device)
        processed_samples_tensor = torch.tensor(total_processed_samples, device=model_engine.device)

        # Gather all mious and overlaps from all ranks
        gathered_mious = [torch.zeros_like(mious_tensor) for _ in range(model_engine.world_size)]
        gathered_overlaps = [torch.zeros_like(overlaps_tensor) for _ in range(model_engine.world_size)]
        gathered_samples = [torch.zeros_like(processed_samples_tensor) for _ in range(model_engine.world_size)]

        torch.distributed.all_gather(gathered_mious, mious_tensor)
        torch.distributed.all_gather(gathered_overlaps, overlaps_tensor)
        torch.distributed.all_gather(gathered_samples, processed_samples_tensor)

        all_mious = [item.item() for sublist in gathered_mious for item in sublist]
        all_overlaps = [item.item() for sublist in gathered_overlaps for item in sublist]
        total_processed_samples = sum([item.item() for item in gathered_samples])

    # Final metrics calculation
    total_miou = sum(all_mious) / len(all_mious) if all_mious else 0.0
    total_overlap = sum(all_overlaps) / len(all_overlaps) if all_overlaps else 0.0

    results = {
        "total_miou_per_object": total_miou,
        "total_overlap_predicted_bboxes": total_overlap,
    }

    if local_rank == 0:
        logger.info("Bounding Box (Per Object)评估任务完成。")
        logger.info(f"Total mIoU (Per Object): {total_miou:.4f}")
        logger.info(f"Total Overlap (predicted bboxes): {total_overlap:.4f}")
        
        # Save results to a JSON file
        results_file = os.path.join(output_dir, f"eval_bbox_per_object_results_step_{model_engine.global_steps}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"评估结果已保存到: {results_file}")
    
    return results

if __name__ == "__main__":
    # This block is for standalone testing and needs to be adapted if run directly.
    # In the context of train.py, this block is not executed.
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
            # For a true standalone test, you'd need a real model and processor.
            return torch.zeros((1, input_ids.shape[1] + 50), dtype=torch.long) # Dummy output

    class MockProcessor:
        def __init__(self):
            class MockTokenizer:
                def __init__(self):
                    self.eos_token_id = 0
                def batch_decode(self, ids, skip_special_tokens):
                    # This needs to return a plausible mock text for the test to work
                    return ['{"data": [{"bbox": [10, 10, 50, 50], "text": "mock_text", "is_what": "mock_type"}]}']

            self.tokenizer = MockTokenizer()
        
        def batch_decode(self, ids, skip_special_tokens=True):
            return ['{"data": [{"bbox": [10, 10, 50, 50], "text": "mock_text", "is_what": "mock_type"}]}']

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
                {"from": "human", "value": "<image>\n这张图里有什么？"},
                {"from": "gpt", "value": '{"data": [{"bbox": [10, 10, 50, 50], "text": "你好", "is_what": "中文文字"}, {"bbox": [60, 60, 100, 100], "text": "按钮", "is_what": "button"}]}'}
            ]
        }
    ]
    mock_dataset = MockDataset(mock_eval_data_for_test)
    mock_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=1)

    mock_model_engine = MockModelEngine()
    mock_processor = MockProcessor()
    mock_output_dir = "./mock_eval_output_per_object"
    os.makedirs(mock_output_dir, exist_ok=True)

    # For standalone testing, you would call:
    # results = evaluate(mock_model_engine, mock_processor, mock_dataloader, mock_output_dir, local_rank=0)
    # print(results)
    pass
