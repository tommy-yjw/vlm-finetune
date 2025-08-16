import torch
from typing import List
import logging
import json # Added for parsing JSON strings
import re # Added for regex in contains_chinese_or_digits if not using bbox_utils directly
from qwen_finetune_project.evaluation import bbox_utils # Import bbox_utils

# 配置一个简单的日志记录器
logger = logging.getLogger(__name__)

def calculate_format_reward(gt_data_list: List[dict], pred_data_list: List[dict]) -> float:
    """
    计算格式奖励。
    - 字段数量是否相符
    - text内容是否相符
    """
    reward = 0.0

    # 字段数量是否相符
    if len(pred_data_list) == len(gt_data_list):
        reward += 0.5
    else:
        reward -= 0.5

    # text内容是否相符 (简单集合比较)
    pred_texts = {item.get('text', '') for item in pred_data_list}
    gt_texts = {item.get('text', '') for item in gt_data_list}
    if pred_texts == gt_texts:
        reward += 0.5
    else:
        reward -= 0.5
    
    return reward

def calculate_pass_rate_reward(gt_data_list: List[dict], pred_data_list: List[dict]) -> float:
    """
    计算通过率奖励 (基于MBR IoU和面积相似度)。
    """
    reward = 0.0

    pred_bboxes_only = [item["bbox"] for item in pred_data_list]
    gt_bboxes_only = [item["bbox"] for item in gt_data_list]

    pred_mbr = bbox_utils.get_min_bounding_rectangle(pred_bboxes_only)
    gt_mbr = bbox_utils.get_min_bounding_rectangle(gt_bboxes_only)

    mbr_iou = bbox_utils.calculate_iou(pred_mbr, gt_mbr)

    if mbr_iou >= 0.7:
        reward += 2.0
    else:
        pred_mbr_area = (pred_mbr[2] - pred_mbr[0]) * (pred_mbr[3] - pred_mbr[1])
        gt_mbr_area = (gt_mbr[2] - gt_mbr[0]) * (gt_mbr[3] - gt_mbr[1])

        if gt_mbr_area > 0:
            area_ratio = pred_mbr_area / gt_mbr_area
            if 0.8 <= area_ratio <= 1.2:
                reward += 1.0
            elif area_ratio < 0.2:
                reward -= 1.0
    
    return reward

def calculate_overlap_reward(gt_data_list: List[dict], pred_data_list: List[dict]) -> float:
    """
    计算重合率奖励。
    - 文字和文字之间不重合
    - 文字在button里面的个数如果相同
    """
    reward = 0.0

    # 文字和文字之间不重合 (惩罚重合)
    pred_text_elements = [item for item in pred_data_list if bbox_utils.contains_chinese_or_digits(item.get("is_what", ""))]
    gt_text_elements = [item for item in gt_data_list if bbox_utils.contains_chinese_or_digits(item.get("is_what", ""))]

    if len(pred_text_elements) > 1:
        pred_text_overlap = bbox_utils.calculate_overlap(pred_text_elements)
        if pred_text_overlap > 0.05: # If there's significant overlap
            reward -= 0.5 # Penalty for text overlap
        else:
            reward += 0.5 # Reward for minimal text overlap
    elif len(pred_text_elements) <= 1 and len(gt_text_elements) <= 1:
        reward += 0.2 # Small reward if both have 0 or 1 text elements (no overlap possible)

    # 文字在button里面的个数如果相同也奖励
    def count_text_in_buttons(data_list):
        count = 0
        text_bboxes = [item for item in data_list if bbox_utils.contains_chinese_or_digits(item.get("is_what", ""))]
        button_bboxes = [item for item in data_list if item.get("is_what", "") == "button"]
        
        for text_item in text_bboxes:
            for button_item in button_bboxes:
                if bbox_utils.is_bbox_inside(text_item["bbox"], button_item["bbox"]):
                    count += 1
                    break # Count each text item only once if it's inside any button
        return count

    pred_text_in_button_count = count_text_in_buttons(pred_data_list)
    gt_text_in_button_count = count_text_in_buttons(gt_data_list)

    if pred_text_in_button_count == gt_text_in_button_count:
        reward += 0.5
    else:
        reward -= 0.5
    
    return reward

def compute_reward(prompts: List[str], responses: List[str], images: List[torch.Tensor]) -> List[float]:
    """
    根据用户定义的标准计算奖励分数。
    奖励包括格式奖励、通过率奖励和重合率奖励。

    Args:
        prompts (List[str]): 包含 Ground Truth (GT) JSON 字符串的列表。
                              格式为 "···json[...]\n```"。
        responses (List[str]): 模型生成的对应回复列表，包含预测的 JSON 字符串。
        images (List[torch.Tensor]): 输入的图像张量列表 (在此函数中未使用，但保留接口)。

    Returns:
        List[float]: 每个回复对应的奖励分数列表。
    """
    rewards = []

    for i in range(len(responses)):
        gt_text_str = prompts[i]
        pred_text_str = responses[i]
        current_reward = 0.0

        # 1. 解析 GT 和 Predicted 数据
        gt_extracted_json = bbox_utils.extract_json_from_text(gt_text_str)
        gt_data_list = bbox_utils.parse_bbox_from_data(gt_extracted_json)

        pred_extracted_json = bbox_utils.extract_json_from_text(pred_text_str)
        pred_data_list = bbox_utils.parse_bbox_from_data(pred_extracted_json)

        # 如果GT或预测数据为空，则给予惩罚或跳过
        if not gt_data_list:
            logger.warning(f"Ground Truth data is empty for prompt {i}. Skipping reward calculation for this item.")
            rewards.append(-5.0) # Significant penalty for missing GT
            continue
        if not pred_data_list:
            logger.warning(f"Predicted data is empty for response {i}. Applying penalty.")
            current_reward -= 2.0 # Penalty for empty prediction
            # Continue to other reward components if possible, or skip
            rewards.append(current_reward)
            continue

        # 计算并累加各项奖励
        current_reward += calculate_format_reward(gt_data_list, pred_data_list)
        current_reward += calculate_pass_rate_reward(gt_data_list, pred_data_list)
        current_reward += calculate_overlap_reward(gt_data_list, pred_data_list)

        rewards.append(current_reward)
    
    return rewards

def get_reward_function():
    """
    获取并返回实际使用的奖励函数。
    """
    return compute_reward
