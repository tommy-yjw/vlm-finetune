import logging
import re
import json

logger = logging.getLogger(__name__)

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)。
    box: [x1, y1, x2, y2]
    """
    # 确保box1和box2是有效的边界框
    if not all(isinstance(coord, (int, float)) for coord in box1) or len(box1) != 4:
        logger.warning(f"Invalid box1 format: {box1}")
        return 0.0
    if not all(isinstance(coord, (int, float)) for coord in box2) or len(box2) != 4:
        logger.warning(f"Invalid box2 format: {box2}")
        return 0.0

    # 获取交集的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 计算交集区域的面积
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集区域的面积
    union_area = box1_area + box2_area - inter_area

    # 避免除以零
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou

def get_min_bounding_rectangle(boxes):
    """
    计算一组边界框的最小外接矩形 (MBR)。
    boxes: [[x1, y1, x2, y2], ...]
    返回: [min_x, min_y, max_x, max_y]
    """
    if not boxes:
        return [0, 0, 0, 0]

    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]

def is_bbox_inside(inner_box, outer_box):
    """
    检查 inner_box 是否完全在 outer_box 内部。
    box: [x1, y1, x2, y2]
    """
    return (inner_box[0] >= outer_box[0] and
            inner_box[1] >= outer_box[1] and
            inner_box[2] <= outer_box[2] and
            inner_box[3] <= outer_box[3])

def calculate_overlap(bbox_data_list):
    """
    计算一组边界框之间的平均重合度，并根据用户反馈排除特定类型的重合。
    bbox_data_list: [{"bbox": [x1, y1, x2, y2], "text": "...", "is_what": "..."}, ...]
    """
    if not bbox_data_list or len(bbox_data_list) < 2:
        return 0.0

    total_iou = 0.0
    pair_count = 0
    num_boxes = len(bbox_data_list)

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            box_info1 = bbox_data_list[i]
            box_info2 = bbox_data_list[j]

            bbox1 = box_info1["bbox"]
            is_what1 = box_info1["is_what"]
            bbox2 = box_info2["bbox"]
            is_what2 = box_info2["is_what"]

            # 排除规则1: 文字bbox在button bbox内部 (现在检测is_what中是否包含中文字符或数字)
            if (contains_chinese_or_digits(is_what1) and is_what2 == "button" and is_bbox_inside(bbox1, bbox2)) or \
               (contains_chinese_or_digits(is_what2) and is_what1 == "button" and is_bbox_inside(bbox2, bbox1)):
                continue
            
            # 排除规则2: button和element之间的overlap
            if (is_what1 == "button" and is_what2 == "element") or \
               (is_what2 == "button" and is_what1 == "element"):
                continue

            total_iou += calculate_iou(bbox1, bbox2)
            pair_count += 1
    
    return total_iou / pair_count if pair_count > 0 else 0.0

def contains_chinese_or_digits(text):
    """
    检查字符串是否包含中文字符或数字。
    """
    # 正则表达式匹配任何中文字符 (Unicode范围) 或任何数字
    return bool(re.search(r'[\u4e00-\u9fff\d]', text))

def extract_json_from_text(text):
    """
    从文本中提取JSON字符串。
    假设JSON格式为 {"data": [{"bbox": [...]}, ...]}
    """
    # 尝试匹配最外层的JSON对象，考虑到可能有多余的文本
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            json_str = match.group(0)
            # 尝试解析JSON，如果失败，可能需要更复杂的正则或解析策略
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e} in text: {text[:100]}...")
            return None
    return None

def parse_bbox_from_data(json_data):
    """
    从提取的JSON数据中解析出所有的bbox、text和is_what字段。
    假设JSON结构为 {"data": [{"bbox": [x1, y1, x2, y2], "text": "...", "is_what": "..."}, ...]}
    或者直接是 [{"bbox": [x1, y1, x2, y2], "text": "...", "is_what": "..."}, ...]
    """
    parsed_data = []
    data_list_to_process = []

    if isinstance(json_data, list):
        # If json_data is already a list, use it directly
        data_list_to_process = json_data
    elif isinstance(json_data, dict) and "data" in json_data and isinstance(json_data["data"], list):
        # If json_data is a dict with a "data" key, use json_data["data"]
        data_list_to_process = json_data["data"]
    else:
        logger.warning(f"Invalid JSON data format for bbox parsing: {json_data}. Expected a list or a dict with 'data' key.")
        return []

    for item in data_list_to_process:
        if "bbox" in item and isinstance(item["bbox"], list) and len(item["bbox"]) == 4 and \
           "text" in item and "is_what" in item:
            parsed_data.append({
                "bbox": item["bbox"],
                "text": item["text"],
                "is_what": item["is_what"]
            })
        else:
            logger.warning(f"Invalid item format found: {item}. Skipping.")
    return parsed_data
