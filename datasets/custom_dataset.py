import os
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 获取日志记录器
logger = logging.getLogger(__name__)

class QwenVLDataset(Dataset):
    """
    用于Qwen-VL微调的自定义数据集。
    可在初始化时根据图片的像素大小进行过滤。
    """
    def __init__(self, json_path, image_root, processor, max_length=2048, min_pixels=None, max_pixels=None):
        logger.info(f"正在加载数据集: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
        except Exception as e:
            logger.error(f"错误：无法加载或解析JSON文件 {json_path}: {e}")
            self.data = []
            return
            
        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length
        
        if min_pixels is not None or max_pixels is not None:
            self.data = self._filter_by_pixel_size(original_data, min_pixels, max_pixels)
            logger.info(f"原始数据: {len(original_data)} 条, 过滤后: {len(self.data)} 条。")
        else:
            self.data = original_data
        
        logger.info(f"成功加载 {len(self.data)} 条数据。")

    def _filter_by_pixel_size(self, data, min_p, max_p):
        filtered_data = []
        for item in data:
            image_paths_to_check = item.get('images', [])
            if not image_paths_to_check and 'image' in item:
                image_paths_to_check = [item['image']]
            
            if not image_paths_to_check: 
                filtered_data.append(item)
                continue

            all_images_pass_filter = True
            for relative_img_path in image_paths_to_check:
                full_img_path = os.path.join(self.image_root, relative_img_path)
                try:
                    with Image.open(full_img_path) as img:
                        w, h = img.size
                        pixels = w * h
                        if (min_p is not None and pixels < min_p) or \
                           (max_p is not None and pixels > max_p):
                            all_images_pass_filter = False
                            break 
                except Exception:
                    all_images_pass_filter = False 
                    break
            
            if all_images_pass_filter:
                filtered_data.append(item)
        return filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        loaded_images = [] 
        conversation = []

        # 统一处理 image 字段，避免 list 嵌套问题
        image_field = item.get("image", [])
        if isinstance(image_field, str):
            image_paths_in_item = [image_field]
        elif isinstance(image_field, list):
            # 如果是 list，需要 flatten 一层，防止 [[...]] 的情况
            image_paths_in_item = []
            for x in image_field:
                if isinstance(x, str):
                    image_paths_in_item.append(x)
                elif isinstance(x, list):
                    image_paths_in_item.extend([str(xx) for xx in x])
        else:
            image_paths_in_item = []


        for relative_img_path in image_paths_in_item:
            full_img_path = os.path.join(self.image_root, relative_img_path)
            try:
                image = Image.open(full_img_path).convert("RGB")
                loaded_images.append(image)
            except FileNotFoundError:
                logger.warning(f"图片文件未找到: {full_img_path}, 将跳过此图片。")
                continue 

        image_tag_counter = 0 
        for turn in item.get('conversations', []):
            role = 'user' if turn.get('from') == 'human' else 'assistant'
            content = turn.get('value', '')
            
            while '<image>' in content:
                content = content.replace(
                    '<image>',
                    f'{self.processor.tokenizer.image_start_tag}{self.processor.tokenizer.image_tag}{self.processor.tokenizer.image_end_tag}',
                    1
                )
                image_tag_counter += 1
            
            conversation.append({'role': role, 'content': content})

        if not conversation:
            return {}

        try:
            inputs = self.processor(
                text=conversation, images=loaded_images, return_tensors="pt", 
                padding="max_length", max_length=self.max_length, truncation=True
            )
        except Exception as e:
            logger.error(f"处理数据时出错 (索引: {idx}): {e}")
            return {}

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs



class QwenVLDatasetWithAug(QwenVLDataset):
    """
    继承自QwenVLDataset，并增加了视觉数据增强功能。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 定义视觉数据增强流程
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((448, 448), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def __getitem__(self, idx):
        item = self.data[idx]
        loaded_images = [] 
        conversation = []
        
        image_paths_in_item = item.get('images', [])
        if not image_paths_in_item and 'image' in item:
            image_paths_in_item = [item['image']]

        for relative_img_path in image_paths_in_item:
            full_img_path = os.path.join(self.image_root, relative_img_path)
            try:
                image = Image.open(full_img_path).convert("RGB")
                image = self.transform(image) 
                loaded_images.append(image)
            except FileNotFoundError:
                logger.warning(f"图片文件未找到: {full_img_path}, 将跳过此图片。")
                continue

        image_tag_counter = 0
        for turn in item.get('conversations', []):
            role = 'user' if turn.get('from') == 'human' else 'assistant'
            content = turn.get('value', '')
            
            while '<image>' in content:
                content = content.replace('<image>', f'{self.processor.tokenizer.image_start_tag}{self.processor.tokenizer.image_tag}{self.processor.tokenizer.image_end_tag}', 1)
                image_tag_counter += 1
            
            conversation.append({'role': role, 'content': content})

        if not conversation: return {}

        try:
            inputs = self.processor(
                text=conversation, images=loaded_images, return_tensors="pt", 
                padding="max_length", max_length=self.max_length, truncation=True
            )
        except Exception as e:
            logger.error(f"处理数据时出错 (索引: {idx}): {e}")
            return {}

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs
