import os
import json
import logging
from .custom_dataset import QwenVLDataset, QwenVLDatasetWithAug # 导入现有的数据集类

logger = logging.getLogger(__name__)

_REGISTERED_DATASETS = {}

def register_dataset(name: str, image_root: str, json_path: str, dataset_class: str = "QwenVLDataset"):
    """
    注册一个数据集。
    
    Args:
        name (str): 数据集的唯一名称。
        image_root (str): 图片文件的主路径。
        json_path (str): 包含数据集元数据的JSON文件路径。
        dataset_class (str): 要使用的数据集类名称，默认为"QwenVLDataset"。
                              可选值："QwenVLDataset", "QwenVLDatasetWithAug"。
    """
    if name in _REGISTERED_DATASETS:
        logger.warning(f"数据集 '{name}' 已经被注册，将覆盖现有配置。")
    
    if not os.path.exists(image_root):
        logger.error(f"注册数据集 '{name}' 失败：图片主路径 '{image_root}' 不存在。")
        return False
    
    if not os.path.exists(json_path):
        logger.error(f"注册数据集 '{name}' 失败：JSON文件 '{json_path}' 不存在。")
        return False

    _REGISTERED_DATASETS[name] = {
        "image_root": image_root,
        "json_path": json_path,
        "dataset_class": dataset_class
    }
    logger.info(f"数据集 '{name}' 已成功注册。")
    return True

def get_dataset(name: str, processor, max_length: int = 2048, min_pixels: int = None, max_pixels: int = None):
    """
    根据注册的名称获取并实例化数据集。
    
    Args:
        name (str): 已注册的数据集名称。
        processor: 用于处理文本和图像的处理器。
        max_length (int): 文本序列的最大长度。
        min_pixels (int, optional): 过滤图片的最小像素数。
        max_pixels (int, optional): 过滤图片的最大像素数。
        
    Returns:
        Dataset: 实例化后的数据集对象，如果数据集未注册则返回None。
    """
    config = _REGISTERED_DATASETS.get(name)
    if not config:
        logger.error(f"错误：数据集 '{name}' 未注册。")
        return None

    dataset_class_name = config["dataset_class"]
    image_root = config["image_root"]
    json_path = config["json_path"]

    if dataset_class_name == "QwenVLDataset":
        dataset_cls = QwenVLDataset
    elif dataset_class_name == "QwenVLDatasetWithAug":
        dataset_cls = QwenVLDatasetWithAug
    else:
        logger.error(f"错误：未知的数据集类 '{dataset_class_name}'。")
        return None

    try:
        dataset = dataset_cls(
            json_path=json_path,
            image_root=image_root,
            processor=processor,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        return dataset
    except Exception as e:
        logger.error(f"实例化数据集 '{name}' 时出错: {e}")
        return None

def list_registered_datasets():
    """
    列出所有已注册的数据集名称。
    """
    return list(_REGISTERED_DATASETS.keys())

if __name__ == '__main__':
    # 示例用法
    # 假设你有一个名为 'dummy_processor' 的处理器对象
    # from transformers import AutoProcessor
    # dummy_processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL") 

    # 为了示例运行，我们创建一个假的处理器类
    class DummyTokenizer:
        image_start_tag = "<image_start>"
        image_tag = "<image>"
        image_end_tag = "<image_end>"

    class DummyProcessor:
        tokenizer = DummyTokenizer()
        def __call__(self, text, images, return_tensors, padding, max_length, truncation):
            print(f"Processing text: {text}, images count: {len(images)}")
            # 模拟返回一个张量字典
            return {'input_ids': torch.randn(1, 10), 'attention_mask': torch.randn(1, 10)}

    dummy_processor = DummyProcessor()

    # 确保示例路径存在
    os.makedirs("temp_data/dataset_a/images", exist_ok=True)
    os.makedirs("temp_data/dataset_b/pictures", exist_ok=True)
    
    with open("temp_data/dataset_a/data.json", "w") as f:
        json.dump([{"image": "images/test.jpg", "conversations": [{"from": "human", "value": "hello <image>"}]}], f)
    with open("temp_data/dataset_b/info.json", "w") as f:
        json.dump([{"image": "pictures/another.png", "conversations": [{"from": "human", "value": "hi <image>"}]}], f)

    # 创建一个假的图片文件
    from PIL import Image
    dummy_image = Image.new('RGB', (600, 400), color = 'red')
    dummy_image.save("temp_data/dataset_a/images/test.jpg")
    dummy_image.save("temp_data/dataset_b/pictures/another.png")


    print("--- 注册数据集 ---")
    register_dataset("my_dataset_a", "temp_data/dataset_a", "temp_data/dataset_a/data.json")
    register_dataset("my_dataset_b", "temp_data/dataset_b", "temp_data/dataset_b/info.json", "QwenVLDatasetWithAug")
    register_dataset("non_existent_image_root", "non_existent_path", "temp_data/dataset_a/data.json")
    register_dataset("non_existent_json", "temp_data/dataset_a", "non_existent_file.json")

    print("\n--- 列出已注册的数据集 ---")
    print(list_registered_datasets())

    print("\n--- 获取并使用数据集 ---")
    dataset1 = get_dataset("my_dataset_a", dummy_processor)
    if dataset1:
        print(f"数据集 'my_dataset_a' 大小: {len(dataset1)}")
        if len(dataset1) > 0:
            item = dataset1[0]
            print(f"数据集 'my_dataset_a' 第一个项目键: {item.keys()}")

    dataset2 = get_dataset("my_dataset_b", dummy_processor, min_pixels=100000)
    if dataset2:
        print(f"数据集 'my_dataset_b' 大小: {len(dataset2)}")
        if len(dataset2) > 0:
            item = dataset2[0]
            print(f"数据集 'my_dataset_b' 第一个项目键: {item.keys()}")

    dataset_unregistered = get_dataset("non_existent_dataset", dummy_processor)
    if dataset_unregistered is None:
        print("成功处理未注册数据集的请求。")

    # 清理临时文件
    import shutil
    shutil.rmtree("temp_data")

def load_datasets_from_config(config_path: str):
    """
    从指定的JSON配置文件中加载并注册所有数据集。
    
    Args:
        config_path (str): 包含数据集配置的JSON文件路径。
    """
    if not os.path.exists(config_path):
        logger.error(f"错误：数据集配置文件 '{config_path}' 不存在。")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"错误：无法加载或解析数据集配置文件 {config_path}: {e}")
        return False
        
    datasets_to_register = config_data.get("datasets", {})
    if not datasets_to_register:
        logger.warning(f"数据集配置文件 '{config_path}' 中未找到任何数据集配置。")
        return False

    for name, params in datasets_to_register.items():
        image_root = params.get("image_root")
        json_path = params.get("json_path")
        dataset_class = params.get("dataset_class", "QwenVLDataset")
        
        if not image_root or not json_path:
            logger.error(f"数据集 '{name}' 配置不完整，缺少 'image_root' 或 'json_path'。跳过注册。")
            continue
            
        register_dataset(name, image_root, json_path, dataset_class)
    
    logger.info(f"已从配置文件 '{config_path}' 加载并注册所有数据集。")
    return True
