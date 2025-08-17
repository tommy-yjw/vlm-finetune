#!/bin/bash

# 定义模型路径，请根据实际情况修改
# 例如: ./output/qwen_lora_finetune/final_model
MODEL_PATH="./output/qwen_lora_finetune/final_model"

# 定义评估脚本注册配置文件的路径
EVAL_SCRIPT_REGISTRY_PATH="./configs/eval_script_registry.json"

# 定义评估结果保存的目录
OUTPUT_DIR="./evaluation_output"

# 定义设备
DEVICE="cuda" # 或者 "cpu"

# 生成参数
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=50

# ==============================================================================
# 配置评估任务
# 对于不同的测试集，在这里设置不同的评估任务。
# 格式: "数据集名称:评估脚本名称1,评估脚本名称2,..."
# 评估脚本名称应在 configs/eval_script_registry.json 中定义。
# ==============================================================================
EVALUATION_TASKS=(
    "dataset1:bbox_mbr_pass_rate"
    # 如果有其他数据集和评估任务，可以像下面这样添加：
    # "dataset2:vqa_template"
    # "another_dataset:bbox_per_object,vqa_template"
)

# 自动收集所有需要评估的数据集名称
DATASET_NAMES=""
for task_entry in "${EVALUATION_TASKS[@]}"; do
    dataset_name=$(echo "${task_entry}" | cut -d':' -f1)
    if [[ ! " ${DATASET_NAMES} " =~ " ${dataset_name} " ]]; then
        DATASET_NAMES+=" ${dataset_name}"
    fi
done
# 移除前导空格
DATASET_NAMES=$(echo "${DATASET_NAMES}" | xargs)

# 构建 --eval_dataset_scripts 参数列表
EVAL_DATASET_SCRIPTS_ARGS=""
for task_entry in "${EVALUATION_TASKS[@]}"; do
    EVAL_DATASET_SCRIPTS_ARGS+=" --eval_dataset_scripts \"${task_entry}\""
done

echo "开始运行评估..."
echo "模型路径: ${MODEL_PATH}"
echo "评估数据集: ${DATASET_NAMES}"
echo "评估任务配置: ${EVALUATION_TASKS[@]}"
echo "输出目录: ${OUTPUT_DIR}"
echo "设备: ${DEVICE}"
echo "生成参数: Temperature=${TEMPERATURE}, Top_P=${TOP_P}, Top_K=${TOP_K}"

python inference.py \
    --mode evaluate \
    --model_path "${MODEL_PATH}" \
    --dataset_names ${DATASET_NAMES} \
    --eval_script_registry_path "${EVAL_SCRIPT_REGISTRY_PATH}" \
    ${EVAL_DATASET_SCRIPTS_ARGS} \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}"

echo "评估完成。结果已保存到 ${OUTPUT_DIR} (如果评估脚本有保存的话)。"
