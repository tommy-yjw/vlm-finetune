import torch
from typing import List, Tuple, Dict

def calculate_log_probs(
    model,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor, # Changed from image_tensors
    prompt_len: int = None # Added prompt_len for flexibility
) -> torch.Tensor:
    """
    计算给定序列的对数概率。

    Args:
        model: 用于计算的模型。
        sequences (torch.Tensor): 输入序列的张量 (batch_size, seq_len)。
        sequence_lengths (torch.Tensor): 每个序列的实际长度 (batch_size,)。
        attention_mask (torch.Tensor): 注意力掩码 (batch_size, seq_len)。
        image_tensors (torch.Tensor): 图像张量。

    Returns:
        torch.Tensor: 每个序列的对数概率 (batch_size,)。
    """
    # 确保模型处于评估模式，这会禁用dropout等
    is_training = model.training
    model.eval()

    with torch.no_grad():
        # 获取模型的logits输出
        outputs = model(input_ids=sequences, attention_mask=attention_mask, pixel_values=pixel_values) # Changed images to pixel_values
        logits = outputs.logits

        # 将logits转换为对数概率
        log_probs = torch.log_softmax(logits, dim=-1)

        # 提取目标token的对数概率 (即输入序列中每个token的概率)
        # 如果提供了 prompt_len，则只计算生成部分的对数概率
        if prompt_len is not None:
            target_sequences = sequences[:, prompt_len:].contiguous()
            log_probs = log_probs[:, prompt_len - 1:-1, :].contiguous() # Logits for generated tokens
        else:
            target_sequences = sequences[:, 1:].contiguous()
            log_probs = log_probs[:, :-1, :].contiguous()

        # 使用gather从log_probs中选取目标token的概率
        selected_log_probs = log_probs.gather(dim=-1, index=target_sequences.unsqueeze(-1)).squeeze(-1)

        # 创建一个掩码，只计算非填充部分的对数概率
        # 对于生成部分，我们通常不需要额外的长度掩码，因为我们只关心生成的token
        # 但如果需要考虑padding，可以保留或调整
        mask = (target_sequences != model.config.pad_token_id) # Mask out padding tokens
        selected_log_probs = selected_log_probs * mask

        # 对每个序列的对数概率求和
        sum_log_probs = selected_log_probs.sum(dim=1)

    # 恢复模型原始的训练状态
    if is_training:
        model.train()

    return sum_log_probs

def compute_grpo_loss(
    log_probs_all_samples: torch.Tensor,
    rewards_all_samples: torch.Tensor,
    k_samples: int,
    beta: float
) -> torch.Tensor:
    """
    计算GRPO损失。

    Args:
        log_probs_all_samples (torch.Tensor): 所有k个样本的对数概率 (batch_size * k_samples)。
        rewards_all_samples (torch.Tensor): 所有k个样本的奖励 (batch_size * k_samples)。
        k_samples (int): 每个prompt生成的样本数。
        beta (float): GRPO的温度超参数。

    Returns:
        torch.Tensor: 计算出的GRPO损失（标量）。
    """
    batch_size = log_probs_all_samples.size(0) // k_samples

    # 重塑为 (batch_size, k_samples)
    log_probs = log_probs_all_samples.view(batch_size, k_samples)
    rewards = rewards_all_samples.view(batch_size, k_samples)

    # 计算每个样本对的概率差和奖励差
    # log_probs.unsqueeze(2) -> (batch_size, k_samples, 1)
    # log_probs.unsqueeze(1) -> (batch_size, 1, k_samples)
    log_prob_diff = log_probs.unsqueeze(2) - log_probs.unsqueeze(1) # (batch_size, k_samples, k_samples)
    reward_diff = rewards.unsqueeze(2) - rewards.unsqueeze(1) # (batch_size, k_samples, k_samples)

    # 计算每个样本对的损失
    # 当 reward_diff > 0 时，我们希望 log_prob_diff 也大，反之亦然
    # 使用 sigmoid 将奖励差异映射到 (0, 1) 区间，表示偏好强度
    preference_strength = torch.sigmoid(reward_diff)
    
    # GRPO损失函数
    loss_pairs = -torch.nn.functional.logsigmoid(beta * log_prob_diff) * preference_strength

    # 创建一个掩码，只计算上三角部分（避免重复计算和i=j的情况）
    mask = torch.triu(torch.ones(k_samples, k_samples, device=log_probs.device), diagonal=1).bool()
    loss_pairs = loss_pairs.masked_select(mask)

    # 对所有样本对的损失求平均
    loss = loss_pairs.mean()

    return loss
