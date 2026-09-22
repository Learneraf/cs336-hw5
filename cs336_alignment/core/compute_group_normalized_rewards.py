from typing import Literal
import torch

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    rewards_reshaped = raw_rewards.view(-1, group_size) # (n_prompt, group_size)
    if baseline == "mean":
        per_prompt_mean = torch.mean(rewards_reshaped, dim=1, keepdim=True) # (n_prompt, 1)
        raw_rewards = (rewards_reshaped - per_prompt_mean).view(-1) # (rollout_batch_size,)
    elif baseline == "none":
        pass
    else:
        raise NotImplementedError(f"Unsupported baseline option {baseline}")
    
    if advantage_normalizer == "std":
        per_prompt_std = torch.std(rewards_reshaped, dim=1, keepdim=True) # (n_prompt, 1)
        raw_rewards = raw_rewards.view(-1, group_size) # (n_prompt, group_size)
        raw_rewards = (raw_rewards / (per_prompt_std + advantage_eps)).view(-1) # (rollout_batch_size,)
    elif advantage_normalizer == "mean":
        per_prompt_mean = torch.mean(rewards_reshaped, dim=1, keepdim=True) # (n_prompt, 1)
        raw_rewards = raw_rewards.view(-1, group_size) # (n_prompt, group_size)
        raw_rewards = (raw_rewards / (per_prompt_mean + advantage_eps)).view(-1) # (rollout_batch_size,)
    elif advantage_normalizer == "none":
        pass
    else:
        raise NotImplementedError(f"Unsupported advantage_normalizer option {advantage_normalizer}")

    metadata = {
        "mean_reward": torch.mean(raw_rewards).item(),
        "std_reward": torch.std(raw_rewards).item(),
        "max_reward": torch.max(raw_rewards).item(),
        "min_reward": torch.min(raw_rewards).item()
    }

    return raw_rewards, metadata