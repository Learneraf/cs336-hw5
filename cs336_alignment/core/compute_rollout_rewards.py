from typing import Callable
import torch

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    device: str | None = "cuda"
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    raw_rewards = []
    mean_total_reward = 0.0
    mean_format_reward = 0.0

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict: dict[str, float] = reward_fn(response, ground_truth)
        reward = reward_dict["reward"]

        mean_total_reward += reward
        mean_format_reward += reward_dict["format_reward"]

        raw_rewards.append(reward)
    raw_rewards_tensor = torch.tensor(raw_rewards, device=device)

    metadata = {
        "mean_total_reward": mean_total_reward / len(rollout_responses),
        "mean_format_reward": mean_format_reward / len(rollout_responses),
    }
    return raw_rewards_tensor, metadata