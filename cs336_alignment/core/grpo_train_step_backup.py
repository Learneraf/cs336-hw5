from typing import Literal, Callable
import torch
from transformers import PreTrainedTokenizerBase
import logging

from .aggregate_loss_across_microbatch import aggregate_loss_across_microbatch
from .compute_group_normalized_rewards import compute_group_normalized_rewards
from .compute_policy_gradient_loss import compute_policy_gradient_loss
from .compute_rollout_rewards import compute_rollout_rewards
from .get_response_log_probs import get_response_log_probs
from .tokenize_prompt_and_output import tokenize_prompt_and_output

logger = logging.getLogger(__name__)

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    device: str | None = "cuda"
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".
        device: str | None = "cuda"
            The device data located.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    return_loss = 0.0
    total_reward = 0.0
    format_reward = 0.0
    rollout_batch_size = len(rollout_responses)
    n_group = rollout_batch_size // group_size # 即n_prompts_per_rollout_batch
    
    # 例如 train_batch_size 为 21， group_size 为 4，gradient_accumulation_steps 为 4
    # 则 n_group = 21， micro_n_group = 5, remainder = 1,
    # group_counts = [6, 5, 5, 5]
    # 意味着第一个 microbatch 取 6 组，第二个 microbatch 取 5 组，...，第八个 microbatch 取 0 组
    micro_n_group = n_group // gradient_accumulation_steps
    remainder = n_group % gradient_accumulation_steps

    # 构建每个 microbatch 应取的组数列表
    group_counts = [micro_n_group] * gradient_accumulation_steps
    for i in range(remainder):
        group_counts[i] += 1

    logger.debug(f"group_counts are {group_counts}")

    prompt_and_output_result_dist = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer
    )
    input_ids, labels, response_masks = prompt_and_output_result_dist["input_ids"], prompt_and_output_result_dist["labels"], prompt_and_output_result_dist["response_mask"]

    group_start_idx = 0
    for gc in group_counts:
        samples_in_microbatch = gc * group_size
        samples_start_idx = group_start_idx * group_size
        samples_end_idx = samples_start_idx + samples_in_microbatch

        input_ids_microbatch = input_ids[samples_start_idx:samples_end_idx].to(device)
        rollout_response_microbatch = rollout_responses[samples_start_idx:samples_end_idx]
        labels_microbatch = labels[samples_start_idx:samples_end_idx].to(device)
        repeated_ground_truths_microbatch = repeated_ground_truths[samples_start_idx:samples_end_idx]
        response_masks_microbatch = response_masks[samples_start_idx:samples_end_idx].to(device)
        old_logprob_microbatch = old_log_probs[samples_start_idx:samples_end_idx] if old_log_probs is not None else None

        raw_rewards, raw_rewards_metadata = compute_rollout_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_response_microbatch,
            repeated_ground_truths=repeated_ground_truths_microbatch,
            device=device
        )
        total_reward += raw_rewards_metadata["mean_total_reward"] * samples_in_microbatch
        format_reward += raw_rewards_metadata["mean_format_reward"] * samples_in_microbatch

        group_normalized_rewards, group_normalized_metadata = compute_group_normalized_rewards(
            raw_rewards=raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer
        )
        response_log_probs_result_dict = get_response_log_probs(
            model=model,
            input_ids=input_ids_microbatch,
            labels=labels_microbatch,
            return_token_entropy=True
        )

        if "token_entropy" in response_log_probs_result_dict:
            policy_log_probs, token_entropy = response_log_probs_result_dict["log_probs"], response_log_probs_result_dict["token_entropy"]
        else:
            policy_log_probs, token_entropy = response_log_probs_result_dict["log_probs"], None

        per_token_loss, per_token_loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=group_normalized_rewards,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_logprob_microbatch,
            cliprange=cliprange,
            response_mask=response_masks_microbatch
        )

        logger.debug(f"response_masks_microbatch.sum(dim=1).min() is {response_masks_microbatch.sum(dim=1).min()}")

        loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=response_masks_microbatch,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant
        ) * len(rollout_response_microbatch) / rollout_batch_size

        loss.backward()
        return_loss += loss

        group_start_idx += gc
    
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()

    metadata = {
        "loss": return_loss.item(),
        "grad_norm": grad_norm.item() if max_grad_norm is not None else None,
        "token_entropy": token_entropy,
        "total_reward": total_reward / rollout_batch_size,
        "format_reward": format_reward / rollout_batch_size
    }
    return return_loss, metadata