from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import logging

logger = logging.getLogger(__name__)


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    prompt_tokens = tokenizer(
        prompt_strs,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    output_tokens = tokenizer(
        output_strs,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    
    response_mask = []
    input_ids = []
    labels = []
    max_prompt_and_output_len = max(len(prompt_tok) + len(output_tok) for prompt_tok, output_tok in zip(prompt_tokens, output_tokens))
    
    for prompt_tok, output_tok in zip(prompt_tokens, output_tokens):
        full_tokens = prompt_tok + output_tok + [tokenizer.pad_token_id] * (max_prompt_and_output_len - len(prompt_tok) - len(output_tok))
        input_ids.append(full_tokens[:-1])
        labels.append(full_tokens[1:])
        response_mask.append(([0] * len(prompt_tok) + [1] * len(output_tok) + [0] * (max_prompt_and_output_len - len(prompt_tok) - len(output_tok)))[1:])
    return {
        "input_ids": torch.stack([torch.tensor(tok) for tok in input_ids]),
        "labels": torch.stack([torch.tensor(tok) for tok in labels]),
        "response_mask": torch.stack([torch.tensor(mask) for mask in response_mask])
    }


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    output_logits = model(input_ids=input_ids).logits # (B, S, V)
    log_softmax = torch.nn.functional.log_softmax(output_logits, dim=-1) # (B, S, V)
    log_probs = torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (B, S)

    if return_token_entropy:
        token_entropy = -torch.sum(
            log_softmax.exp() * log_softmax,
            dim=-1
        )
        return {"log_probs": log_probs, "token_entropy": token_entropy}
    else:
        return {"log_probs": log_probs}


def run_compute_rollout_rewards(
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



def run_compute_group_normalized_rewards(
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
    else:
        raise NotImplementedError(f"Unsupported baseline option {baseline}")
    
    if advantage_normalizer == "std":
        per_prompt_std = torch.std(rewards_reshaped, dim=1, keepdim=True) # (n_prompt, 1)
        raw_rewards = raw_rewards.view(-1, group_size) # (n_prompt, group_size)
        raw_rewards = (raw_rewards / (per_prompt_std + advantage_eps)).view(-1) # (rollout_batch_size,)
    else:
        raise NotImplementedError(f"Unsupported advantage_normalizer option {advantage_normalizer}")

    metadata = {
        "mean_reward": torch.mean(raw_rewards).item(),
        "std_reward": torch.std(raw_rewards).item(),
        "max_reward": torch.max(raw_rewards).item(),
        "min_reward": torch.min(raw_rewards).item()
    }

    return raw_rewards, metadata

def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    if importance_reweighting_method == "none":
        if raw_rewards_or_advantages.dim() == 1:
            raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1) # (batch_size, 1)

        per_token_loss = -policy_log_probs * raw_rewards_or_advantages # (batch_size, sequence_length)
        metadata = {}
        return per_token_loss, metadata
    else:
        raise NotImplementedError(f"Unsupported importance reweighting method {importance_reweighting_method}")


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    if loss_normalization == "sequence":
        loss = (per_token_policy_gradient_loss * mask).sum(dim=1) / mask.sum(dim=1) # (batch_size,)
        loss = loss.mean() # scalar
        return loss
    else:
        raise NotImplementedError(f"Unsupported loss_normalization option {loss_normalization}")


def run_grpo_train_step(
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
    
    mirco_n_group = n_group // gradient_accumulation_steps
    remainder = n_group % gradient_accumulation_steps

    # 构建每个 microbatch 应取的组数列表
    group_counts = [mirco_n_group] * gradient_accumulation_steps
    for i in range(remainder):
        group_counts[i] += 1

    logger.debug(f"group_counts are {group_counts}")

    prompt_and_output_result_dist = run_tokenize_prompt_and_output(
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

        raw_rewards, raw_rewards_metadata = run_compute_rollout_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_response_microbatch,
            repeated_ground_truths=repeated_ground_truths_microbatch,
            device=device
        )
        total_reward += raw_rewards_metadata["mean_total_reward"] * samples_in_microbatch
        format_reward += raw_rewards_metadata["mean_format_reward"] * samples_in_microbatch

        group_normalized_rewards, group_normalized_metadata = run_compute_group_normalized_rewards(
            raw_rewards=raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer
        )
        response_log_probs_result_dict = run_get_response_log_probs(
            model=model,
            input_ids=input_ids_microbatch,
            labels=labels_microbatch,
            return_token_entropy=True
        )

        if "token_entropy" in response_log_probs_result_dict:
            policy_log_probs, token_entropy = response_log_probs_result_dict["log_probs"], response_log_probs_result_dict["token_entropy"]
        else:
            policy_log_probs, token_entropy = response_log_probs_result_dict["log_probs"], None

        per_token_loss, per_token_loss_metadata = run_compute_policy_gradient_loss(
            raw_rewards_or_advantages=group_normalized_rewards,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_logprob_microbatch,
            cliprange=cliprange,
            response_mask=response_masks_microbatch
        )

        logger.debug(f"response_masks_microbatch.sum(dim=1).min() is {response_masks_microbatch.sum(dim=1).min()}")

        loss = run_aggregate_loss_across_microbatch(
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


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
