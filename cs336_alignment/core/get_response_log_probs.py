import torch

def get_response_log_probs(
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
        # entropy 反应的是一种确定性的指标 
        # entropy = - p * log(p)
        with torch.no_grad():
            token_entropy = -torch.sum(
                log_softmax.exp() * log_softmax,
                dim=-1
            ) # (B, S)
        return {"log_probs": log_probs, "token_entropy": token_entropy}
    else:
        return {"log_probs": log_probs}