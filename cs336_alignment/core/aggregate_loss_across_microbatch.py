from typing import Literal
import torch

def aggregate_loss_across_microbatch(
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
    elif loss_normalization == "constant":
        # The default value of normalization_constant is BGL, where B is the batchsize, G is groupsize and L is the maxlength of the whole batch.
        if normalization_constant == None:
            # The per_token_policy_gradient_loss.shape is (BG, L), where BG is named as rollout batchsize, batchsize as simplifed.
            normalization_constant = per_token_policy_gradient_loss.size(0) * per_token_policy_gradient_loss.size(1)
        loss = (per_token_policy_gradient_loss * mask).sum() # scalar
        loss = loss / normalization_constant # scalar
        return loss
    else:
        raise NotImplementedError(f"Unsupported loss_normalization option {loss_normalization}")