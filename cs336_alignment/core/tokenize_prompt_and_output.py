import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
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
        "response_mask": torch.stack([torch.tensor(mask) for mask in response_mask]).bool()
    }
