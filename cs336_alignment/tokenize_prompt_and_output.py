import torch
from transformers import PreTrainedTokenizer
import torch.nn.functional as F

def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    max_seqlen = 0
    tokens_list = []
    masks = []
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        tokens_prompt = tokenizer.encode(prompt_str) 
        tokens_output = tokenizer.encode(output_str)
        tokens = tokens_prompt + tokens_output
        mask = [False] * len(tokens_prompt) + [True] * len(tokens_output)
        tokens_list.append(tokens)
        masks.append(mask)

        if len(tokens) > max_seqlen:
            max_seqlen = len(tokens)
    
    tokens_padded = [tokens + [tokenizer.pad_token_id] * (max_seqlen - len(tokens)) for tokens in tokens_list]
    masks_padded = [mask + [False] * (max_seqlen - len(mask)) for mask in masks]

    result = {
        "input_ids": torch.tensor(tokens_padded)[:, :-1],
        "labels": torch.tensor(tokens_padded)[:, 1:],
        "response_mask": torch.tensor(masks_padded)[:, 1:],
    }
    return result