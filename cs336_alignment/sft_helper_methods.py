import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
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

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - logsumexp
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, axis=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    # Move to same device as model
    input_ids = input_ids.to(model.device)

    device_data = labels.device
    
    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1).to(device_data)

        # Learn: advanced indexing; notice the `unsqueeze`
        batch_idx = torch.arange(labels.shape[0]).unsqueeze(1)
        seq_idx = torch.arange(labels.shape[1]).unsqueeze(0)
        log_probs = log_probs[batch_idx, seq_idx, labels]

        result = {"log_probs": log_probs}
        if return_token_entropy:
            result["token_entropy"] = compute_entropy(logits.to(device_data))
        return result

def mask_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float=1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    print(policy_log_probs)
    print(response_mask)
    loss = -1 * mask_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant,
        dim=None
    ) / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "loss": loss,
        "policy_log_probs_grad": policy_log_probs.grad
    } 

    return loss, metadata