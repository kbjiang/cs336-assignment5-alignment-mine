import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch.nn.functional as F
from contextlib import nullcontext
import pandas as pd
import random
from nltk.tokenize import word_tokenize


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
        "response_masks": torch.tensor(masks_padded)[:, 1:],
    }
    return result

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-token entropy"""
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - logsumexp
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, axis=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    inference_mode: bool = False,
) -> dict[str, torch.Tensor]:
    # Move to same device as model
    device = model.device
    input_ids = input_ids.to(device)

    # nullcontext to enable gradient update
    context = torch.inference_mode() if inference_mode else nullcontext()
    
    with context:
        logits = model(input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1).to(device)

        # Learn: advanced indexing; notice the `unsqueeze`
        batch_idx = torch.arange(labels.shape[0]).unsqueeze(1)
        seq_idx = torch.arange(labels.shape[1]).unsqueeze(0)
        log_probs = log_probs[batch_idx, seq_idx, labels]

        result = {"log_probs": log_probs}
        if return_token_entropy:
            result["token_entropy"] = compute_entropy(logits.to(device))
        return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """batch-wise normalize_constant"""
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float=1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -1 * masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant,
        dim=-1,
    )
    # average alone batch_size to reproduce "reduction='mean'" as in PyTorch
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss = loss.mean() / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "loss": loss,
        "policy_log_probs_grad": policy_log_probs.grad
    } 

    return loss, metadata

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    step: int,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    evals: list[dict],
):
    log = {"step": step}
    # evaluation on full dataset
    df = pd.DataFrame(evals)
    ids_format = df[df.format_reward == 1].index.tolist()
    ids_answer = df[df.answer_reward == 1].index.tolist()
    ids_total = df[df.reward == 1].index.tolist()
    log["reward_format"] = round(len(ids_format) / len(df), 3)
    log["reward_answer"] = round(len(ids_answer) / len(df), 3)
    log["reward"] = round(len(ids_total) / len(df), 3)

    res_len = sum([len(word_tokenize(res)) for res in responses])
    res_len_correct = sum([len(word_tokenize(responses[i])) for i in ids_total])
    avg_res_len_correct = res_len_correct / len(ids_total)
    avg_res_len_incorrect = (res_len - res_len_correct) / (len(responses) - len(ids_total))
    log["average_response_length"] = round(res_len / len(responses), 3)
    log["average_response_length_correct"] = round(avg_res_len_correct, 3)
    log["average_response_length_incorrect"] = round(avg_res_len_incorrect, 3)

    # small sample for token entropy
    sample_logs = []
    sample_ids = random.sample(range(len(prompts)), 2)
    # sample_ids = [863, 2714]
    samples_tokenized = tokenize_prompt_and_output(
        [prompts[i] for i in sample_ids],
        [responses[i] for i in sample_ids],
        tokenizer,
    )
    input_ids = samples_tokenized["input_ids"].to(model.device)
    with torch.inference_mode():
        logits = model(input_ids).logits
        entropies = compute_entropy(logits)
    
    for i, id in enumerate(sorted(sample_ids)):
        sample_log = {
            "id": id, 
            "prompt": prompts[id],
            "ground_truth": ground_truths[id],
            "response": responses[id],
            "response_average_token_entropy": torch.mean(entropies, -1)[i].item(),
            "eval": evals[id]
        }
        sample_logs.append(sample_log)
    
    log["samples"] = sample_logs
    
    # Reorder so fields with 'step' or 'epoch' come first
    priority_keys = [k for k in log if 'step' in k.lower() or 'epoch' in k.lower()]
    other_keys = [k for k in log if k not in priority_keys]
    log = {k: log[k] for k in priority_keys + other_keys}
    
    return log