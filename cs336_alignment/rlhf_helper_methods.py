import os
import torch
from typing import Any
import re
from torch.utils.data import Dataset, DataLoader
import random
import gzip
import json
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch.nn.functional as F
from contextlib import nullcontext


def parse_mmlu_response(
    response: str,
    mmlu_example: dict[str, Any] | None = None,
):
    chars = re.findall(r'the correct answer is (.)', response.lower())
    for char in chars[::-1]:
        if char in ["a", "b", "c", "d"]:
            return char.upper()
    return None

def parse_gsm8k_response(
    response: str,
):
    # 1. the `replace` takes care of strings like "1,600 grams"
    # 2. the `\.?` optional decimal point for strings like "$8.00"
    # 3. last `?` to avoid confusion with period.
    numbers = re.findall(r'\d+(?:\.\d+)?', response.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None

# file = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/train.jsonl.gz"
def load_jsonl_gz(filepath):
    data = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

alpaca_prompt_file = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/alpaca_sft.prompt"
with open(alpaca_prompt_file, encoding="utf-8") as f:
    ALPACA_SFT_TEMPLATE = f.read().strip() # `strip` to pass test

class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle
        
        # Load the data
        if str(dataset_path).endswith(".gz"):
            data = load_jsonl_gz(dataset_path)
        elif str(dataset_path).endswith(".jsonl"):
            # with open(dataset_path, 'r', encoding='utf-8') as f:
            with open(dataset_path) as f:
                data = [json.loads(line.strip()) for line in f]
        else:
            raise ValueError("Wrong format of dataset file")


        prompts = [ALPACA_SFT_TEMPLATE.format(
            instruction=s["prompt"], response=s["response"]) for s in data]
        if self.shuffle:
            random.shuffle(prompts)
        
        # instruction from assignment, but too slow.
        # prompt = tokenizer.eos_token.join(prompts)
        # input_ids = tokenizer(prompt)["input_ids"]
    
        # Batch tokenize all prompts (fast - parallelized)
        encoded = tokenizer(prompts, add_special_tokens=False)

        # Concatenate token IDs with eos_token between each
        input_ids = []
        for ids in encoded["input_ids"]:
            ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            input_ids.extend(ids)
            # input_ids.append(tokenizer.eos_token_id)
        # input_ids = [tokenizer.bos_token_id] + input_ids

        self.items = []
        for i in range(0, len(input_ids), self.seq_length):
            item = {
                "input_ids": torch.tensor(input_ids[i:i+self.seq_length]),
                "labels": torch.tensor(input_ids[i+1:i+1+self.seq_length])
            }

            # At the end `labels`` is the shorter one
            if len(item["labels"]) != self.seq_length:
                break
            self.items.append(item)
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

# 5.2 look_at_hh
def is_multi_turn(conv: str) -> bool:
    # neither human nor assistant should have more than one msgs
    return conv.count("Human:") > 1 or conv.count("Assistant:") > 1

def reformat(file, datapoint: dict) -> dict:
    def split_conv_(conv: str) -> list[str]:
        human, assistant = [s.strip() for s in conv.split("Assistant:")]
        # human, assistant = conv.split("\n\nAssistant: ")
        # turns out sometimes assistant msg can be empty
        # assert (human and assistant), "The message should not be None."
        return human.split("Human:")[-1].strip(), assistant

    human_chosen, assistant_chosen = split_conv_(datapoint["chosen"])
    human_rejected, assistant_rejected = split_conv_(datapoint["rejected"])
    assert human_chosen == human_rejected, "Human instruction should be same for 'chosen' and 'rejected'."
    return {"file": file.name, "instruction": human_chosen, "chosen": assistant_chosen, "rejected": assistant_rejected}

# 5.3 dpo_loss
def get_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    inference_mode: bool = False,
) -> dict[str, torch.Tensor]:
    # Move to same device as model
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    eos = torch.tensor([[tokenizer.eos_token_id]])
    input_ids = torch.cat([input_ids, eos], dim=-1)

    device = model.device
    input_ids = input_ids.to(device)
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]

    # nullcontext to enable gradient update
    context = torch.inference_mode() if inference_mode else nullcontext()
    
    with context:
        logits = model(input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Learn: advanced indexing; notice the `unsqueeze`
        batch_idx = torch.arange(labels.shape[0]).unsqueeze(1)
        seq_idx = torch.arange(labels.shape[1]).unsqueeze(0)
        log_probs = log_probs[batch_idx, seq_idx, labels]
        return log_probs

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    prompt_chosen = ALPACA_SFT_TEMPLATE.format(instruction=prompt, response=response_chosen)
    prompt_rejected = ALPACA_SFT_TEMPLATE.format(instruction=prompt, response=response_rejected)
    log_probs_lm_chosen = get_log_probs(lm, tokenizer, prompt_chosen)
    log_probs_lmr_chosen = get_log_probs(lm_ref, tokenizer, prompt_chosen)
    log_probs_lm_rejected = get_log_probs(lm, tokenizer, prompt_rejected)
    log_probs_lmr_rejected = get_log_probs(lm_ref, tokenizer, prompt_rejected)
    log_probs_diff_lm = torch.sum(log_probs_lm_chosen) - torch.sum(log_probs_lm_rejected)
    log_probs_diff_lmr = torch.sum(log_probs_lmr_chosen) - torch.sum(log_probs_lmr_rejected)
    return -F.logsigmoid(beta * (log_probs_diff_lm - log_probs_diff_lmr))