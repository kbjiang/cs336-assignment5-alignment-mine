import os
import torch
from typing import Any
import re
from torch.utils.data import Dataset, DataLoader
import random
import gzip
import json
from transformers import PreTrainedTokenizerBase

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
