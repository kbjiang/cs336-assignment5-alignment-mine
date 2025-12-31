import json
from typing import Callable
from os import PathLike
import torch
from unittest.mock import patch
from transformers import PreTrainedModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
)
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import wandb
from rlhf_helper_methods import *
from rlhf_train_config import DPOTrainingConfig
from torch.nn import CrossEntropyLoss
import random
from sklearn.model_selection import train_test_split

def get_optimizer(cfg, model):
    # Set up RMSprop optimizer.
    # Note: RMSprop doesn't have built-in weight decay like AdamW,
    # so we just pass all trainable parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.RMSprop(
        params,
        lr=cfg.lr,
    )
    return optimizer

def dpo_train(cfg, model, model_ref, optimizer, scheduler, tokenizer, ds_train, ds_eval, total_train_steps, save_dir):
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0
    # assert cfg.train_steps % cfg.gradient_accumulation_steps == 0
    assert cfg.eval_steps % cfg.gradient_accumulation_steps == 0

    loss_accumulated = 0.
    step = 0
    model.train()
    
    # Progress bar tracks optimizer steps, not batches
    pbar = tqdm(total=total_train_steps)
    
    for epoch in range(cfg.train_epochs):
        # data loader recreated for each epoch
        for d in ds_train:
            # Forwad pass
            loss = compute_per_instance_dpo_loss(
                model, model_ref, tokenizer, cfg.dpo_loss_beta,
                d["instruction"], d["chosen"], d["rejected"],
            ) / cfg.gradient_accumulation_steps
            loss_accumulated += loss.item()

            # Backward pass
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                wandb.log({
                    "train/loss": loss_accumulated,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train_step": step + 1
                })
                loss_accumulated = 0.
                pbar.update(1)

            if step == 0 or (step + 1) == total_train_steps or (step + 1) % cfg.eval_steps == 0:
                model.eval()
                wins = 0
                for d in ds_eval:
                    prompt_chosen = ALPACA_SFT_TEMPLATE.format(
                        instruction=d["instruction"], response=d["chosen"])
                    prompt_rejected = ALPACA_SFT_TEMPLATE.format(
                        instruction=d["instruction"], response=d["rejected"])
                    log_probs_chosen = get_log_probs(
                        model, tokenizer, prompt_chosen, inference_mode=True
                    )    
                    log_probs_rejected = get_log_probs(
                        model, tokenizer, prompt_rejected, inference_mode=True
                    )    
                    if log_probs_chosen.sum() > log_probs_rejected.sum():
                        wins += 1

                eval_result = {
                    "eval/win_rate": wins/len(ds_eval),
                    "eval_step": step + 1
                }
                wandb.log(eval_result)
                print(eval_result)
                model.train()
            
            step += 1

    pbar.close()
    # save the model weights
    model.save_pretrained(save_directory=save_dir)
    tokenizer.save_pretrained(save_directory=save_dir)
    print(f"SFT model saved in {save_dir}")


if __name__ == "__main__":
    cfg = DPOTrainingConfig()

    # Initialize wandb
    wandb.init(
        project = cfg.wandb_project,
        name = f"rlhf_dpo_lr{cfg.lr}_beta{cfg.dpo_loss_beta}",
        entity=cfg.wandb_entity,
        config=vars(cfg)
    )
    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # load model, tokenizer, optimizer
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=cfg.device_train,
    )
    model_ref = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=cfg.device_ref,
    )

    # make sure reference model is in `eval` model
    model_ref.eval()
    for param in model_ref.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    # # When batching with `padding=True`, need to set `tokenizer.pad_token`
    # tokenizer.pad_token = tokenizer.eos_token
    optimizer = get_optimizer(cfg, model)

    # load data
    data_dir = Path(__file__).parent.parent / "data"
    files = list(data_dir.glob("h*.jsonl.gz"))

    ds = []
    for file in files:
        data = load_jsonl_gz(file)
        for i, d in enumerate(data):
            if any([is_multi_turn(v) for v in d.values()]):
                continue
            d = reformat(file, d)
            ds.append(d)

    ds_train, ds_eval = train_test_split(ds, test_size=1000, random_state=42, shuffle=True)
    print(f"Num of train samples: {len(ds_train)}")
    print(f"Num of eval samples: {len(ds_eval)}")
    print("Random sample of train dataset:")
    print(json.dumps(ds_train[0], indent=2))

    # create scheduler with cosine decay and linear warmup (3% of total steps)
    total_train_steps = cfg.train_epochs * len(ds_train) // cfg.train_batch_size
    # warmup_steps = int(cfg.warmup_ratio * total_train_steps)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_train_steps
    # )
    # print(f"Total training steps: {total_train_steps}, warmup steps: {warmup_steps}")
    scheduler = get_constant_schedule(optimizer)
    print(f"Total training steps: {total_train_steps}")

    # train
    save_dir = Path(cfg.save_dir) / f"rlhf_dpo_lr{cfg.lr}_beta{cfg.dpo_loss_beta}"
    dpo_train(cfg, model, model_ref, optimizer, scheduler, tokenizer, ds_train, ds_eval, total_train_steps, save_dir)
        