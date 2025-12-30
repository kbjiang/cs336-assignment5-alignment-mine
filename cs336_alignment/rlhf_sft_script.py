import json
from typing import Callable
from os import PathLike
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import wandb
from cs336_alignment.rlhf_helper_methods import *
from rlhf_train_config import SFTTrainingConfig
from torch.nn import CrossEntropyLoss

def get_prompts(prompt_template, problems):
    prompts = [prompt_template.replace("{question}", p) for p in problems]
    return prompts

def get_optimizer(cfg, model):
    # Set up the AdamW optimizer.
    # First, we need to group the parameters that should
    # be decayed and those that shouldn't.
    # In particular, we do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": cfg.weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        fused=True,
    )
    return optimizer

def sft_train(cfg, model, optimizer, scheduler, tokenizer, ds_train, ds_eval):
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0
    # assert cfg.train_steps % cfg.gradient_accumulation_steps == 0
    assert cfg.eval_interval % cfg.gradient_accumulation_steps == 0
    microbatch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps

    loss_fn = CrossEntropyLoss()
    loss_accumulated = 0.
    step = 0
    model.train()
    for epoch in range(cfg.train_epoch):
        # data loader recreated for each epoch
        dl_train = iterate_batches(ds_train, microbatch_size, shuffle=True)
        dl_eval = iterate_batches(ds_eval, microbatch_size*2, shuffle=False)
        for _, (input_ids, labels) in tqdm(enumerate(dl_train), total = len(ds_train)//microbatch_size):
            # Forwad pass
            logits = model(input_ids).logits
            loss = loss_fn(logits, labels) / cfg.gradient_accumulation_steps
            loss_accumulated += loss.item()

            # Backward pass
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # print(f"Loss {loss_accumulated}")
                wandb.log({"train/loss": loss_accumulated, "train/lr": scheduler.get_last_lr()[0], "train_step": step + 1})
                loss_accumulated = 0.

            if step == 0 or (step + 1) % cfg.eval_interval == 0:
                model.eval()
                loss_eval = 0.
                with torch.inference_mode():
                    for input_ids, labels in dl_eval:
                        logits = model(input_ids).logits
                        loss = loss_fn(logits, labels)
                        loss_eval += loss.item()
                wandb.log({"eval/loss": loss_eval/len(ds_eval), "eval_step": step + 1})
                model.train()
            
            step += 1

    # save the model weights
    model.save_pretrained(save_directory=cfg.save_dir)
    tokenizer.save_pretrained(save_directory=cfg.save_dir)
    print(f"SFT model saved in {cfg.save_dir}")


if __name__ == "__main__":
    cfg = SFTTrainingConfig()

    # Initialize wandb
    wandb.init(
        project = cfg.wandb_project,
        # name = f"rlhf_sft_train_",
        # name = "sft_n_train_filtered",
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    optimizer = get_optimizer(cfg, model)

    # load data
    ds_train = SFTDataset(tokenizer, cfg.file_train, cfg.seq_length)
    ds_eval = SFTDataset(tokenizer, cfg.file_eval, cfg.seq_length)
    print(f"Num of train samples: {len(ds_train)}")

    # create scheduler with cosine decay and linear warmup (3% of total steps)
    microbatch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    total_train_steps = cfg.train_epoch * len(ds_train) // cfg.gradient_accumulation_steps
    warmup_steps = int(0.03 * total_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps
    )
    print(f"Total training steps: {total_train_steps}, warmup steps: {warmup_steps}")

    # train
    sft_train(cfg, model, optimizer, scheduler, tokenizer, ds_train, ds_eval)
        