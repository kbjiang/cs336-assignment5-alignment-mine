from dataclasses import dataclass
from typing import Literal

@dataclass
class SFTTrainingConfig:
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    file_prompt_r1_zero = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/r1_zero.prompt"
    file_train = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/sft.jsonl"
    file_eval = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/validation.jsonl"
    seed: int = 0
    dtype: str = "bfloat16"
    train_batch_size: int = 16
    eval_batch_size: int = 8
    train_steps: int = 800
    gradient_accumulation_steps: int = 10
    compile: bool = True
    # eval_iterations: int = 100
    eval_interval: int = 20
    max_grad_norm: float | None = 1.0
    device_train: str = "cuda:0"
    device_eval: str = "cuda:1"
    lr: float = 1e-5
    lr_fin: float = 1e-5
    warmup_ratio: float = 0.01
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str | None = "cs336_assgn5_sft"
    wandb_entity: str | None = "kebeijiang"
    log_interval: int = 20
    save_checkpoints: bool = False
    save_dir: str = "ei_model"
    num_train_examples: int | None = None
    ei_batch_size: int = 512
    n_ei_steps: int = 5
    n_ei_epochs: int = 4
    file_train_ei = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/train.jsonl"
    G: int = 8

@dataclass
class GRPOTrainingConfig:
    n_grpo_steps: int = 50
    lr: float = 4e-5
    lr_fin: float = 1e-5
    epochs_per_rollout_batch: int = 2  # 1 means on-policy
    train_batch_size: int = 256  # on-policy
    gradient_accumulation_steps: int = 64  # microbatch size is 2, will fit on H100
    eval_interval: int = 64
    eval_sample_frac: float = 0.5
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip",
    # ] = "reinforce_with_baseline"
    # ] = "no_baseline"
    ] = "grpo_clip"
    loss_normalization: Literal[
        "masked_mean", "masked_normalize"
    ] = "masked_mean"
    # ] = "masked_normalize"
    use_std_normalization: bool = True
    rollout_batch_size: int = 256
    group_size: int = 8
    advantage_eps: float = 1e-6
    clip_range: float = 0.1
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    save_dir: str = "grpo_model"
    wandb_project: str | None = "cs336_assgn5_grpo_ablate"
    wandb_entity: str | None = "kebeijiang"
    seed: int = 0
    dtype: str = "bfloat16"
    eval_batch_size: int = 8
    train_steps: int = 800
    compile: bool = True
    # eval_iterations: int = 100
    max_grad_norm: float | None = 1.0
    device_train: str = "cuda:0"
    device_eval: str = "cuda:1"
    warmup_ratio: float = 0.01
    log_interval: int = 20
    save_checkpoints: bool = False
    file_prompt_r1_zero = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/r1_zero.prompt"
    file_train = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/train.jsonl"
    file_eval = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/validation.jsonl"
    model_id = "Qwen/Qwen2.5-Math-1.5B"