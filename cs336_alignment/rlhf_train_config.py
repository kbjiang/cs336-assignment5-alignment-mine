from dataclasses import dataclass
from typing import Literal

@dataclass
class SFTTrainingConfig:
    model_id = "meta-llama/Llama-3.1-8B"
    file_train = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/train.jsonl.gz"
    file_eval = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/test.jsonl.gz"
    seed: int = 0
    dtype: str = "bfloat16"
    train_batch_size: int = 32
    seq_length: int = 512
    train_epochs: int = 1
    gradient_accumulation_steps: int = 16
    compile: bool = True
    # eval_iterations: int = 100
    eval_steps: int = 16*134
    max_grad_norm: float | None = 1.0
    device_train: str = "cuda:0"
    device_eval: str = "cuda:1"
    lr: float = 2e-5
    lr_fin: float = 1e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str | None = "cs336_assgn5_rlhf_sft"
    wandb_entity: str | None = "kebeijiang"
    log_interval: int = 20
    save_checkpoints: bool = False
    save_dir: str = "rlhf_sft_model"
    num_train_examples: int | None = None

@dataclass
class DPOTrainingConfig:
    model_id = "meta-llama/Llama-3.1-8B"
    seed: int = 0
    dtype: str = "bfloat16"
    train_batch_size: int = 64
    gradient_accumulation_steps: int = 64
    train_epochs: int = 1
    compile: bool = True
    # eval_iterations: int = 100
    eval_steps: int = 1280
    max_grad_norm: float | None = 1.0
    device_train: str = "cuda:0"
    device_ref: str = "cuda:1"
    dpo_loss_beta: float = 0.1
    lr: float = 1e-6
    lr_fin: float = 1e-5
    warmup_ratio: float = 0.03
    wandb_project: str | None = "cs336_assgn5_rlhf_dpo"
    wandb_entity: str | None = "kebeijiang"
    log_interval: int = 20
    save_checkpoints: bool = False
    save_dir: str = "rlhf_dpo_model"
    num_train_examples: int | None = None