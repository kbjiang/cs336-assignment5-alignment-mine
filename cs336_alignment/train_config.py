from dataclasses import dataclass

@dataclass
class SFTTrainingConfig:
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    file_prompt_r1_zero = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/cs336_alignment/prompts/r1_zero.prompt"
    file_train = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/sft.jsonl"
    file_eval = "/home/azureuser/localfiles/cs336-assignment5-alignment-mine/data/validation.jsonl"
    num_train_examples: int | None = 128
    seed: int = 0
    dtype: str = "bfloat16"
    train_batch_size: int = 16
    eval_batch_size: int = 16
    train_steps: int = 800
    gradient_accumulation_steps: int = 10
    compile: bool = True
    # eval_iterations: int = 100
    eval_interval: int = 200
    max_grad_norm: float | None = 1.0
    device_train: str = "cuda:0"
    device_eval: str = "cuda:1"
    lr: float = 5e-5
    warmup_ratio: float = 0.01
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str | None = None
    wandb_entity: str | None = None
    log_interval: int = 20
    save_checkpoints: bool = False
    save_dir: str = "sft_model"
