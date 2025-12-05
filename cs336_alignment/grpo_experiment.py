import json
from typing import Callable
from os import PathLike
import torch
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
from train_config import GRPOTrainingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from grpo_helper_methods import *
from sft_helper_methods import get_response_log_probs, tokenize_prompt_and_output, log_generations
import wandb


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.   
    """
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value = None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def get_prompts(prompt_template, problems):
    prompts = [prompt_template.replace("{question}", p) for p in problems]
    return prompts

def evaluate_vllm(
    vllm_model: LLM,
    eval_sampling_params: SamplingParams,
    prompts: list[str],
    solutions: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    output_file: str | PathLike | None = None
) -> tuple[list, list]:
    """
    Evaluatea languagemodelon a listof prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    responses = vllm_model.generate(prompts, eval_sampling_params, use_tqdm=False)
    solutions_generated = [opt.outputs[0].text for opt in responses]

    evals = [reward_fn(sol_gen, sol) for sol_gen, sol in zip(solutions_generated, solutions)]

    # Serialize the prompts, solutions, solutions generated, and corresponding evals to disk
    if output_file:
        with open(output_file, 'w') as f:
            for prompt, solution, sol_gen, eval_dict in zip(prompts, solutions, solutions_generated, evals):
                result = {
                    "prompt": prompt,
                    "ground_truth": solution,
                    "generated": sol_gen,
                    "eval": eval_dict
                }
                f.write(json.dumps(result) + '\n')

    return evals, solutions, solutions_generated

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

def sample_rollouts(
    vllm_model: LLM,
    sampling_params: SamplingParams,
    prompt_template_file: str,
    problems: list[str],
    answers: list[str],
    use_tqdm: bool = False
) -> list[dict]:
    """
    Sample group_size rollouts for each question. 
    """
    with open(prompt_template_file) as f:
        prompt_template = f.read()
    prompts = [prompt_template.replace("{question}", p) for p in problems]

    responses = vllm_model.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    answers_generated = [[output.text for output in response.outputs] for response in responses]

    # answers_generated is list[list]
    df = pd.DataFrame(
        {"prompt": prompts, "response": answers_generated, "ground_truth": answers}
    )
    return df.explode(column="response")
    # return df.to_dict(orient="records")

# model -> policy, vllm_model -> old_policy
def grpo_train_loop(cfg, policy, old_policy, optimizer, tokenizer, df_train, df_eval, log_file):
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    assert cfg.rollout_batch_size % cfg.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    assert cfg.train_batch_size >= cfg.group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    # helpful variable
    micro_train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    n_prompts_per_rollout_batch = cfg.rollout_batch_size // cfg.group_size
    n_microbatches_per_rollout_batch = cfg.rollout_batch_size // micro_train_batch_size

    with open(cfg.file_prompt_r1_zero) as f:
        prompt_r1_zero = f.read()

    # Sampling params for eval
    sampling_params_eval = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Sampling params for rollout
    sampling_params_rollout = SamplingParams(
        temperature=1.0, top_p=1.0, stop=["</answer>"],
        include_stop_str_in_output=True,
        max_tokens=1024, 
        min_tokens=4, 
        n=cfg.group_size,
    )

    loss_accumulated = 0.
    micro_step = 0
    for grpo_step in tqdm(range(cfg.n_grpo_steps), total=cfg.n_grpo_steps):
        # Update learning rate based on ei_step (linear schedule from cfg.lr to cfg.lr_fin)
        # lr = cfg.lr - (cfg.lr - cfg.lr_fin) * (ei_step / (cfg.n_ei_steps - 1))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # print(f">>> EI step {ei_step}, learning rate: {lr}")
        # wandb.log({"train/lr": lr, "train_step": step})
        wandb.log({"train/lr": cfg.lr, "train_step": micro_step})

        # 3. Sample a batch of questions        
        rollout_prompts = df_train.sample(n_prompts_per_rollout_batch)

        # 4. Set the old policy model
        load_policy_into_vllm_instance(policy, old_policy)

        # 5. Sample G outputs for reach rollout question
        rollout_batch = sample_rollouts(
            old_policy, sampling_params_rollout, cfg.file_prompt_r1_zero, 
            rollout_prompts.problem.tolist(),
            rollout_prompts.answer.tolist(),
        )
        assert len(rollout_batch) == cfg.rollout_batch_size, "Wrong number of rollout samples"

        # 6 & 7. Compute rewards/advantages for every sampled output
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_respones=rollout_batch.response.tolist(),
            repeated_ground_truths=rollout_batch.ground_truth.tolist(),
            group_size=cfg.group_size,
            advantage_eps=cfg.advantage_eps,
            normalize_by_std=True,
        )
        assert advantages.shape[0] == cfg.rollout_batch_size, "Wrong advantages size"
        assert raw_rewards.shape[0] == cfg.rollout_batch_size, "Wrong raw_rewards size"

        # 8. tokenize the whole batch so that `old_log_probs` can be calculated
        tokenized_dict = tokenize_prompt_and_output(
            rollout_batch.prompt.tolist(),
            rollout_batch.response.tolist(),
            tokenizer,
        )
        input_ids = tokenized_dict["input_ids"]
        labels = tokenized_dict["labels"]
        response_masks = tokenized_dict["response_masks"]

        # 9. get `old_log_probs` is necessary
        # I used `policy` before it's been updated; `old_policy` is vllm model, hard to get log_probs
        # Compute in chunks to avoid OOM
        if cfg.loss_type == "grpo_clip":
            old_log_probs_chunks = []
            chunk_size = 16  # reasonable batch size for inference
            with torch.no_grad():
                for i in range(0, len(input_ids), chunk_size):
                    chunk_input_ids = input_ids[i:i+chunk_size].to(cfg.device_train)
                    chunk_labels = labels[i:i+chunk_size].to(cfg.device_train)
                    chunk_log_probs = get_response_log_probs(
                        policy, chunk_input_ids, chunk_labels, return_token_entropy=False
                    )["log_probs"].cpu()
                    old_log_probs_chunks.append(chunk_log_probs)
            old_log_probs = torch.cat(old_log_probs_chunks, dim=0)
        else:
            old_log_probs = None

        # 10. train
        for _ in range(cfg.epochs_per_rollout_batch):
            # TODO: shuffle the rollout batch
            # rollout_batch = rollout_batch.sample(frac=1)
            for step in range(n_microbatches_per_rollout_batch):
                micro_input_ids = input_ids[
                    step * micro_train_batch_size : (step + 1) * micro_train_batch_size
                ].to(cfg.device_train)
                micro_labels = labels[
                    step * micro_train_batch_size : (step + 1) * micro_train_batch_size
                ].to(cfg.device_train)
                micro_response_masks = response_masks[
                    step * micro_train_batch_size : (step + 1) * micro_train_batch_size
                ].to(cfg.device_train)

                policy_log_probs = get_response_log_probs(
                    policy, micro_input_ids, micro_labels, return_token_entropy=False
                )["log_probs"]

                micro_advantages = advantages[
                    step * micro_train_batch_size : (step + 1) * micro_train_batch_size
                ].to(cfg.device_train)
                micro_raw_rewards = raw_rewards[
                    step * micro_train_batch_size : (step + 1) * micro_train_batch_size
                ].to(cfg.device_train)
                # need to match the shape of `policy_log_probs`
                micro_advantages = micro_advantages.unsqueeze(-1)
                micro_raw_rewards = micro_raw_rewards.unsqueeze(-1)
                assert len(micro_advantages.shape) == len(policy_log_probs.shape), "shape mismatch!"

                if old_log_probs is None:
                    micro_old_log_probs = None
                else:
                    micro_old_log_probs = old_log_probs[
                        step * micro_train_batch_size : (step + 1) * micro_train_batch_size
                    ].to(cfg.device_train)

                # loss.backward() is inside `microbatch_train_step`
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs,
                    micro_response_masks,
                    cfg.gradient_accumulation_steps,
                    cfg.loss_type,
                    micro_raw_rewards,
                    micro_advantages,
                    micro_old_log_probs,
                    cfg.clip_range,
                )

                loss_accumulated += loss.item()

                # take a step
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"Loss {loss_accumulated}")
                    wandb.log({"train/loss": loss_accumulated, "train_step": micro_step + 1})
                    loss_accumulated = 0.

                # do eval regularly
                if step == 0 or (step + 1) % cfg.eval_interval == 0:
                    load_policy_into_vllm_instance(policy, old_policy)
                    
                    prompts = get_prompts(
                        prompt_r1_zero, df_eval.problem.tolist())
                    evals, solutions, solutions_generated = evaluate_vllm(
                        old_policy, sampling_params_eval, prompts, df_eval.answer.tolist(), r1_zero_reward_fn
                    )
                
                    # logging
                    log = log_generations(
                        policy, tokenizer, micro_step, prompts, solutions_generated, solutions, evals
                    )

                    print({k:v for k, v in log.items() if k != "samples"})

                    with open(log_file, "w" if micro_step==0 else "a") as f:
                        f.write(json.dumps(log) + "\n")  
                    
                    wandb.log({
                        "eval/reward": log['reward'], 
                        "eval/reward_format": log['reward_format'], 
                        "eval/reward_answer": log['reward_answer'], 
                        "eval_step": micro_step + 1
                    })

                micro_step += 1

    # save the model weights
    policy.save_pretrained(save_directory=cfg.save_dir)
    tokenizer.save_pretrained(save_directory=cfg.save_dir)
    print(f"GRPO policy saved in {cfg.save_dir}")


if __name__ == "__main__":
    cfg = GRPOTrainingConfig()

    # Initialize wandb
    wandb.init(
        project = cfg.wandb_project,
        name = (
            f"grpo_log_loss{cfg.loss_type_id}_ro{cfg.rollout_batch_size}_G{cfg.group_size}"
            f"_ep{cfg.epochs_per_rollout_batch}_gaccum{cfg.gradient_accumulation_steps}.jsonl"
        ),
        entity=cfg.wandb_entity,
        config=vars(cfg)
    )
    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # load model, tokenizer, optimizer
    old_policy = init_vllm(cfg.model_id, device=cfg.device_eval, seed=42)
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=cfg.device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    optimizer = get_optimizer(cfg, policy)

    # load data
    df_train = pd.read_json(cfg.file_train, lines=True)
    print(f"Num of train samples: {df_train.shape}")
    df_train = df_train.drop_duplicates().reset_index(drop=True)
    print(f"Num of train samples after deduplication: {df_train.shape}")

    df_eval = pd.read_json(cfg.file_eval, lines=True).sample(cfg.eval_sample_size)

    # train
    log_file = (
        f"grpo_log_loss{cfg.loss_type_id}_ro{cfg.rollout_batch_size}_G{cfg.group_size}"
        f"_ep{cfg.epochs_per_rollout_batch}_gaccum{cfg.gradient_accumulation_steps}.jsonl"
    )
    grpo_train_loop(cfg, policy, old_policy, optimizer, tokenizer, df_train, df_eval, log_file)
        