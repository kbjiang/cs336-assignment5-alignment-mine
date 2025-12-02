import json
from typing import Callable
from os import PathLike
import torch
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
from train_config import SFTTrainingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sft_helper_methods import *
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

def sample_expert_outputs(
    vllm_model: LLM,
    eval_sampling_params: SamplingParams,
    prompt_template_file: str,
    problems: list[str],
    answers: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    use_tqdm: bool = False
) -> list[dict]:
    """
    Sample G outputs for each question, then select `expert` ones with `reward==1`.
    Questions can have mulitple or zero valid outputs; increase G if num of valid is too low.
    """
    with open(prompt_template_file) as f:
        prompt_template = f.read()
    prompts = [prompt_template.replace("{question}", p) for p in problems]

    responses = vllm_model.generate(prompts, eval_sampling_params, use_tqdm=use_tqdm)
    answers_generated = [[output.text for output in response.outputs] for response in responses]
    evals = [[reward_fn(ans_gen, ans) for ans_gen in ans_gens] for ans_gens, ans in zip(answers_generated, answers)]

    answers_generated = [a for ans_gen in answers_generated for a in ans_gen]
    evals = [e for eval in evals for e in eval]
    ids2keep = [i for i in range(len(evals)) if evals[i]["reward"] == 1]

    G = eval_sampling_params.n  # one prompt maps to multiple generated
    return [
        {
            "prompt": prompts[i//G], 
            "response": answers_generated[i],
            "ground_truth": answers[i//G],
        } for i in ids2keep
    ]

def ei_train(cfg, model, vllm_model, optimizer, tokenizer, df_train, df_eval, log_file):
    assert cfg.train_steps % cfg.gradient_accumulation_steps == 0
    assert cfg.eval_interval % cfg.gradient_accumulation_steps == 0

    with open(cfg.file_prompt_r1_zero) as f:
        prompt_r1_zero = f.read()

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    sampling_params_ei = SamplingParams(
        temperature=1.0, top_p=1.0, stop=["</answer>"],
        include_stop_str_in_output=True,
        max_tokens=1024, 
        min_tokens=4, 
        n=cfg.G,
    )

    loss_accumulated = 0.
    step = 0  # train step
    for ei_step in tqdm(range(cfg.n_ei_steps), total=cfg.n_ei_steps):
        # Update learning rate based on ei_step (linear schedule from cfg.lr to cfg.lr_fin)
        lr = cfg.lr - (cfg.lr - cfg.lr_fin) * (ei_step / (cfg.n_ei_steps - 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f">>> EI step {ei_step}, learning rate: {lr}")
        wandb.log({"train/lr": lr, "train_step": step})
        
        ei_batch = df_train.sample(cfg.ei_batch_size)

        # get expert iteration samples
        ei_samples = sample_expert_outputs(
            vllm_model, sampling_params_ei, cfg.file_prompt_r1_zero, 
            ei_batch.problem.tolist(),
            ei_batch.answer.tolist(),
            r1_zero_reward_fn, 
        )
        ei_samples = pd.DataFrame(ei_samples)
        print(f">>> EI step {ei_step}, size of EI samples: {len(ei_samples)}")
        wandb.log({"train/ei_sample": len(ei_samples), "train_step": step})

        for ei_epoch in range(cfg.n_ei_epochs):
            ei_samples = ei_samples.sample(frac=1)
            for i in range(0, len(ei_samples), cfg.train_batch_size):
                batch = ei_samples.iloc[i: i+cfg.train_batch_size]

                tokenized_dict = tokenize_prompt_and_output(
                    batch.prompt.tolist(),
                    batch.response.tolist(),
                    tokenizer,
                )
                input_ids = tokenized_dict["input_ids"].to(cfg.device_train)
                labels = tokenized_dict["labels"].to(cfg.device_train)
                response_masks = tokenized_dict["response_masks"].to(cfg.device_train)

                policy_log_probs = get_response_log_probs(
                    model, input_ids, labels, return_token_entropy=False
                )["log_probs"]

                # loss.backward() is inside `sft_microbatch_train_step`
                # `normalize_constant` removes the impact of different seq-lens btw batches
                loss, metadata = sft_microbatch_train_step(
                    policy_log_probs, response_masks, cfg.gradient_accumulation_steps,
                    normalize_constant=torch.sum(response_masks).item()
                )
                loss_accumulated += loss.item()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"Loss {loss_accumulated}")
                    wandb.log({"train/loss": loss_accumulated, "train_step": step + 1})
                    loss_accumulated = 0.

                if step == 0 or (step + 1) % cfg.eval_interval == 0:
                    load_policy_into_vllm_instance(model, vllm_model)
                    
                    prompts = get_prompts(
                        prompt_r1_zero, df_eval.problem.tolist())
                    evals, solutions, solutions_generated = evaluate_vllm(
                        vllm_model, sampling_params, prompts, df_eval.answer.tolist(), r1_zero_reward_fn)
                
                    # logging
                    log = log_generations(
                        model, tokenizer, step, prompts, solutions_generated, solutions, evals
                    )

                    print({k:v for k, v in log.items() if k != "samples"})

                    with open(log_file, "w" if step==0 else "a") as f:
                        f.write(json.dumps(log) + "\n")  
                    
                    wandb.log({
                        "eval/reward": log['reward'], 
                        "eval/reward_format": log['reward_format'], 
                        "eval/reward_answer": log['reward_answer'], 
                        "eval_step": step + 1
                    })

                step += 1

    # final eval
    print("final eval and logging")
    load_policy_into_vllm_instance(model, vllm_model)
    prompts = get_prompts(
        prompt_r1_zero, df_eval.problem.tolist())
    evals, solutions, solutions_generated = evaluate_vllm(
        vllm_model, sampling_params, prompts, df_eval.answer.tolist(), r1_zero_reward_fn)
    # final logging
    log = log_generations(
        model, tokenizer, step, prompts, solutions_generated, solutions, evals
    )
    print({k:v for k, v in log.items() if k != "samples"})
    with open(log_file, "w" if step==0 else "a") as f:
        f.write(json.dumps(log) + "\n")  
    wandb.log({
        "eval/reward": log['reward'], 
        "eval/reward_format": log['reward_format'], 
        "eval/reward_answer": log['reward_answer'], 
        "eval_step": step + 1
    })

    # save the model weights
    model.save_pretrained(save_directory=cfg.save_dir)
    tokenizer.save_pretrained(save_directory=cfg.save_dir)
    print(f"EI model saved in {cfg.save_dir}")


if __name__ == "__main__":
    cfg = SFTTrainingConfig()

    # Initialize wandb
    wandb.init(
        project = cfg.wandb_project,
        name = f"ei_bs{cfg.ei_batch_size}_G{cfg.G}_ep{cfg.n_ei_epochs}_gaccum{cfg.gradient_accumulation_steps}",
        entity=cfg.wandb_entity,
        config=vars(cfg)
    )
    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # load model, tokenizer, optimizer
    vllm_model = init_vllm(cfg.model_id, device=cfg.device_eval, seed=42)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=cfg.device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    optimizer = get_optimizer(cfg, model)

    # load data
    df_train = pd.read_json(cfg.file_train_ei, lines=True)
    print(f"Num of train samples: {df_train.shape}")
    df_train = df_train.drop_duplicates().reset_index(drop=True)
    print(f"Num of train samples after deduplication: {df_train.shape}")

    df_eval = pd.read_json(cfg.file_eval, lines=True)

    # train
    log_file = f"ei_log_bs{cfg.ei_batch_size}_G{cfg.G}_ep{cfg.n_ei_epochs}_gaccum{cfg.gradient_accumulation_steps}.jsonl"
    ei_train(cfg, model, vllm_model, optimizer, tokenizer, df_train, df_eval, log_file)
        