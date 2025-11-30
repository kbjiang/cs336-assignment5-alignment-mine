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

    return evals, solutions_generated

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

def sft_train(cfg, model, vllm_model, optimizer, tokenizer, df_train, df_eval):
    assert cfg.train_steps % cfg.gradient_accumulation_steps == 0
    assert cfg.eval_interval % cfg.gradient_accumulation_steps == 0

    with open(cfg.file_prompt_r1_zero) as f:
        prompt_r1_zero = f.read()

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    for step in tqdm(range(cfg.train_steps), total=cfg.train_steps):
        batch = df_train.sample(cfg.train_batch_size)
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

        # requires gradient
        policy_log_probs.requires_grad_(True)

        # loss.backward() is inside `sft_microbatch_train_step`
        loss, metadata = sft_microbatch_train_step(
            policy_log_probs, response_masks, cfg.gradient_accumulation_steps
        )
        # log loss and metadata...

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Loss {loss.item()}")

        if (step + 1) % cfg.eval_interval == 0:
            load_policy_into_vllm_instance(model, vllm_model)
            
            prompts = get_prompts(
                prompt_r1_zero, df_eval.problem.tolist())
            evals, _ = evaluate_vllm(
                vllm_model, sampling_params, prompts, df_eval.answer.tolist(), r1_zero_reward_fn)

            accuracy = sum([eval["reward"] for eval in evals]) / len(evals)
            accuracy_format = sum([eval["format_reward"] for eval in evals]) / len(evals)
            print(f"Step {step}: accuracy {accuracy}")
            print(f"Step {step}: format {accuracy_format}")

    # save the model weights
    model.save_pretrained(save_directory=cfg.save_dir)
    tokenizer.save_pretrained(save_directory=cfg.save_dir)


if __name__ == "__main__":
    cfg = SFTTrainingConfig()

    vllm_model = init_vllm(cfg.model_id, device=cfg.device_eval, seed=42)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=cfg.device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    optimizer = get_optimizer(cfg, model)

    df_train = pd.read_json(cfg.file_train, lines=True)
    print(f"Num of train samples: {df_train.shape}")
    df_train = df_train.drop_duplicates().reset_index(drop=True)
    print(f"Num of train samples after deduplication: {df_train.shape}")

    df_eval = pd.read_json(cfg.file_eval, lines=True)[:100]

    sft_train(cfg, model, vllm_model, optimizer, tokenizer, df_train, df_eval)