from typing import Callable, Union
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_helper_methods import masked_normalize
from typing import Literal

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_respones: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    # use `group_size` to calculate by group
    raw_rewards = torch.tensor(
        [reward_fn(ro, gt)["reward"] for ro, gt in zip(rollout_respones, repeated_ground_truths)]
    ).view(-1, group_size)

    advantages = raw_rewards - raw_rewards.mean(dim=-1, keepdim=True)

    if normalize_by_std:
        advantages = advantages / (raw_rewards.std(dim=-1, keepdim=True) + advantage_eps)

    metadata = {
        "init_raw_rewards_mean": raw_rewards.mean().item(),
        "init_raw_rewards_std": raw_rewards.std().item(),
        "init_raw_rewards_min": raw_rewards.min().item(),
        "init_raw_rewards_max": raw_rewards.max().item(),
    }

    return advantages.flatten(), raw_rewards.flatten(), metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:  
    """
    advantage is on rollout level, therefore the same for every token in same rollout
    Return: per-token loss (batch_size, sequence_length)
    """
    # batch_sz, seq_len = policy_log_probs.shape
    loss_per_tok = -raw_rewards_or_advantages * policy_log_probs
    return loss_per_tok

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Return: per-token loss (batch_size, sequence_length)
    """
    probs_ratio = torch.exp(policy_log_probs - old_log_probs)
    min_lhs = probs_ratio * advantages
    min_rhs = torch.clip(probs_ratio, 1-cliprange, 1+cliprange) * advantages
    loss_per_tok = -1 * torch.min(min_lhs, min_rhs)

    # values will mismatch when clipping happened
    # `detach` to avoid gradient graph retention.
    clipped_token = (min_lhs != min_rhs).detach()
    metadata = {"clipped_token": clipped_token}

    return loss_per_tok, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    wrapper for per-token loss (batch_size, sequence_length)
    """
    if loss_type == "no_baseline":
        loss_per_tok = compute_naive_policy_gradient_loss(
            raw_rewards, policy_log_probs
        )
        return loss_per_tok, {}
    elif loss_type == "reinforce_with_baseline":
        loss_per_tok = compute_naive_policy_gradient_loss(
            advantages, policy_log_probs
        )
        return loss_per_tok, {}
    elif loss_type == "grpo_clip":
        loss_per_tok, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        return loss_per_tok, metadata
    else:
        raise ValueError("Wrong loss type.")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    """when dim=-1, mean is sequence-specific"""
    return torch.sum(tensor*mask, dim=dim) / torch.sum(mask, dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    masked_norm_func: Callable = masked_mean,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_per_tok, metadata = compute_policy_gradient_loss(
        policy_log_probs, 
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )
    # `masked_mean` is sequence specific when dim=-1
    if masked_norm_func == masked_mean:
        loss = masked_norm_func(loss_per_tok, response_mask, dim=-1)
        loss = loss.mean() / gradient_accumulation_steps
    # `masked_normalize` has batch-wise normalize_constant
    elif masked_norm_func == masked_normalize:
        normalize_constant = response_mask.sum(dim=-1).max()
        loss = masked_norm_func(
            loss_per_tok, response_mask, normalize_constant, dim=-1
        )
        loss = loss.mean() / gradient_accumulation_steps
    else:
        raise ValueError(f"Wrong masked normalization: {masked_norm_func}")

    # backward called
    loss.backward()

    metadata["loss_type"] = loss_type
    return loss, metadata