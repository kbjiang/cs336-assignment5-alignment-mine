from typing import Callable
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
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
    return advantages.flatten(), raw_rewards.flatten(), None

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:  
    """advantage is on rollout level, therefore the same for every token in same rollout"""
    # batch_sz, seq_len = policy_log_probs.shape
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probs_ratio = torch.exp(policy_log_probs - old_log_probs)
    min_lhs = probs_ratio * advantages
    min_rhs = torch.clip(probs_ratio, 1-cliprange, 1+cliprange) * advantages
    loss = -1 * torch.min(min_lhs, min_rhs)

    # value mismatch when clipping happened
    clipped = min_lhs!=min_rhs
    metadata = {
        "clipped": clipped
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards, policy_log_probs
        )
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            advantages, policy_log_probs
        )
        return loss, {}
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        return loss, metadata
    else:
        raise ValueError("Wrong loss type.")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, 
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )
    batch_size = policy_log_probs.shape[0]
    # `/batch_size` to match `reduction=mean`
    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps

    # backward called
    loss.backward()

    metadata["loss_type"] = loss_type
    return loss, metadata