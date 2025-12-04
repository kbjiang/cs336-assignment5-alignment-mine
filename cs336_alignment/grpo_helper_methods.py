from typing import Callable
import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

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