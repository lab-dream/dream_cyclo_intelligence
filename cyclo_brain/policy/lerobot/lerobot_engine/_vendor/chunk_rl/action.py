# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Composite action distribution for the chunk manager.

The manager emits a joint action ``(update, index)``:

    * update in {0, 1} : keep the current chunk (0) or switch to the new one (1),
      drawn from ``Bernoulli(sigmoid(update_logit))``;
    * index in {0..L-1} : which new-chunk index to resume from, drawn from
      ``Categorical(softmax(index_logits))``.

The index only matters when updating, so it contributes to the joint log-prob and
entropy only through / weighted by the update decision. Keeping the softmax and
sigmoid out of ``ChunkUpdateDecider.forward`` (it returns raw logits) lets both
rollout sampling and the PPO update share exactly this code path.
"""

import math
from typing import Optional, Tuple

import torch
from torch.distributions import Bernoulli, Categorical


def mask_update_logit(
    update_logit: torch.Tensor,
    force_update: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply action masking to update_logit when chunk update must be forced.

    When force_update is True (e.g. current chunk is exhausted with no remaining
    actions), we mask the keep action (update=0) by setting update_logit to a large
    positive constant (1e4). In float32, sigmoid(1e4) == 1.0, log(1.0) == 0.0,
    and Bernoulli entropy == 0.0.

    Args:
        update_logit: [B, 1] or [B] raw logits for the keep/update decision.
        force_update: Optional boolean tensor indicating per-element forced update.

    Returns:
        update_logit with masked elements replaced by 1e4.
    """
    if force_update is None:
        return update_logit
    mask = torch.as_tensor(force_update, device=update_logit.device, dtype=torch.bool)
    mask = mask.reshape(update_logit.shape)
    return torch.where(mask, torch.full_like(update_logit, 1e4), update_logit)


def is_chunk_exhausted(ctx_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Check if the current chunk is exhausted (no remaining valid actions).

    Args:
        ctx_mask: [..., H + L] boolean mask tensor where the last L elements
                  correspond to the remaining actions in the current chunk.
        chunk_size: L, the chunk size.

    Returns:
        Boolean tensor [...] indicating whether the chunk is exhausted.
    """
    rem_mask = ctx_mask[..., -chunk_size:]
    return ~rem_mask.any(dim=-1)


def build_distributions(
    index_logits: torch.Tensor,
    update_logit: torch.Tensor,
    force_update: Optional[torch.Tensor] = None,
) -> Tuple[Bernoulli, Categorical]:
    """Build the ``Bernoulli(update)`` and ``Categorical(index)`` distributions.

    Args:
        index_logits: [B, L] raw logits over the resume index.
        update_logit: [B, 1] raw logit for the keep/update decision.
        force_update: Optional [B] boolean tensor forcing update=1 (action masking).

    Returns:
        (update_dist over {0,1} with batch shape [B], index_dist over L indices).
    """
    masked_update_logit = mask_update_logit(update_logit, force_update)
    update_dist = Bernoulli(logits=masked_update_logit.squeeze(-1))
    index_dist = Categorical(logits=index_logits)
    return update_dist, index_dist


def _joint_log_prob(
    update_dist: Bernoulli,
    index_dist: Categorical,
    update: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """``log p(update) + [update == 1] * log p(index)`` -> [B]."""
    update = update.float()
    return update_dist.log_prob(update) + update * index_dist.log_prob(index)


def sample_action(
    index_logits: torch.Tensor,
    update_logit: torch.Tensor,
    deterministic: bool = False,
    force_update: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a composite action.

    Args:
        index_logits: [B, L] raw logits.
        update_logit: [B, 1] raw logit.
        deterministic: if True take the mode (argmax / prob > 0.5) instead.
        force_update: Optional [B] boolean tensor forcing update=1 (action masking).

    Returns:
        update [B] float in {0,1}, index [B] long, log_prob [B].
    """
    masked_update_logit = mask_update_logit(update_logit, force_update)
    update_dist, index_dist = build_distributions(index_logits, masked_update_logit)
    if deterministic:
        update = (torch.sigmoid(masked_update_logit.squeeze(-1)) > 0.5).float()
        index = torch.argmax(index_logits, dim=-1)
    else:
        update = update_dist.sample()
        index = index_dist.sample()
    if force_update is not None:
        mask = torch.as_tensor(force_update, device=update.device, dtype=torch.bool).reshape(update.shape)
        update = torch.where(mask, torch.ones_like(update), update)
    log_prob = _joint_log_prob(update_dist, index_dist, update, index)
    return update, index, log_prob


def evaluate_action(
    index_logits: torch.Tensor,
    update_logit: torch.Tensor,
    update: torch.Tensor,
    index: torch.Tensor,
    force_update: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-evaluate stored actions under current logits (PPO update).

    Args:
        index_logits: [B, L] raw logits.
        update_logit: [B, 1] raw logit.
        update: [B] stored update decisions (0/1).
        index: [B] stored resume indices.
        force_update: Optional [B] boolean tensor forcing update=1 (action masking).

    Returns:
        log_prob [B], entropy [B] of the composite distribution.
    """
    masked_update_logit = mask_update_logit(update_logit, force_update)
    update_dist, index_dist = build_distributions(index_logits, masked_update_logit)
    log_prob = _joint_log_prob(update_dist, index_dist, update, index)
    # the index choice only takes effect when updating, so weight its entropy by
    # P(update): the expected contribution to the joint action's entropy.
    p_update = torch.sigmoid(masked_update_logit.squeeze(-1))
    entropy = update_dist.entropy() + p_update * index_dist.entropy()
    return log_prob, entropy


def compute_rl_weight(
    update: int,
    warmup_updates: int,
    conversion_mode: str = "soft",
    schedule: str = "linear",
) -> float:
    """Compute the RL policy selection weight in [0.0, 1.0] for the current update.

    Args:
        update: Current PPO update iteration (1-indexed).
        warmup_updates: Total updates dedicated to the receding-horizon warmup phase.
        conversion_mode: "soft" (gradual annealing) or "hard" (discrete switch at warmup_updates).
        schedule: "linear" or "cosine" annealing for soft conversion.

    Returns:
        float in [0.0, 1.0], where 0.0 is pure receding horizon and 1.0 is pure RL policy.
    """
    if warmup_updates <= 0 or update > warmup_updates:
        return 1.0
    if conversion_mode.lower() == "hard":
        return 0.0

    # Soft conversion schedule
    progress = float(update) / float(warmup_updates)
    progress = min(max(progress, 0.0), 1.0)
    if schedule.lower() == "cosine":
        return float(0.5 * (1.0 - math.cos(math.pi * progress)))
    return progress


def sample_action_with_warmup(
    index_logits: torch.Tensor,
    update_logit: torch.Tensor,
    step: int,
    receding_horizon_k: int = 8,
    rl_weight: float = 1.0,
    force_update: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a composite action with optional receding-horizon warmup mixing.

    Traditional receding horizon execution replans every K steps (update=1, index=0)
    and executes open-loop across the remaining chunk steps (update=0, index=0).

    Args:
        index_logits: [B, L] raw logits.
        update_logit: [B, 1] raw logit.
        step: Current environment rollout step within the episode (0-indexed).
        receding_horizon_k: Replanning interval for traditional receding horizon (default: 8).
        rl_weight: Weight in [0, 1] for choosing RL policy vs receding horizon.
        force_update: Optional [B] boolean tensor forcing update=1 (action masking).
        deterministic: If True, take the mode of the RL policy when selected.

    Returns:
        exec_update [B] float in {0,1}, exec_index [B] long, exec_log_prob [B].
    """
    update_rl, index_rl, log_prob_rl = sample_action(
        index_logits, update_logit, deterministic=deterministic, force_update=force_update
    )
    if rl_weight >= 1.0:
        return update_rl, index_rl, log_prob_rl

    B = index_logits.shape[0]
    device = index_logits.device

    # Traditional receding horizon: replan (update=1, index=0) every K steps, keep (update=0, index=0) otherwise
    rh_update_val = 1.0 if (step % receding_horizon_k == 0) else 0.0
    rh_update = torch.full((B,), rh_update_val, dtype=torch.float32, device=device)
    rh_index = torch.zeros(B, dtype=torch.long, device=device)

    # When chunk is exhausted, receding horizon MUST also update
    if force_update is not None:
        mask = torch.as_tensor(force_update, device=device, dtype=torch.bool).reshape(B)
        rh_update = torch.where(mask, torch.ones_like(rh_update), rh_update)

    update_dist, index_dist = build_distributions(index_logits, update_logit, force_update=force_update)

    if rl_weight <= 0.0:
        exec_update = rh_update
        exec_index = rh_index
    else:
        rl_mask = (torch.rand(B, device=device) < rl_weight)
        exec_update = torch.where(rl_mask, update_rl, rh_update)
        exec_index = torch.where(rl_mask, index_rl, rh_index)

    exec_log_prob = _joint_log_prob(update_dist, index_dist, exec_update, exec_index)
    return exec_update, exec_index, exec_log_prob


