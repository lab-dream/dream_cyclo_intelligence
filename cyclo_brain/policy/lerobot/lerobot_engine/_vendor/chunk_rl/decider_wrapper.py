# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""Stateful API wrapper for ChunkUpdateDecider.

Provides a clean, intuitive API for deploying ChunkRL policies on real robots
or in external evaluation loops (e.g. LeRobot, Gymnasium, ROS nodes) without
requiring the full RL training environment infrastructure.

Usage:
    # 1. Initialize decider
    decider = ChunkDeciderWrapper.from_checkpoint("/path/to/checkpoint.pt", device="cuda")

    # 2. Episode loop
    decider.reset()
    while not done:
        new_chunk = base_policy.predict(obs)  # [L, action_dim]
        # Option A: Two-step API
        update_logit, index_logits = decider.forward(new_chunk, state_ref=obs)
        action, update, index = decider.get_action(update_logit, index_logits, deterministic=True)

        # Option B: One-step convenience API
        # action, update, index = decider.step(new_chunk, state_ref=obs, deterministic=True)

        obs, reward, done, info = env.step(action)
"""

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .action import sample_action
from .model import ChunkUpdateConfig, ChunkUpdateDecider
from .transform import TF_mat, absolute_actions_to_relative, relative_actions_to_absolute


class ChunkDeciderWrapper:
    """Stateful deployment wrapper around a trained :class:`ChunkUpdateDecider`."""

    def __init__(
        self,
        actor: ChunkUpdateDecider,
        past_horizon: Optional[int] = None,
        relative_action: bool = False,
        is_gym: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            actor: A trained :class:`ChunkUpdateDecider` instance.
            past_horizon: Number of past actions to retain for context (H). Defaults to actor.config.H.
            relative_action: True if base policy outputs relative/delta actions.
            is_gym: True for vector/gym delta environments (False for robomimic pose TF).
            device: Torch device to run actor inference on.
        """
        self.actor = actor
        self.config = actor.config
        self.past_horizon = past_horizon if past_horizon is not None else getattr(self.config, "H", 8)
        self.relative_action = relative_action
        self.is_gym = is_gym

        if device is None:
            # Infer device from actor parameters
            try:
                self.device = next(actor.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            self.actor = self.actor.to(self.device)

        self.actor.eval()

        # Stateful episode variables
        self.current_chunk: Optional[np.ndarray] = None
        self.new_chunk: Optional[np.ndarray] = None
        self.chunk_step: int = 0
        self.past_actions: Deque[np.ndarray] = deque(maxlen=self.past_horizon)

        # Cache last logits for get_action()
        self.last_update_logit: Optional[torch.Tensor] = None
        self.last_index_logits: Optional[torch.Tensor] = None

        self.reset()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        past_horizon: Optional[int] = None,
        relative_action: bool = False,
        is_gym: bool = False,
    ) -> "ChunkDeciderWrapper":
        """Load actor from checkpoint and instantiate wrapper.

        Args:
            checkpoint_path: Path to PyTorch checkpoint containing 'actor' weights.
            device: Target device for actor model.
            past_horizon: Context past horizon H.
            relative_action: Whether base policy actions are relative.
            is_gym: Whether environment uses Gym vector actions.

        Returns:
            Configured and initialized :class:`ChunkDeciderWrapper`.
        """
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        saved_config = checkpoint.get("actor_config", checkpoint.get("config", {}))
        if hasattr(saved_config, "__dict__"):
            saved_config = saved_config.__dict__

        config_fields = set(ChunkUpdateConfig.__annotations__.keys())
        # Provide safe default for index_decay if missing in older checkpoints
        if "index_decay" not in saved_config:
            saved_config = {**saved_config, "index_decay": 0.0}

        filtered_config = {k: saved_config[k] for k in config_fields if k in saved_config}
        config = ChunkUpdateConfig(**filtered_config)

        actor = ChunkUpdateDecider(config).to(device)
        actor.load_state_dict(checkpoint["actor"], strict=True)
        actor.eval()

        return cls(
            actor=actor,
            past_horizon=past_horizon,
            relative_action=relative_action,
            is_gym=is_gym,
            device=device,
        )

    def reset(self) -> None:
        """Reset internal episode state.

        Must be called at the beginning of each episode.
        """
        self.current_chunk = None
        self.new_chunk = None
        self.chunk_step = 0
        self.past_actions.clear()
        self.last_update_logit = None
        self.last_index_logits = None

    def _extract_state_ref(self, state_ref: Any) -> Any:
        """Extract transform or reference vector from observation dictionary or object."""
        if state_ref is None:
            return None
        if isinstance(state_ref, TF_mat):
            return state_ref
        if isinstance(state_ref, np.ndarray) and state_ref.shape == (4, 4):
            return TF_mat(state_ref)
        if isinstance(state_ref, (tuple, list)) and len(state_ref) == 2:
            return TF_mat.from_vectors(
                np.asarray(state_ref[0], dtype=np.float32),
                np.asarray(state_ref[1], dtype=np.float32),
            )
        if isinstance(state_ref, dict):
            # 1. Standard robosuite / robomimic
            if "robot0_eef_pos" in state_ref and "robot0_eef_quat" in state_ref:
                return TF_mat.from_vectors(
                    np.asarray(state_ref["robot0_eef_pos"], dtype=np.float32),
                    np.asarray(state_ref["robot0_eef_quat"], dtype=np.float32),
                )
            # 2. General robot state keys (e.g. eef_pos / eef_quat)
            if "eef_pos" in state_ref and "eef_quat" in state_ref:
                return TF_mat.from_vectors(
                    np.asarray(state_ref["eef_pos"], dtype=np.float32),
                    np.asarray(state_ref["eef_quat"], dtype=np.float32),
                )
            if "pos" in state_ref and "quat" in state_ref:
                return TF_mat.from_vectors(
                    np.asarray(state_ref["pos"], dtype=np.float32),
                    np.asarray(state_ref["quat"], dtype=np.float32),
                )
            if "robot0_eef_pos" in state_ref:
                return np.asarray(state_ref["robot0_eef_pos"], dtype=np.float32).reshape(-1)
            if "eef_pos" in state_ref:
                return np.asarray(state_ref["eef_pos"], dtype=np.float32).reshape(-1)
        return state_ref

    def _build_context(
        self, new_chunk: np.ndarray, state_ref: Any = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (new_chunk_obs, ctx, ctx_mask) for actor network.

        Args:
            new_chunk: (L, action_dim) raw action chunk.
            state_ref: Current robot state reference (e.g. TF_mat, raw_obs, or vector).

        Returns:
            Tuple of:
                new_chunk_obs: (L, action_dim) model input chunk (relative if needed).
                ctx: (H + L, action_dim) context matrix.
                ctx_mask: (H + L,) boolean mask.
        """
        H = self.past_horizon
        L = new_chunk.shape[0]
        A = new_chunk.shape[1]
        state_ref = self._extract_state_ref(state_ref)

        if self.is_gym:
            ref_vec = (
                np.asarray(state_ref, dtype=np.float32).reshape(-1)
                if state_ref is not None
                else None
            )
            if not self.relative_action or ref_vec is None:
                new_chunk_obs = new_chunk.copy()
            else:
                new_chunk_obs = new_chunk - ref_vec

            # 1. Past actions
            past = np.zeros((H, A), dtype=np.float32)
            past_valid = np.zeros(H, dtype=bool)
            k = len(self.past_actions)
            if k > 0:
                past_arr = np.stack(list(self.past_actions), axis=0)
                if self.relative_action and ref_vec is not None:
                    past_arr = past_arr - ref_vec
                past[H - k :] = past_arr
                past_valid[H - k :] = True

            # 2. Remaining current chunk
            s = min(self.chunk_step, L)
            rem = np.zeros((L, A), dtype=np.float32)
            rem_valid = np.zeros(L, dtype=bool)
            if self.current_chunk is not None:
                remaining = self.current_chunk[s:]
                r = remaining.shape[0]
                if r > 0:
                    rem_arr = remaining.copy()
                    if self.relative_action and ref_vec is not None:
                        rem_arr = rem_arr - ref_vec
                    rem[:r] = rem_arr
                    rem_valid[:r] = True

            ctx = np.concatenate([past, rem], axis=0)
            ctx_mask = np.concatenate([past_valid, rem_valid])
            return new_chunk_obs.astype(np.float32), ctx, ctx_mask

        # Robomimic / 6D Pose domain
        if not self.relative_action and isinstance(state_ref, TF_mat):
            new_chunk_obs = absolute_actions_to_relative(new_chunk, state_ref)
        else:
            # If relative_action is True, new_chunk is ALREADY relative to the current state_ref
            new_chunk_obs = new_chunk.copy()

        # 1. Past actions (stored in absolute world coordinates)
        past = np.zeros((H, A), dtype=np.float32)
        past_valid = np.zeros(H, dtype=bool)
        k = len(self.past_actions)
        if k > 0:
            past_arr = np.stack(list(self.past_actions), axis=0)
            if isinstance(state_ref, TF_mat):
                past_arr = absolute_actions_to_relative(past_arr, state_ref)
            past[H - k :] = past_arr
            past_valid[H - k :] = True

        # 2. Remaining current chunk (stored in absolute world coordinates)
        s = min(self.chunk_step, L)
        rem = np.zeros((L, A), dtype=np.float32)
        rem_valid = np.zeros(L, dtype=bool)
        if self.current_chunk is not None:
            remaining = self.current_chunk[s:]
            r = remaining.shape[0]
            if r > 0:
                if isinstance(state_ref, TF_mat):
                    rem_arr = absolute_actions_to_relative(remaining, state_ref)
                else:
                    rem_arr = remaining.copy()
                rem[:r] = rem_arr
                rem_valid[:r] = True

        ctx = np.concatenate([past, rem], axis=0)
        ctx_mask = np.concatenate([past_valid, rem_valid])
        return new_chunk_obs.astype(np.float32), ctx, ctx_mask

    def forward(
        self,
        new_chunk: Union[np.ndarray, torch.Tensor],
        state_ref: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process newly predicted chunk and return decision logits.

        1. Stores self.new_chunk = new_chunk.
        2. Constructs context vector and mask from self.current_chunk, self.past_actions, self.chunk_step.
        3. Computes update_logit and index_logits from actor model.

        Args:
            new_chunk: (L, action_dim) newly predicted action chunk.
            state_ref: Optional robot state reference (EEF TF_mat, pose vector, or raw_obs dict).

        Returns:
            Tuple of:
                update_logit: Tensor [1, 1] raw logit for switch/keep decision.
                index_logits: Tensor [1, L] raw logits for start index selection.
        """
        if isinstance(new_chunk, torch.Tensor):
            new_chunk_np = new_chunk.detach().cpu().numpy().astype(np.float32)
        else:
            new_chunk_np = np.asarray(new_chunk, dtype=np.float32)

        self.new_chunk = new_chunk_np

        # Store extracted state_ref for get_action
        self.last_state_ref = self._extract_state_ref(state_ref)

        # First step handling: initialize current_chunk if not set
        if self.current_chunk is None:
            if self.relative_action:
                if self.is_gym:
                    ref_vec = (
                        np.asarray(self.last_state_ref, dtype=np.float32).reshape(-1)
                        if self.last_state_ref is not None
                        else None
                    )
                    self.current_chunk = (
                        (self.new_chunk + ref_vec)
                        if ref_vec is not None
                        else self.new_chunk.copy()
                    )
                elif isinstance(self.last_state_ref, TF_mat):
                    self.current_chunk = relative_actions_to_absolute(
                        self.new_chunk, self.last_state_ref
                    )
                else:
                    self.current_chunk = self.new_chunk.copy()
            else:
                self.current_chunk = self.new_chunk.copy()
            self.chunk_step = 0

        # Build context vector and mask
        new_chunk_obs, ctx, ctx_mask = self._build_context(self.new_chunk, state_ref=self.last_state_ref)

        # Prepare tensors for actor
        t_chunk = torch.as_tensor(new_chunk_obs, device=self.device).unsqueeze(0)
        t_ctx = torch.as_tensor(ctx, device=self.device).unsqueeze(0)
        t_mask = torch.as_tensor(ctx_mask, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            output = self.actor(t_chunk, t_ctx, t_mask)

        self.last_update_logit = output["update_logit"]
        self.last_index_logits = output["index_logits"]

        return self.last_update_logit, self.last_index_logits

    def get_action(
        self,
        update_logit: Optional[torch.Tensor] = None,
        index_logit: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        force_update: Optional[bool] = None,
    ) -> Tuple[np.ndarray, bool, int]:
        """Sample decision from logits, update internal chunk state, and return action.

        1. Determines update (keep/switch) and index via argmax or sampling.
        2. If chunk_step >= L, forces update = True.
        3. If update == True, converts new_chunk to absolute and updates current_chunk.
        4. Extracts action in the base policy representation.
        5. Updates past_actions and increments self.chunk_step += 1.

        Args:
            update_logit: Optional [1, 1] Tensor. If None, uses cached logit from forward().
            index_logit: Optional [1, L] Tensor. If None, uses cached logits from forward().
            deterministic: True for mode (greedy/argmax), False for stochastic sampling.
            force_update: Optional bool override. If None, auto-detected from (chunk_step >= L).

        Returns:
            Tuple of:
                action: (action_dim,) single action to execute in environment.
                update: bool indicating whether chunk switch occurred.
                index: int selected index.
        """
        if update_logit is None:
            update_logit = self.last_update_logit
        if index_logit is None:
            index_logit = self.last_index_logits

        if update_logit is None or index_logit is None:
            raise RuntimeError("Logits not available. Call forward(new_chunk) before get_action().")

        L = self.config.L

        # Check if existing chunk is exhausted
        if force_update is None:
            force_update = (self.chunk_step >= L)

        force_update_t = torch.as_tensor([force_update], device=update_logit.device)

        with torch.inference_mode():
            update_t, index_t, _ = sample_action(
                index_logit,
                update_logit,
                deterministic=deterministic,
                force_update=force_update_t,
            )

        update = bool(update_t.squeeze().item() > 0.5)
        index = int(index_t.squeeze().item())

        if force_update:
            update = True

        # Update internal chunk and step if switched or uninitialized
        if update or self.current_chunk is None:
            if self.new_chunk is not None:
                if self.relative_action:
                    if self.is_gym:
                        ref_vec = (
                            np.asarray(self.last_state_ref, dtype=np.float32).reshape(-1)
                            if self.last_state_ref is not None
                            else None
                        )
                        self.current_chunk = (
                            (self.new_chunk + ref_vec)
                            if ref_vec is not None
                            else self.new_chunk.copy()
                        )
                    elif isinstance(self.last_state_ref, TF_mat):
                        self.current_chunk = relative_actions_to_absolute(
                            self.new_chunk, self.last_state_ref
                        )
                    else:
                        self.current_chunk = self.new_chunk.copy()
                else:
                    self.current_chunk = self.new_chunk.copy()
            self.chunk_step = index

        # Extract executed action
        exec_idx = min(self.chunk_step, L - 1)
        action_abs = self.current_chunk[exec_idx].copy()

        # Update past actions history (always in absolute world frame)
        self.past_actions.append(action_abs.copy())

        # If base policy outputs relative actions, return action relative to current state_ref
        if self.relative_action:
            if self.is_gym:
                ref_vec = (
                    np.asarray(self.last_state_ref, dtype=np.float32).reshape(-1)
                    if self.last_state_ref is not None
                    else None
                )
                action = (action_abs - ref_vec) if ref_vec is not None else action_abs
            elif isinstance(self.last_state_ref, TF_mat):
                action = absolute_actions_to_relative(
                    action_abs[np.newaxis, :], self.last_state_ref
                )[0]
            else:
                action = action_abs
        else:
            action = action_abs

        # Increment chunk step
        self.chunk_step += 1

        return action, update, index

    def step(
        self,
        new_chunk: Union[np.ndarray, torch.Tensor],
        state_ref: Any = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, bool, int]:
        """Convenience all-in-one method: forward(new_chunk) + get_action().

        Args:
            new_chunk: (L, action_dim) newly predicted action chunk.
            state_ref: Optional robot state reference (e.g. EEF pose or raw_obs).
            deterministic: True for greedy execution, False for sampling.

        Returns:
            Tuple of:
                action: (action_dim,) action to execute.
                update: bool whether chunk was switched.
                index: int executed index.
        """
        u_logit, i_logit = self.forward(new_chunk, state_ref=state_ref)
        return self.get_action(u_logit, i_logit, deterministic=deterministic)
