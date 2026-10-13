"""Independent arm deciders with shared bimanual IK downstream.

Dream actions use row-based rot6d; ChunkRL uses column-based rot6d. Each
decider stores its active trajectory in absolute coordinates and re-expresses
it against the newest measured TCP, including when it keeps an older chunk.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ._vendor.chunk_rl import ChunkDeciderWrapper
from ._vendor.chunk_rl.transform import TF_mat
from ._vendor.umi.pose_util import mat_to_rot6d, pose_to_mat, rot6d_to_mat


DEFAULT_CHECKPOINT = "/workspace/model/chunkrl/robomimic_multitask_v5/best_eval.pt"


class BimanualChunkRL:
    """Share frozen actor weights, keep each arm's execution state separate."""

    def __init__(self, checkpoint: str, device="cpu") -> None:
        path = Path(checkpoint or DEFAULT_CHECKPOINT).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ChunkRL checkpoint not found: {path}")
        left = ChunkDeciderWrapper.from_checkpoint(path, device=device, relative_action=True)
        if left.config.d_a != 10:
            raise ValueError("Per-arm ChunkRL requires d_a=10 (xyz, column rot6d, gripper).")
        self.arms = [left, ChunkDeciderWrapper(left.actor, relative_action=True, device=device)]
        self.length = left.config.L
        self.last_decisions = []

    def reset(self) -> None:
        for arm in self.arms:
            arm.reset()
        self.last_decisions = []

    def snapshot(self):
        # Do not duplicate actor weights. This also makes failed IK transactional.
        fields = ("current_chunk", "new_chunk", "chunk_step", "past_actions",
                  "last_update_logit", "last_index_logits", "last_state_ref")
        return ([{name: copy.deepcopy(getattr(arm, name, None)) for name in fields}
                 for arm in self.arms], copy.deepcopy(self.last_decisions))

    def restore(self, snapshot) -> None:
        states, self.last_decisions = snapshot
        for arm, state in zip(self.arms, states, strict=True):
            for name, value in state.items():
                setattr(arm, name, value)

    def select(self, chunk: np.ndarray, env_obs: dict) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (self.length, 20) or not np.isfinite(chunk).all():
            raise ValueError(f"ChunkRL expects finite ({self.length}, 20) relative EEF actions, got {chunk.shape}.")
        selected, decisions = [], []
        snapshot = self.snapshot()
        try:
            for side, arm in enumerate(self.arms):
                proposal = chunk[:, side * 10 : (side + 1) * 10].copy()
                rotations = rot6d_to_mat(proposal[:, 3:9])
                if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
                    raise ValueError("ChunkRL received degenerate rot6d predictions.")
                proposal[:, 3:9] = rotations[:, :, :2].swapaxes(-1, -2).reshape(-1, 6)
                anchor = pose_to_mat(np.concatenate([
                    env_obs[f"robot{side}_eef_pos"][-1],
                    env_obs[f"robot{side}_eef_rot_axis_angle"][-1],
                ]))
                action, switched, index = arm.step(proposal, state_ref=TF_mat(anchor))
                # Wrapper returns relative column rot6d; UMI/IK consumes row rot6d.
                rotation = TF_mat.from_rot6d(action[:3], action[3:9]).as_matrix()[:3, :3]
                action[3:9] = mat_to_rot6d(rotation)
                selected.append(action)
                decisions.append({"arm": "left" if side == 0 else "right",
                                  "switched": bool(switched), "index": int(arm.chunk_step - 1),
                                  "proposed_index": int(index)})
        except Exception:
            self.restore(snapshot)
            raise
        self.last_decisions = decisions
        return np.concatenate(selected).reshape(1, 20)


def configure_chunkrl(engine, request) -> None:
    """LOAD-only mode setup; restore the original output window on every reload."""
    engine._chunkrl = None
    config = engine._policy.config
    if not hasattr(engine, "_base_n_action_steps"):
        engine._base_n_action_steps = getattr(config, "n_action_steps", None)
    if engine._base_n_action_steps is not None:
        config.n_action_steps = engine._base_n_action_steps
    mode = str(getattr(request, "action_request_mode", "") or "async").strip().lower()
    if mode != "chunkrl":
        return
    if config.type != "diffusion" or engine._relative_eef is None:
        raise ValueError("ChunkRL requires a Diffusion checkpoint with state='relative' and 20-dimensional EEF actions. Select the relative model; joint-action models support Async/Sync.")
    manager = BimanualChunkRL(getattr(request, "chunkrl_checkpoint", ""), device=engine._device)
    available = config.horizon - config.n_obs_steps + 1
    if manager.length > available:
        raise ValueError(f"ChunkRL L={manager.length} exceeds Diffusion's {available} available future actions.")
    config.n_action_steps = manager.length
    engine._policy.reset()
    engine._chunkrl = manager
