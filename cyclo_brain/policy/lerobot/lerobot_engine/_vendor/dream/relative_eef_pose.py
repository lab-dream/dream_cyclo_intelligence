"""Sample-wise relative EEF trajectories for the fixed-base AI Worker model."""

from __future__ import annotations

from collections import OrderedDict
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.utils.constants import ACTION, OBS_STATE

from .ai_worker import AIWorkerConfig, AIWorkerRobot
from .math import convert_pose_mat_rep, mat_to_pose10d, normalize

if TYPE_CHECKING:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

EEF_NAMES = [
    f"{side}.{name}"
    for side in ("left", "right")
    for name in ("x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper")
]


class RelativeEEFTransform:
    """Use the latest measured TCP as each arm's fixed trajectory anchor.

    Position stays in meters and rotation uses UMI's row-based rot6d. Absolute
    gripper joints are range-normalized separately for observation and action.
    The original metadata and source samples are never mutated.
    """

    def __init__(
        self, features: dict[str, dict], stats: dict | None, *, observation_anchor_index: int = -1
    ) -> None:
        """Resolve source joint names and absolute gripper ranges."""
        self.config = AIWorkerConfig()
        self.observation_anchor_index = observation_anchor_index
        self._robot: AIWorkerRobot | None = None
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self.indices: dict[str, dict[str, Any]] = {}
        self.gripper_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key in (OBS_STATE, ACTION):
            names = features.get(key, {}).get("names")
            if not isinstance(names, list) or len(names) != features[key]["shape"][0]:
                raise ValueError(f"Relative EEF requires a named joint vector for {key!r}.")
            required = [
                *self.config.left_arm_joint_names,
                *self.config.right_arm_joint_names,
                self.config.lift_joint_name,
                "gripper_l_joint1",
                "gripper_r_joint1",
            ]
            missing = [name for name in required if name not in names]
            if missing or len(names) != len(set(names)):
                raise ValueError(f"Invalid AI Worker joint names in {key!r}; missing: {missing}.")
            self.indices[key] = {
                "left": [names.index(name) for name in self.config.left_arm_joint_names],
                "right": [names.index(name) for name in self.config.right_arm_joint_names],
                "lift": names.index(self.config.lift_joint_name),
                "grippers": [names.index("gripper_l_joint1"), names.index("gripper_r_joint1")],
                "size": len(names),
            }
            if stats is None or key not in stats or not {"min", "max"}.issubset(stats[key]):
                raise ValueError(f"Relative EEF requires min/max gripper statistics for {key!r}.")
            grippers = self.indices[key]["grippers"]
            lower = np.asarray(stats[key]["min"], dtype=np.float64)[grippers]
            upper = np.asarray(stats[key]["max"], dtype=np.float64)[grippers]
            if not np.isfinite([lower, upper]).all() or np.any(upper < lower):
                raise ValueError(f"Invalid gripper bounds for {key!r}.")
            self.gripper_bounds[key] = lower, upper

    def __getstate__(self) -> dict:
        """Serialize configuration without sharing a mutable FK model between workers."""
        # Pinocchio Data is process-local; spawned DataLoader workers build their own model.
        state = dict(self.__dict__)
        state["_robot"] = None
        state["_cache"] = OrderedDict()
        return state

    def poses(self, joints: np.ndarray, key: str) -> np.ndarray:
        """Return (time, arm, 4, 4) measured or commanded TCP transforms."""
        joints = np.atleast_2d(np.asarray(joints, dtype=np.float64))
        mapping = self.indices[key]
        if joints.ndim != 2 or joints.shape[-1] != mapping["size"] or not np.isfinite(joints).all():
            raise ValueError(f"Expected finite {key} with shape (time, {mapping['size']}).")
        if self._robot is None:
            self._robot = AIWorkerRobot(self.config)
        result = []
        for row in joints:
            # Canonical named-joint order also allows observation/action cache sharing.
            values = row[mapping["left"] + mapping["right"] + [mapping["lift"]]]
            cache_key = values.tobytes()
            if cache_key not in self._cache:
                pair = self._robot.forward_kinematics_bimanual(
                    values[:7], values[7:14], lift=float(values[14])
                )
                self._cache[cache_key] = np.stack([pair["left"], pair["right"]])
                if len(self._cache) > 8192:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(cache_key)
            result.append(self._cache[cache_key])
        return np.stack(result)

    def grippers(self, joints: np.ndarray, key: str) -> np.ndarray:
        """Normalize absolute gripper joints with their observation/action bounds."""
        values = np.asarray(joints)[..., self.indices[key]["grippers"]]
        return normalize(values, *self.gripper_bounds[key])

    @staticmethod
    def encode(poses: np.ndarray, grippers: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Encode both arms against one measured anchor per sample."""
        relative = convert_pose_mat_rep(poses, np.expand_dims(anchor, axis=-4), pose_rep="relative")
        arms = [
            np.concatenate(
                [mat_to_pose10d(relative[..., side, :, :]), grippers[..., side : side + 1]], axis=-1
            )
            for side in range(2)
        ]
        return np.concatenate(arms, axis=-1).astype(np.float32)

    def __call__(self, item: dict) -> dict:
        """Replace the two joint trajectories, preserving padding and other sample fields."""
        observation = np.asarray(item[OBS_STATE])
        action = np.asarray(item[ACTION])
        observation_sequence = np.atleast_2d(observation)
        action_sequence = np.atleast_2d(action)
        obs_poses = self.poses(observation_sequence, OBS_STATE)
        action_poses = self.poses(action_sequence, ACTION)
        anchor = obs_poses[self.observation_anchor_index]
        state = self.encode(obs_poses, self.grippers(observation_sequence, OBS_STATE), anchor)
        target = self.encode(action_poses, self.grippers(action_sequence, ACTION), anchor)
        result = dict(item)
        result[OBS_STATE] = torch.from_numpy(state[0] if observation.ndim == 1 else state)
        result[ACTION] = torch.from_numpy(target[0] if action.ndim == 1 else target)
        return result


class RelativeStatsAccumulator:
    """Accumulate transformed window statistics without retaining all windows."""

    def __init__(self) -> None:
        """Initialize running moments and extrema."""
        self.count = 0
        self.minimum = np.full(20, np.inf)
        self.maximum = np.full(20, -np.inf)
        self.total = np.zeros(20)
        self.squared_total = np.zeros(20)

    def update(self, values: np.ndarray) -> None:
        """Include one batch of relative trajectories."""
        flat: np.ndarray = values.reshape(-1, 20).astype(np.float64)
        self.count += len(flat)
        self.minimum = np.minimum(self.minimum, flat.min(axis=0))
        self.maximum = np.maximum(self.maximum, flat.max(axis=0))
        self.total += flat.sum(axis=0)
        self.squared_total += (flat * flat).sum(axis=0)

    def finish(self) -> dict[str, np.ndarray]:
        """Keep rotation and already-normalized gripper dimensions unchanged."""
        if self.count == 0:
            raise ValueError("Cannot compute relative statistics from an empty dataset.")
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.squared_total / self.count - mean**2, 0))
        minimum, maximum = self.minimum.copy(), self.maximum.copy()
        identity_dims = [*range(3, 10), *range(13, 20)]
        minimum[identity_dims], maximum[identity_dims] = -1, 1
        mean[identity_dims], std[identity_dims] = 0, 1
        return {"min": minimum, "max": maximum, "mean": mean, "std": std, "count": np.array([self.count])}


def relative_metadata(meta: LeRobotDatasetMetadata, stats: dict[str, dict]) -> LeRobotDatasetMetadata:
    """Build an in-memory training contract while preserving the on-disk joint schema."""
    result = copy(meta)
    result.info = deepcopy(meta.info)
    result.stats = deepcopy(meta.stats or {})
    for key in (OBS_STATE, ACTION):
        result.info.features[key] = {**result.info.features[key], "shape": (20,), "names": list(EEF_NAMES)}
        result.stats[key] = deepcopy(stats[key])
    return result
