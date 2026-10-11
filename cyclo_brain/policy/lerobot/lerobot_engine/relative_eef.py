"""Bridge Dream's relative EEF checkpoints through UMI inference to Cyclo IK."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.utils.constants import ACTION, OBS_STATE

from ._vendor.dream.math import denormalize
from ._vendor.dream.relative_eef_pose import RelativeEEFTransform
from ._vendor.umi.pose_util import mat_to_pose, pose_to_mat
from ._vendor.umi.real_inference_util import get_real_umi_action, get_real_umi_obs_dict

logger = logging.getLogger("lerobot_engine")


class RelativeEEFAdapter:
    """Keep one measured TCP anchor for the entire observation/action window."""

    action_keys = ["arm_left", "arm_right", "lift"]

    def __init__(self, model_path: str, robot: Any, robot_type: str) -> None:
        if robot_type != "ffw_sg2_rev1":
            raise ValueError(
                "Dream relative EEF checkpoints require robot_type=ffw_sg2_rev1."
            )
        self.lift_weight = float(os.environ.get("CYCLO_IK_LIFT_WEIGHT", "10.0"))
        if not np.isfinite(self.lift_weight) or self.lift_weight <= 0:
            raise ValueError("CYCLO_IK_LIFT_WEIGHT must be positive and finite")
        metadata = self._metadata_dir(Path(model_path))
        features = json.loads((metadata / "info.json").read_text())["features"]
        stats = json.loads((metadata / "stats.json").read_text())
        self.transform = RelativeEEFTransform(features, stats)
        self.names = features[OBS_STATE]["names"]
        self.robot = robot
        self.env_obs: dict[str, np.ndarray] | None = None
        self.measured: dict[str, float] = {}
        self.shape_meta = {
            "obs": {
                f"robot{side}_{key}": {"shape": [width], "type": "low_dim"}
                for side in range(2)
                for key, width in (
                    ("eef_pos", 3),
                    ("eef_rot_axis_angle", 6),
                    ("gripper_width", 1),
                )
            }
        }

        # IK needs the URDF's joints/frames, whereas rendering meshes belong to
        # the GUI. Strip mesh geometry so the copied training URDF works without ROS.
        self._urdf_dir = tempfile.TemporaryDirectory(prefix="cyclo_eef_ik_")
        tree = ET.parse(self.transform.config.urdf_path)
        for link in tree.getroot().findall("link"):
            for geometry in list(link.findall("visual")) + list(
                link.findall("collision")
            ):
                link.remove(geometry)
        # Cyclo's scalar q/qdot API needs nq=nv; continuous wheels use nq=2.
        # The mobile base is fixed during EEF inference.
        for joint in tree.getroot().findall("joint"):
            if joint.get("type") == "continuous":
                joint.set("type", "fixed")
        urdf_path = Path(self._urdf_dir.name) / "robot.urdf"
        tree.write(urdf_path)
        cfg = self.transform.config
        self._arm_names = {
            "left": list(cfg.left_arm_joint_names),
            "right": list(cfg.right_arm_joint_names),
        }
        self._lift_name = cfg.lift_joint_name
        lift_joint = tree.getroot().find(f"joint[@name='{self._lift_name}']")
        if lift_joint is None or lift_joint.get("type") != "prismatic":
            raise ValueError("Relative EEF lift IK requires a prismatic lift joint.")
        lift_limit = lift_joint.find("limit")
        if lift_limit is None or not {"lower", "upper"}.issubset(lift_limit.attrib):
            raise ValueError("Relative EEF lift IK requires URDF lift position limits.")
        self._lift_limits = (
            float(lift_limit.get("lower")), float(lift_limit.get("upper"))
        )
        if not np.isfinite(self._lift_limits).all() or self._lift_limits[0] >= self._lift_limits[1]:
            raise ValueError("Invalid URDF lift position limits.")
        if robot._action_groups.get("lift", {}).get("joint_names") != [self._lift_name]:
            raise ValueError("Relative EEF lift IK requires a lift action group.")
        for side, names in (
            ("left", cfg.left_arm_joint_names),
            ("right", cfg.right_arm_joint_names),
        ):
            expected = [*names, f"gripper_{side[0]}_joint1"]
            actual = robot._action_groups[f"arm_{side}"]["joint_names"]
            if actual != expected:
                raise ValueError(
                    f"Unexpected arm_{side} command joint order: {actual}."
                )
        # Both TCP targets share one lift in the official Cyclo MoveL QP core.
        # Revolute joints use radians; the prismatic lift uses meters.
        active_names = [
            *self._arm_names["left"], *self._arm_names["right"], self._lift_name
        ]
        self._active_names = active_names
        native_path = os.environ.get("CYCLO_IK_MODULE_PATH", "/opt/cyclo_ik")
        if native_path not in sys.path:
            sys.path.insert(0, native_path)
        try:
            from _cyclo_ik import CycloIK
        except ImportError as error:
            raise RuntimeError(
                "Cyclo native IK binding is unavailable. Initialize the cyclo_control "
                "submodule and rebuild the LeRobot image (or run sdk/cyclo_ik/build.sh)."
            ) from error
        self.kinematics = CycloIK(
            str(urdf_path), active_names, [cfg.left_ee_frame, cfg.right_ee_frame],
            self.lift_weight,
        )
        logger.info(
            "Relative EEF metadata: %s; Cyclo IK damping: arm=1e-4, lift=%g",
            metadata, self.lift_weight,
        )

    @staticmethod
    def _metadata_dir(model_dir: Path) -> Path:
        # A portable checkpoint needs only these two original joint-schema
        # metadata files; relative processor statistics cannot recover gripper ranges.
        candidates = [model_dir / "dataset_meta"]
        train_path = model_dir / "train_config.json"
        if train_path.is_file():
            dataset = json.loads(train_path.read_text()).get("dataset", {})
            if dataset.get("root"):
                candidates.append(Path(dataset["root"]) / "meta")
            from lerobot.utils.constants import HF_LEROBOT_HOME

            repo_id = dataset.get("repo_id")
            if repo_id:
                candidates.append(HF_LEROBOT_HOME / repo_id / "meta")
        for candidate in candidates:
            if all(
                (candidate / name).is_file() for name in ("info.json", "stats.json")
            ):
                return candidate
        raise FileNotFoundError(
            "Relative EEF requires the original dataset meta/info.json and meta/stats.json. "
            "Copy both into <policy_path>/dataset_meta/ when transferring the checkpoint."
        )

    def raw_state(self, joints: dict[str, np.ndarray]) -> np.ndarray:
        """Resolve measured joints by name, independent of runtime modality order."""
        measured = {}
        for group, values in joints.items():
            names = self.robot.get_joint_names(group)
            measured.update(zip(names, np.asarray(values).tolist(), strict=True))
        mapping = self.transform.indices[OBS_STATE]
        required = (
            mapping["left"] + mapping["right"] + [mapping["lift"]] + mapping["grippers"]
        )
        missing = [
            self.names[index] for index in required if self.names[index] not in measured
        ]
        if missing:
            raise ValueError(f"Missing measured joints for relative EEF: {missing}.")
        state = np.asarray(
            [measured.get(name, 0.0) for name in self.names], dtype=np.float32
        )
        if not np.isfinite(state).all():
            raise ValueError("Non-finite measured joints for relative EEF.")
        self.measured = measured
        return state

    def encode_observation(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Encode past poses as inv(T_current) @ T_past, using the latest frame.

        UMI keeps this same current TCP anchor in ``env_obs`` for decoding the
        predicted action window. Grippers retain Dream's scalar range scaling.
        """
        tensor = batch[OBS_STATE]
        joints = tensor[0].detach().cpu().numpy()
        poses = self.transform.poses(joints, OBS_STATE)
        grippers = self.transform.grippers(joints, OBS_STATE)
        env_obs = {}
        for side in range(2):
            pose = mat_to_pose(poses[:, side])
            env_obs[f"robot{side}_eef_pos"] = pose[:, :3]
            env_obs[f"robot{side}_eef_rot_axis_angle"] = pose[:, 3:]
            env_obs[f"robot{side}_gripper_width"] = grippers[:, side : side + 1]
        self.env_obs = env_obs
        obs = get_real_umi_obs_dict(
            env_obs,
            self.shape_meta,
            obs_pose_repr="relative",
            tx_robot1_robot0=np.eye(4),
        )
        state = np.concatenate(
            [
                obs[f"robot{side}_{key}"]
                for side in range(2)
                for key in ("eef_pos", "eef_rot_axis_angle", "gripper_width")
            ],
            axis=-1,
        )
        batch[OBS_STATE] = torch.as_tensor(
            state, dtype=tensor.dtype, device=tensor.device
        ).unsqueeze(0)
        return batch

    def decode_action(self, chunk: np.ndarray) -> np.ndarray:
        """Restore absolute TCP poses and return arms/grippers + shared lift."""
        if chunk.ndim != 2 or chunk.shape[1] != 20 or not np.isfinite(chunk).all():
            raise ValueError(
                "Expected finite relative EEF actions of shape (time, 20)."
            )
        if self.env_obs is None:
            raise RuntimeError(
                "Capture the relative EEF observation before decoding actions."
            )
        action = get_real_umi_action(chunk, self.env_obs, action_pose_repr="relative")
        lower, upper = self.transform.gripper_bounds[ACTION]
        grippers = denormalize(np.clip(action[:, [6, 13]], -1, 1), lower, upper)
        commands = []
        seed = {name: self.measured[name] for name in self._active_names}
        for waypoint in range(len(action)):
            poses = {
                side: pose_to_mat(action[waypoint, index * 7 : index * 7 + 6])
                for index, side in enumerate(("left", "right"))
            }
            seed = self._solve_bimanual(poses, seed)
            command = []
            for index, side in enumerate(("left", "right")):
                command.extend(
                    [*[seed[name] for name in self._arm_names[side]], grippers[waypoint, index]]
                )
            command.append(seed[self._lift_name])
            commands.append(command)
        result = np.asarray(commands, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("IK produced non-finite joint commands.")
        return result

    def _solve_bimanual(
        self, poses: dict[str, np.ndarray], seed: dict[str, float]
    ) -> dict[str, float]:
        """Solve both targets together, preserving SI units and URDF limits."""
        lower, upper = self._lift_limits
        result, position_error, rotation_error = self.kinematics.solve(
            {**self.measured, **seed}, poses["left"], poses["right"]
        )
        if position_error >= 1e-4 or rotation_error >= 1e-3:
            logger.debug("Cyclo IK residual: position=%.6fm, rotation=%.6frad",
                         position_error, rotation_error)
        if not np.isfinite(list(result.values())).all():
            raise ValueError("Shared arm/lift IK produced non-finite joint commands.")
        if not lower - 1e-8 <= result[self._lift_name] <= upper + 1e-8:
            raise ValueError("Shared arm/lift IK exceeded the lift joint limits.")
        result[self._lift_name] = float(np.clip(result[self._lift_name], lower, upper))
        return result

    def close(self) -> None:
        self.kinematics = None
        self._urdf_dir.cleanup()
