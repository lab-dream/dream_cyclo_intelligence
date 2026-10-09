"""Bridge Dream's relative EEF checkpoints through UMI inference to Cyclo IK."""

from __future__ import annotations

import json
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.constants import ACTION, OBS_STATE

from ._vendor.dream.math import denormalize
from ._vendor.dream.relative_eef_pose import RelativeEEFTransform
from ._vendor.umi.pose_util import mat_to_pose, pose_to_mat
from ._vendor.umi.real_inference_util import get_real_umi_action, get_real_umi_obs_dict

logger = logging.getLogger("lerobot_engine")


class RelativeEEFAdapter:
    """Keep one measured TCP anchor for the entire observation/action window."""

    action_keys = ["arm_left", "arm_right"]

    def __init__(self, model_path: str, robot: Any, robot_type: str) -> None:
        if robot_type != "ffw_sg2_rev1":
            raise ValueError(
                "Dream relative EEF checkpoints require robot_type=ffw_sg2_rev1."
            )
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
        # Placo's scalar joint API assumes nq=1. Continuous wheel joints use
        # nq=2 in Pinocchio and shift its joint offsets; the base is fixed here.
        for joint in tree.getroot().findall("joint"):
            if joint.get("type") == "continuous":
                joint.set("type", "fixed")
        urdf_path = Path(self._urdf_dir.name) / "robot.urdf"
        tree.write(urdf_path)
        self.kinematics: dict[str, RobotKinematics] = {}
        cfg = self.transform.config
        for side, frame, names in (
            ("left", cfg.left_ee_frame, cfg.left_arm_joint_names),
            ("right", cfg.right_ee_frame, cfg.right_arm_joint_names),
        ):
            expected = [*names, f"gripper_{side[0]}_joint1"]
            actual = robot._action_groups[f"arm_{side}"]["joint_names"]
            if actual != expected:
                raise ValueError(
                    f"Unexpected arm_{side} command joint order: {actual}."
                )
            solver = RobotKinematics(str(urdf_path), frame, list(names))
            for name in solver.robot.joint_names():
                if name not in names:
                    solver.solver.mask_dof(name)
                    # A measured, locked head/lift/gripper must not make arm IK
                    # infeasible because its sensor value exceeds a URDF limit.
                    solver.robot.set_joint_limits(name, -1e6, 1e6)
            solver.solver.enable_joint_limits(True)
            self.kinematics[side] = solver
        logger.info(
            "Relative EEF metadata: %s; UMI pose decoding + Cyclo IK enabled", metadata
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
        """Use UMI's observation conversion with Dream's FK and gripper scaling."""
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
        """Restore UMI absolute poses, then solve the existing Cyclo arm IK."""
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
        seeds = {
            side: np.asarray([self.measured[name] for name in solver.joint_names])
            for side, solver in self.kinematics.items()
        }
        for waypoint in range(len(action)):
            arms = []
            for index, side in enumerate(("left", "right")):
                solver = self.kinematics[side]
                for name in solver.robot.joint_names():
                    if name in self.measured:
                        solver.robot.set_joint(name, float(self.measured[name]))
                pose = pose_to_mat(action[waypoint, index * 7 : index * 7 + 6])
                seed = np.rad2deg(
                    seeds[side]
                )  # Cyclo IK's API uses degrees; ROS uses radians.
                solver.forward_kinematics(seed)
                for _ in range(20):
                    seed = solver.inverse_kinematics(seed, pose, orientation_weight=1.0)
                    reached = solver.robot.get_T_world_frame(solver.target_frame_name)
                    if (
                        np.linalg.norm(reached[:3, 3] - pose[:3, 3]) < 1e-4
                        and np.linalg.norm(reached[:3, :3] - pose[:3, :3]) < 1e-3
                    ):
                        break
                seeds[side] = np.deg2rad(seed)
                arms.extend([*seeds[side], grippers[waypoint, index]])
            commands.append(arms)
        result = np.asarray(commands, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("IK produced non-finite joint commands.")
        return result

    def close(self) -> None:
        self.kinematics.clear()
        self._urdf_dir.cleanup()
