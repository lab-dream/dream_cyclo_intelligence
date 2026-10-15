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

    def __init__(self, model_path: str, robot: Any, robot_type: str,
                 ik_backend: str = "cyclo_control", ik_model_path: str = "",
                 device: str = "cpu") -> None:
        self.ik_backend = str(ik_backend or "cyclo_control").strip().lower()
        if self.ik_backend not in {"cyclo_control", "nodeik", "dls"}:
            raise ValueError(f"Unsupported IK backend: {self.ik_backend!r}.")
        if robot_type != "ffw_sg2_rev1":
            raise ValueError(
                "Dream relative EEF checkpoints require robot_type=ffw_sg2_rev1."
            )
        config_path = Path(model_path) / "config.json"
        saved = json.loads(config_path.read_text()) if config_path.is_file() else {}
        self.policy_hand = saved.get("hand", "both")
        if self.policy_hand not in {"both", "left", "right"}:
            raise ValueError(f"Unsupported relative EEF hand: {self.policy_hand!r}.")
        self.command_arms = (("left", "right") if self.policy_hand == "both"
                             else (self.policy_hand,))
        self.action_keys = ["arm_left", "arm_right"]
        if self.ik_backend != "dls":
            self.action_keys.append("lift")
            self.lift_weight = float(os.environ.get("CYCLO_IK_LIFT_WEIGHT", "10.0"))
            if not np.isfinite(self.lift_weight) or self.lift_weight <= 0:
                raise ValueError("CYCLO_IK_LIFT_WEIGHT must be positive and finite")
        dataset_root = getattr(robot, "dataset_root", None)
        try:
            metadata = self._metadata_dir(Path(model_path))
        except FileNotFoundError:
            if dataset_root is None:
                raise
            # A transferred checkpoint may omit its metadata. In that case the
            # simulation must use its original training dataset, not test-set stats.
            training = json.loads((Path(model_path) / "train_config.json").read_text())["dataset"]
            if training.get("repo_id", "").split("/")[-1] != Path(dataset_root).name:
                raise ValueError("Bundle original training dataset_meta before using another simulation dataset.")
            metadata = Path(dataset_root) / "meta"
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
        if self.ik_backend != "dls" and robot._action_groups.get("lift", {}).get("joint_names") != [self._lift_name]:
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
        # DLS reads the measured lift for FK but optimizes only the arm joints.
        active_names = [*self._arm_names["left"], *self._arm_names["right"]]
        if self.ik_backend != "dls":
            active_names.append(self._lift_name)
        self._active_names = active_names
        if self.ik_backend == "dls":
            candidates = [Path(os.environ.get("DLS_IK_SDK_PATH", "/opt/cyclo_sdk/dls_ik"))]
            candidates.extend(parent / "sdk/dls_ik" for parent in Path(__file__).resolve().parents)
            for sdk_path in candidates:
                if (sdk_path / "dls_ik/__init__.py").is_file():
                    if str(sdk_path) not in sys.path:
                        sys.path.insert(0, str(sdk_path))
                    break
            try:
                from dls_ik import DLSIK, DLSOptions
            except ImportError as error:
                self._urdf_dir.cleanup()
                raise RuntimeError(
                    f"DLS IK dependency is unavailable ({error.name}). Mount sdk/dls_ik "
                    "at /opt/cyclo_sdk/dls_ik (or set DLS_IK_SDK_PATH) and install "
                    "modern_robotics==1.1.1 in the LeRobot Python environment. "
                    "Rebuild the LeRobot image to include these dependencies permanently."
                ) from error

            self.kinematics = DLSIK(
                str(urdf_path), active_names, [cfg.left_ee_frame, cfg.right_ee_frame],
                options=DLSOptions(),
            )
            logger.info("Relative EEF metadata: %s; IK backend: DLS arms only + elbow null-space", metadata)
            return
        if self.ik_backend == "nodeik":
            from .nodeik import NodeIK

            self.kinematics = NodeIK(ik_model_path, str(urdf_path), active_names, device)
            missing = set(self.kinematics.image_keys) - {
                f"observation.images.rgb.{camera}" for camera in robot.camera_names
            }
            if missing:
                raise ValueError(f"Robot cannot supply NodeIK vision cameras: {sorted(missing)}.")
            logger.info("Relative EEF metadata: %s; IK backend: NodeIK", metadata)
            return
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
        hand = getattr(self, "policy_hand", "both")
        if hand != "both":
            start = 0 if hand == "left" else 10
            state = state[..., start:start + 10]
        batch[OBS_STATE] = torch.as_tensor(
            state, dtype=tensor.dtype, device=tensor.device
        ).unsqueeze(0)
        self.observation_relative = state.copy()
        return batch

    def bimanual_relative(self, values: np.ndarray) -> np.ndarray:
        """Keep shared UMI/IK decoding; the untrained hand holds its measured pose."""
        values = np.asarray(values)
        hand = getattr(self, "policy_hand", "both")
        width = 20 if hand == "both" else 10
        if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
            raise ValueError(f"Expected finite relative EEF actions of shape (time, {width}).")
        if self.env_obs is None:
            raise RuntimeError("Capture the relative EEF observation before decoding actions.")
        if hand == "both":
            return values
        full = np.zeros((len(values), 20), dtype=values.dtype)
        for side in range(2):
            full[:, side * 10 + 3:side * 10 + 9] = [1, 0, 0, 0, 1, 0]
            full[:, side * 10 + 9] = self.env_obs[f"robot{side}_gripper_width"][-1, 0]
        start = 0 if hand == "left" else 10
        full[:, start:start + 10] = values
        return full

    def pose_preview(self, action=None) -> dict:
        """The same TCP anchors and UMI poses for live and dataset viewers."""
        anchors = []
        for side in range(2):
            pose = np.concatenate([self.env_obs[f"robot{side}_eef_pos"][-1],
                                   self.env_obs[f"robot{side}_eef_rot_axis_angle"][-1]])
            anchors.append(pose_to_mat(pose).tolist())
        observation = self.observation_relative
        return dict(
            policy_hand=getattr(self, "policy_hand", "both"),
            arm_selection="both" if len(self.command_arms) == 2 else self.command_arms[0],
            anchor_poses=anchors,
            observation_poses=get_real_umi_action(self.bimanual_relative(observation), self.env_obs,
                                                 action_pose_repr="relative").tolist(),
            action_poses=get_real_umi_action(self.bimanual_relative(action), self.env_obs,
                                            action_pose_repr="relative").tolist() if action is not None else [],
        )

    def decode_action(self, chunk: np.ndarray) -> np.ndarray:
        """Restore TCP targets; DLS outputs only the arms and two grippers."""
        chunk = self.bimanual_relative(chunk)
        action = get_real_umi_action(chunk, self.env_obs, action_pose_repr="relative")
        lower, upper = self.transform.gripper_bounds[ACTION]
        grippers = denormalize(np.clip(action[:, [6, 13]], -1, 1), lower, upper)
        commands = []
        seed = {name: self.measured[name] for name in self._active_names}
        inactive = set(("left", "right")) - set(getattr(self, "command_arms", ("left", "right")))
        held_poses = {}
        if inactive:
            current_poses = self.transform.poses(
                np.array([self.measured.get(name, 0.0) for name in self.names]), OBS_STATE)[0]
            held_poses = {side: current_poses[index] for index, side in enumerate(("left", "right")) if side in inactive}
        for waypoint in range(len(action)):
            poses = {
                side: pose_to_mat(action[waypoint, index * 7 : index * 7 + 6])
                for index, side in enumerate(("left", "right"))
            }
            poses.update(held_poses)
            seed = self._solve_bimanual(poses, seed)
            command = []
            for index, side in enumerate(("left", "right")):
                command.extend(
                    [*[seed[name] for name in self._arm_names[side]], grippers[waypoint, index]]
                )
            if self.ik_backend != "dls":
                command.append(seed[self._lift_name])
            commands.append(command)
        result = np.asarray(commands, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("IK produced non-finite joint commands.")
        return result

    def _solve_bimanual(
        self, poses: dict[str, np.ndarray], seed: dict[str, float]
    ) -> dict[str, float]:
        """Solve both targets with the current measured non-active joints."""
        options = {}
        if self.ik_backend == "dls":
            inactive = set(("left", "right")) - set(getattr(self, "command_arms", ("left", "right")))
            if inactive:
                options["locked_joint_names"] = [name for side in inactive for name in self._arm_names[side]]
        result, position_error, rotation_error = self.kinematics.solve(
            {**self.measured, **seed}, poses["left"], poses["right"], **options
        )
        if self.ik_backend == "dls" and set(result) != set(self._active_names):
            raise ValueError("DLS IK must return exactly the 14 arm joints without lift.")
        if position_error >= 1e-4 or rotation_error >= 1e-3:
            logger.debug("IK residual: position=%.6fm, rotation=%.6frad",
                         position_error, rotation_error)
            if self.ik_backend == "dls":
                raise ValueError(
                    "DLS IK target did not converge: "
                    f"position={position_error:.6f}m, rotation={rotation_error:.6f}rad. "
                    "The action chunk was not accepted."
                )
        if not np.isfinite(list(result.values())).all():
            raise ValueError("IK produced non-finite joint commands.")
        if self.ik_backend != "dls":
            lower, upper = self._lift_limits
            if not lower - 1e-8 <= result[self._lift_name] <= upper + 1e-8:
                raise ValueError("Shared arm/lift IK exceeded the lift joint limits.")
            result[self._lift_name] = float(np.clip(result[self._lift_name], lower, upper))
        return result

    def close(self) -> None:
        self.kinematics = None
        self._urdf_dir.cleanup()
