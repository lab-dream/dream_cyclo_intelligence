"""Pinocchio forward-kinematics wrapper for ROBOTIS AI Worker.

Requires: numpy, pinocchio (Python package commonly distributed as ``pin``).
Input must be a fully expanded URDF (not a .xacro file).

The model is fixed-base. Angles use radians; the lift joint uses meters.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin


@dataclass
class AIWorkerConfig:
    """URDF, joint names and TCP definitions for AI Worker FK."""

    urdf_path: str = str(
        Path(__file__).resolve().parent
        / "ai_worker/ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
    )
    right_ee_frame: str = "end_effector_r_link"
    left_ee_frame: str = "end_effector_l_link"
    base_frame: str = "base_link"
    right_arm_joint_names: tuple[str, ...] = tuple(f"arm_r_joint{i}" for i in range(1, 8))
    left_arm_joint_names: tuple[str, ...] = tuple(f"arm_l_joint{i}" for i in range(1, 8))
    lift_joint_name: str = "lift_joint"
    # TCP offsets are expressed in the corresponding EE frame, in meters.
    right_tcp_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    left_tcp_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))


class AIWorkerRobot:
    """Compute fixed-base FK for the lift and both AI Worker arms."""

    def __init__(self, cfg: AIWorkerConfig):
        """Load the fixed-base model and map scalar joints by name."""
        self.cfg = cfg
        urdf = Path(cfg.urdf_path).expanduser()
        if not urdf.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf}")

        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)

        self.right_arm_joint_names = cfg.right_arm_joint_names
        self.left_arm_joint_names = cfg.left_arm_joint_names
        self.arm_joint_names = {
            "right": self.right_arm_joint_names,
            "left": self.left_arm_joint_names,
        }
        self.ee_frame_ids = {
            "right": self._frame_id(cfg.right_ee_frame),
            "left": self._frame_id(cfg.left_ee_frame),
        }
        self.base_frame_id = self._frame_id(cfg.base_frame)
        self.tcp_offsets = {
            "right": np.asarray(cfg.right_tcp_offset, dtype=float).reshape(3).copy(),
            "left": np.asarray(cfg.left_tcp_offset, dtype=float).reshape(3).copy(),
        }

        # Pinocchio configuration-vector indices are not necessarily URDF order.
        self.arm_qpos_indices = {
            side: np.array([self._joint_q_index(name) for name in names], dtype=int)
            for side, names in self.arm_joint_names.items()
        }
        self.lift_qpos_index = self._joint_q_index(cfg.lift_joint_name)
        self.fk()

    def _frame_id(self, name: str) -> int:
        if not self.model.existFrame(name):
            raise ValueError(f"Frame '{name}' not found in URDF")
        return int(self.model.getFrameId(name))

    def _joint_q_index(self, name: str) -> int:
        if not self.model.existJointName(name):
            raise ValueError(f"Joint '{name}' not found in URDF")
        joint_id = self.model.getJointId(name)
        if self.model.nqs[joint_id] != 1:
            raise ValueError(
                f"Joint '{name}' has nq={self.model.nqs[joint_id]}; this wrapper expects a 1-DoF scalar joint"
            )
        return int(self.model.idx_qs[joint_id])

    def get_joint_names(self) -> list[str]:
        """Return the model's joint names."""
        return list(self.model.names)[1:]

    def get_frame_names(self) -> list[str]:
        """Return the model's frame names."""
        return [frame.name for frame in self.model.frames]

    def get_q(self) -> np.ndarray:
        """Return the complete Pinocchio configuration vector (copy)."""
        return self.q.copy()

    def set_q(self, q: np.ndarray) -> None:
        """Set a full configuration vector and update all link transforms."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.model.nq:
            raise ValueError(f"Expected q shape ({self.model.nq},), got {q.shape}")
        self.q = q.copy()
        self.fk()

    def set_joint_positions(self, positions: dict[str, float]) -> None:
        """Update named scalar joints (radians; lift in meters)."""
        q = self.q.copy()
        for name, position in positions.items():
            q[self._joint_q_index(name)] = float(position)
        self.set_q(q)

    def fk(self, q: np.ndarray | None = None) -> None:
        """Compute FK for all frames, with optional full configuration q."""
        if q is not None:
            q = np.asarray(q, dtype=float).reshape(-1)
            if q.size != self.model.nq:
                raise ValueError(f"Expected q shape ({self.model.nq},), got {q.shape}")
            self.q = q.copy()
        pin.framesForwardKinematics(self.model, self.data, self.q)

    def get_arm_q(self, side: str) -> np.ndarray:
        """Return the seven arm positions for 'left' or 'right'."""
        return self.q[self.arm_qpos_indices[side]].copy()

    def forward_kinematics_arm(
        self, q_arm: np.ndarray, side: str = "right", lift: float | None = None
    ) -> np.ndarray:
        """Update one arm (and optionally lift), return its TCP world pose."""
        q_arm = np.asarray(q_arm, dtype=float).reshape(-1)
        indices = self.arm_qpos_indices[side]
        if q_arm.size != indices.size:
            raise ValueError(f"Expected {indices.size} {side} arm values, got {q_arm.size}")
        q = self.q.copy()
        q[indices] = q_arm
        if lift is not None:
            q[self.lift_qpos_index] = float(lift)
        self.set_q(q)
        return self.tcp_pose_world(side)

    def forward_kinematics_bimanual(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        lift: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Update both arms and lift in one FK pass; return both TCP poses."""
        q_left = np.asarray(q_left, dtype=float).reshape(-1)
        q_right = np.asarray(q_right, dtype=float).reshape(-1)
        il, ir = self.arm_qpos_indices["left"], self.arm_qpos_indices["right"]
        if q_left.size != il.size or q_right.size != ir.size:
            raise ValueError(f"Expected ({il.size},) left and ({ir.size},) right arm q")
        q = self.q.copy()
        q[il], q[ir] = q_left, q_right
        if lift is not None:
            q[self.lift_qpos_index] = float(lift)
        self.set_q(q)
        return {
            "left": self.tcp_pose_world("left"),
            "right": self.tcp_pose_world("right"),
        }

    def tcp_pose_world(self, side: str = "right") -> np.ndarray:
        """4x4 transform of TCP in the URDF model's world/reference frame."""
        transform = self.data.oMf[self.ee_frame_ids[side]].homogeneous.copy()
        transform[:3, 3] += transform[:3, :3] @ self.tcp_offsets[side]
        return transform

    def tcp_pose_base(self, side: str = "right") -> np.ndarray:
        """4x4 TCP pose relative to cfg.base_frame."""
        world_base = self.data.oMf[self.base_frame_id]
        tcp = self.tcp_pose_world(side)
        world_tcp = pin.SE3(tcp[:3, :3], tcp[:3, 3])
        return (world_base.inverse() * world_tcp).homogeneous.copy()
