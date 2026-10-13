# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""Lightweight SE(3) transformation and action-space coordinate conversion utilities.

Only depends on numpy, scipy, and torch. Does NOT require robomimic, robosuite,
lerobot, or ROS.
"""

from typing import Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def _is_torch_tensor(value):
    return isinstance(value, torch.Tensor)


def _torch_quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert xyzw quaternions to rotation matrices without leaving torch."""
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    x, y, z, w = quat.unbind(dim=-1)
    two = quat.new_tensor(2)
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def _torch_rotmat_to_quat(rotm: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to xyzw quaternions using torch operations."""
    r00, r11, r22 = rotm[..., 0, 0], rotm[..., 1, 1], rotm[..., 2, 2]
    zero = rotm.new_zeros(())
    qx = torch.sqrt(torch.clamp(1 + r00 - r11 - r22, min=zero)) / 2
    qy = torch.sqrt(torch.clamp(1 - r00 + r11 - r22, min=zero)) / 2
    qz = torch.sqrt(torch.clamp(1 - r00 - r11 + r22, min=zero)) / 2
    qw = torch.sqrt(torch.clamp(1 + r00 + r11 + r22, min=zero)) / 2
    qx = torch.copysign(qx, rotm[..., 2, 1] - rotm[..., 1, 2])
    qy = torch.copysign(qy, rotm[..., 0, 2] - rotm[..., 2, 0])
    qz = torch.copysign(qz, rotm[..., 1, 0] - rotm[..., 0, 1])
    return torch.stack((qx, qy, qz, qw), dim=-1)


class TF_mat:
    """4x4 SE(3) Homogeneous Transformation matrix wrapper supporting numpy and torch."""

    def __init__(self, T=None):
        if T is None:
            self.T = np.identity(4, dtype=np.float32)
        else:
            self.T = T

    @classmethod
    def from_vectors(cls, pos, quat=(0.0, 0.0, 0.0, 1.0)):
        """Build TF_mat from position [x, y, z] and xyzw quaternion."""
        if _is_torch_tensor(pos) or _is_torch_tensor(quat):
            if not _is_torch_tensor(pos):
                pos = torch.as_tensor(pos, dtype=quat.dtype, device=quat.device)
            quat = torch.as_tensor(quat, dtype=pos.dtype, device=pos.device)
            tf_mat = torch.zeros(*pos.shape[:-1], 4, 4, dtype=pos.dtype, device=pos.device)
            tf_mat[..., :3, 3] = pos
            tf_mat[..., :3, :3] = _torch_quat_to_rotmat(quat)
            tf_mat[..., 3, 3] = 1
            return cls(tf_mat)

        pos = np.asarray(pos, dtype=np.float32)
        quat = np.asarray(quat, dtype=np.float32)
        tf_mat = cls(np.zeros([*pos.shape[:-1], 4, 4], dtype=np.float32))
        tf_mat.T[..., :3, 3] = pos
        tf_mat.T[..., :3, :3] = Rotation.from_quat(quat).as_matrix()
        tf_mat.T[..., 3, 3] = 1
        return tf_mat

    @classmethod
    def from_mat(cls, pos, rotm):
        """Build TF_mat from position and 3x3 rotation matrix."""
        if _is_torch_tensor(pos) or _is_torch_tensor(rotm):
            if not _is_torch_tensor(pos):
                pos = torch.as_tensor(pos, dtype=rotm.dtype, device=rotm.device)
            rotm = torch.as_tensor(rotm, dtype=pos.dtype, device=pos.device)
            tf_mat = torch.zeros(*pos.shape[:-1], 4, 4, dtype=pos.dtype, device=pos.device)
            tf_mat[..., :3, 3] = pos
            tf_mat[..., :3, :3] = rotm
            tf_mat[..., 3, 3] = 1
            return cls(tf_mat)

        pos = np.asarray(pos, dtype=np.float32)
        rotm = np.asarray(rotm, dtype=np.float32)
        tf_mat = cls(np.zeros([*pos.shape[:-1], 4, 4], dtype=np.float32))
        tf_mat.T[..., :3, 3] = pos
        tf_mat.T[..., :3, :3] = rotm
        tf_mat.T[..., 3, 3] = 1
        return tf_mat

    @classmethod
    def from_rot6d(cls, pos, rot6d):
        """Build TF_mat from position and 6D rotation representation."""
        if _is_torch_tensor(pos) or _is_torch_tensor(rot6d):
            if not _is_torch_tensor(pos):
                pos = torch.as_tensor(pos, dtype=rot6d.dtype, device=rot6d.device)
            rot6d = torch.as_tensor(rot6d, dtype=pos.dtype, device=pos.device)
            rot1 = torch.nn.functional.normalize(rot6d[..., :3], dim=-1)
            rot2 = rot6d[..., 3:] - (rot6d[..., 3:] * rot1).sum(dim=-1, keepdim=True) * rot1
            rot2 = torch.nn.functional.normalize(rot2, dim=-1)
            rotm = torch.stack((rot1, rot2, torch.cross(rot1, rot2, dim=-1)), dim=-1)
            return cls.from_mat(pos, rotm)

        pos = np.asarray(pos, dtype=np.float32)
        rot6d = np.asarray(rot6d, dtype=np.float32)
        num_TFs = rot6d.shape[:-1]
        rot1 = rot6d[..., :3] / np.linalg.norm(rot6d[..., :3], axis=-1, keepdims=True)
        rot2 = rot6d[..., 3:] - (rot6d[..., 3:] * rot1).sum(axis=-1).reshape(*num_TFs, -1) * rot1
        rot2 = rot2 / np.linalg.norm(rot2, axis=-1, keepdims=True)
        rot3 = np.cross(rot1, rot2)
        rotm = np.stack([rot1, rot2, rot3], axis=-1)

        tf_mat = cls(np.zeros([*num_TFs, 4, 4], dtype=np.float32))
        tf_mat.T[..., :3, 3] = pos
        tf_mat.T[..., :3, :3] = rotm
        tf_mat.T[..., 3, 3] = 1
        return tf_mat

    @classmethod
    def mul(cls, tf1: "TF_mat", tf2: "TF_mat") -> "TF_mat":
        """Matrix multiplication of two transformations."""
        if _is_torch_tensor(tf1.T) or _is_torch_tensor(tf2.T):
            return cls(torch.matmul(tf1.T, tf2.T))
        return cls(np.matmul(tf1.T, tf2.T))

    def inverse(self) -> "TF_mat":
        """Compute the SE(3) matrix inverse."""
        p = self.T[..., :3, 3:]
        R = self.T[..., :3, :3]
        if _is_torch_tensor(self.T):
            inv = torch.zeros_like(self.T)
            inv[..., :3, :3] = R.swapaxes(-1, -2)
            inv[..., :3, 3:] = -torch.matmul(R.swapaxes(-1, -2), p)
            inv[..., 3, 3] = 1
        else:
            inv = np.zeros_like(self.T)
            inv[..., :3, :3] = R.swapaxes(-1, -2)
            inv[..., :3, 3:] = -np.matmul(R.swapaxes(-1, -2), p)
            inv[..., 3, 3] = 1
        return TF_mat(inv)

    def as_matrix(self):
        return self.T

    def get_pos(self):
        p = self.T[..., :3, 3]
        return p.clone() if _is_torch_tensor(p) else p.copy()

    def get_rot6d(self):
        R = self.T[..., :3, :2]
        if _is_torch_tensor(R):
            return R.transpose(-1, -2).reshape(*self.T.shape[:-2], -1)
        return R.swapaxes(-1, -2).reshape(*self.T.shape[:-2], -1)


def absolute_actions_to_relative(
    absolute_actions: np.ndarray, state_tf: TF_mat
) -> np.ndarray:
    """Express absolute [xyz, rot6d, gripper] actions relative to state_tf."""
    absolute_actions = np.asarray(absolute_actions, dtype=np.float32)
    if absolute_actions.ndim != 2 or absolute_actions.shape[-1] != 10:
        raise ValueError(
            "Relative action inputs require an action sequence shaped "
            f"(steps, 10), got {absolute_actions.shape}."
        )

    absolute_tf = TF_mat.from_rot6d(absolute_actions[:, :3], absolute_actions[:, 3:9])
    relative_tf = TF_mat.mul(state_tf.inverse(), absolute_tf)
    return np.concatenate(
        [relative_tf.get_pos(), relative_tf.get_rot6d(), absolute_actions[:, 9:10]],
        axis=-1,
    ).astype(np.float32)


def relative_actions_to_absolute(
    relative_actions: np.ndarray, state_tf: TF_mat
) -> np.ndarray:
    """Anchor relative [xyz, rot6d, gripper] actions at state_tf to make them absolute."""
    relative_actions = np.asarray(relative_actions, dtype=np.float32)
    if relative_actions.ndim != 2 or relative_actions.shape[-1] != 10:
        raise ValueError(
            "Relative action inference requires an action chunk shaped "
            f"(steps, 10), got {relative_actions.shape}."
        )

    relative_tf = TF_mat.from_rot6d(relative_actions[:, :3], relative_actions[:, 3:9])
    absolute_tf = TF_mat.mul(state_tf, relative_tf)
    return np.concatenate(
        [absolute_tf.get_pos(), absolute_tf.get_rot6d(), relative_actions[:, 9:10]],
        axis=-1,
    ).astype(np.float32)

