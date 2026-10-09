"""Pose conversion follows real-stanford/universal_manipulation_interface.

https://github.com/real-stanford/universal_manipulation_interface
"""

import numpy as np


def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep="abs", backward=False):
    """Convert poses to or from an absolute, relative, or delta representation."""
    if not backward:
        # training transform
        if pose_rep == "abs":
            return pose_mat
        elif pose_rep == "rel":
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[..., :3, 3] - base_pose_mat[:3, 3]
            rot = pose_mat[..., :3, :3] @ np.linalg.inv(base_pose_mat[:3, :3])
            out = np.copy(pose_mat)
            out[..., :3, :3] = rot
            out[..., :3, 3] = pos
            return out
        elif pose_rep == "relative":
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out
        elif pose_rep == "delta":
            all_pos = np.concatenate([base_pose_mat[None, :3, 3], pose_mat[..., :3, 3]], axis=0)
            out_pos = np.diff(all_pos, axis=0)

            all_rot_mat = np.concatenate([base_pose_mat[None, :3, :3], pose_mat[..., :3, :3]], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            out_rot = np.matmul(curr_rot, prev_rot)

            out = np.copy(pose_mat)
            out[..., :3, :3] = out_rot
            out[..., :3, 3] = out_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == "abs":
            return pose_mat
        elif pose_rep == "rel":
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[..., :3, 3] + base_pose_mat[:3, 3]
            rot = pose_mat[..., :3, :3] @ base_pose_mat[:3, :3]
            out = np.copy(pose_mat)
            out[..., :3, :3] = rot
            out[..., :3, 3] = pos
            return out
        elif pose_rep == "relative":
            out = base_pose_mat @ pose_mat
            return out
        elif pose_rep == "delta":
            output_pos = np.cumsum(pose_mat[..., :3, 3], axis=0) + base_pose_mat[:3, 3]

            output_rot_mat = np.zeros_like(pose_mat[..., :3, :3])
            curr_rot = base_pose_mat[:3, :3]
            for i in range(len(pose_mat)):
                curr_rot = pose_mat[i, :3, :3] @ curr_rot
                output_rot_mat[i] = curr_rot

            out = np.copy(pose_mat)
            out[..., :3, :3] = output_rot_mat
            out[..., :3, 3] = output_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")


def mat_to_rot6d(mat):
    """Flatten the first two rotation-matrix rows, following UMI's convention."""
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def mat_to_pose10d(mat):
    """Return position + rot6d (nine values); append gripper to obtain ten."""
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10


def normalize(x, x_min, x_max):
    """Map a range to [-1, 1], mapping constant dimensions to zero."""
    # Normalize x to [-1, 1] range based on x_min and x_max.
    span = np.asarray(x_max) - np.asarray(x_min)
    constant = np.abs(span) <= 1e-8
    return np.where(constant, 0.0, 2 * (x - x_min) / np.where(constant, 1.0, span) - 1)


def denormalize(x_norm, x_min, x_max):
    """Restore the original scalar range."""
    # Denormalize x_norm from [-1, 1] range back to original range based on x_min and x_max.
    return (x_norm + 1) / 2 * (x_max - x_min) + x_min
