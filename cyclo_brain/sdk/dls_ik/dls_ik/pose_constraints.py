"""Build a fixed-lift pose with hands side by side in front and cameras down.

Camera body links follow the ROS camera convention: local +X looks forward.
https://github.com/realsenseai/realsense-ros#ros2robot-vs-opticalcamera-coordination-systems
"""

import numpy as np
from scipy.optimize import least_squares

from .solver import DLSIK


def floor_camera_initial_pose(urdf_path, seed, separation_increase=0.15):
    """Change only 14 arm joints; preserve every other seed value exactly.

    Both TCPs use the average of their original x and z coordinates. Their
    distance grows by separation_increase along lateral y, with the left hand
    at positive y and the right at negative y. Camera yaw about the viewing axis
    remains free, giving room to preserve elbow posture.
    """
    if not np.isfinite(separation_increase) or separation_increase <= 0:
        raise ValueError("TCP separation increase must be positive and finite")
    ik = DLSIK(urdf_path)
    q0 = ik._configuration(seed)
    before = np.array([T[:3, 3] for T in ik.forward(seed)])
    center = before.mean(axis=0)
    old_distance = float(np.linalg.norm(before[0] - before[1]))
    distance = old_distance + separation_increase
    free = [i for i, name in enumerate(ik.names) if name != "lift_joint"]
    free_names = [ik.names[i] for i in free]
    indices = ik.q_indices[free]
    original = q0[indices].copy()
    camera_ids = [ik._frame_id(f"camera_{side}_link") for side in ("left", "right")]
    before_axes = [ik.data.oMf[frame].rotation[:, 0].copy() for frame in camera_ids]
    before_elbows = ik._elbows().copy()
    weights = np.ones(len(free))
    for side in ("l", "r"):
        weights[free_names.index(f"arm_{side}_joint4")] = 3.0

    targets = np.tile(center, (2, 1))
    targets[:, 1] = [center[1] + distance / 2, center[1] - distance / 2]

    def residual(x):
        q = q0.copy()
        q[indices] = x
        ik._update(q)
        primary = []
        for tcp, camera, target in zip(ik.tcp_ids, camera_ids, targets):
            primary.extend(1000 * (ik.data.oMf[tcp].translation - target))
            primary.extend(300 * (ik.data.oMf[camera].rotation[:, 0] - [0, 0, -1]))
        # Secondary terms select a pose close to the previous arms and elbows.
        # Primary constraints are checked explicitly before accepting the result.
        return np.r_[primary, .03 * weights * (x - original),
                     .3 * (ik._elbows() - before_elbows).ravel()]

    fit = least_squares(
        residual, original,
        bounds=(ik.lower[free], ik.upper[free]),
        max_nfev=1000, gtol=1e-11, ftol=1e-12, xtol=1e-12,
    )
    result = dict(seed)
    result.update(zip(free_names, fit.x.tolist(), strict=True))
    after = np.array([T[:3, 3] for T in ik.forward(result)])
    after_axes = np.array([ik.data.oMf[frame].rotation[:, 0].copy() for frame in camera_ids])
    position_error = float(np.linalg.norm(after - targets, axis=1).max())
    direction_error = float(np.linalg.norm(after_axes - [0, 0, -1], axis=1).max())
    if position_error > 1e-6 or direction_error > 1e-5:
        raise RuntimeError(
            f"Cannot satisfy TCP/camera constraints: position={position_error:.6f}m, "
            f"viewing-axis error={direction_error:.6f}"
        )
    actual_distance = float(np.linalg.norm(after[0] - after[1]))
    return result, {
        "original_tcp_positions_m": before.tolist(),
        "target_tcp_positions_m": targets.tolist(),
        "actual_tcp_positions_m": after.tolist(),
        "layout": "side_by_side_in_front_of_chest", "separation_axis": "y",
        "common_x_m": float(center[0]), "common_z_m": float(center[2]),
        "center_y_m": float(center[1]),
        "original_distance_m": old_distance, "target_distance_m": distance,
        "actual_distance_m": actual_distance,
        "separation_increase_m": separation_increase,
        "original_camera_forward_world": [axis.tolist() for axis in before_axes],
        "camera_forward_world": after_axes.tolist(),
        "camera_forward_axis": "+X in camera_left_link/camera_right_link",
        "camera_down_error_deg": np.rad2deg(np.arccos(np.clip(-after_axes[:, 2], -1, 1))).tolist(),
        "elbow_angle_change_rad": [float(result[f"arm_{side}_joint4"] - seed[f"arm_{side}_joint4"]) for side in ("l", "r")],
        "position_error_max_m": position_error, "viewing_axis_error_max": direction_error,
        "evaluations": fit.nfev, "hardware_publish": False,
    }
