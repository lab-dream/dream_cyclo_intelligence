"""Check prescribed TCP geometry, actual camera axes and fixed non-arm state."""

import json
from pathlib import Path

import numpy as np

from dls_ik import DLSIK
from dls_ik.pose_constraints import floor_camera_initial_pose


def test_floor_camera_initial_pose_satisfies_geometry_and_preserves_other_values():
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "config/initial_state_first_frame_means.json").read_text())
    seed = dict(zip(reference["names"], reference["state"], strict=True))
    urdf = root.parents[1] / "policy/lerobot/lerobot_engine/_vendor/dream/ai_worker/ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
    if not urdf.is_file():
        urdf = Path("/app/lerobot_engine/_vendor/dream/ai_worker/ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf")
    result, _ = floor_camera_initial_pose(urdf, seed)
    ik = DLSIK(urdf)
    before = np.array([T[:3, 3] for T in ik.forward(seed)])
    after = np.array([T[:3, 3] for T in ik.forward(result)])
    np.testing.assert_allclose(after[:, [0, 2]], np.tile(before[:, [0, 2]].mean(axis=0), (2, 1)), atol=1e-6)
    assert after[0, 1] > 0 and after[1, 1] < 0
    assert after[0, 0] > .4 and after[1, 0] > .4
    np.testing.assert_allclose(np.linalg.norm(after[0] - after[1]), np.linalg.norm(before[0] - before[1]) + .15, atol=1e-6)
    for side in ("left", "right"):
        frame = ik.model.getFrameId(f"camera_{side}_link")
        np.testing.assert_allclose(ik.data.oMf[frame].rotation[:, 0], [0, 0, -1], atol=1e-5)
    for name in seed:
        if not name.startswith(("arm_l_joint", "arm_r_joint")):
            assert result[name] == seed[name]
    for side in ("l", "r"):
        assert abs(result[f"arm_{side}_joint4"] - seed[f"arm_{side}_joint4"]) < np.deg2rad(2)
    for name, lower, upper in zip(ik.names, ik.lower, ik.upper):
        assert lower <= result[name] <= upper
