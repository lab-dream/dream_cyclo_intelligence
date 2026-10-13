"""Check the backend boundary without a checkpoint or robot transport."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from lerobot_engine.nodeik import NodeIK
from lerobot_engine.relative_eef import RelativeEEFAdapter
from robot_client.robot_client import robot_schema


class NodeIKTest(unittest.TestCase):
    def test_folder_selection_prefers_pose_and_accepts_explicit_vision_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("pose", "vision"):
                (root / mode).mkdir()
                (root / mode / "best.pt").touch()
            self.assertEqual(NodeIK.resolve_model_path(root), root / "pose/best.pt")
            self.assertEqual(NodeIK.resolve_model_path(root / "vision"), root / "vision/best.pt")
            self.assertEqual(NodeIK.resolve_model_path(root / "pose/best.pt"), root / "pose/best.pt")
            with self.assertRaises(ValueError):
                NodeIK.resolve_model_path("")

    def test_relative_waypoints_reach_nodeik_as_absolute_left_right_tcp_and_keep_grippers(self):
        section = robot_schema.load_robot_section("ffw_sg2_rev1")
        names = section["observation"]["state"]["upper_body"]["joint_names"] + ["linear_x", "linear_y", "angular_z"]
        robot = SimpleNamespace(_action_groups=robot_schema.get_action_groups(section), camera_names=[])
        solved = {**{f"arm_l_joint{i}": i * 0.01 for i in range(1, 8)},
                  **{f"arm_r_joint{i}": -i * 0.02 for i in range(1, 8)}, "lift_joint": -0.2}
        targets = []
        def solve(seed, left, right):
            targets.append(np.stack([left, right]))
            return solved.copy(), 0.0, 0.0
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / "dataset_meta"
            meta.mkdir()
            feature = {"shape": [22], "names": names}
            (meta / "info.json").write_text(json.dumps({"features": {key: feature for key in ("observation.state", "action")}}))
            (meta / "stats.json").write_text(json.dumps({key: {"min": [0.0]*22, "max": [1.0]*22} for key in ("observation.state", "action")}))
            with patch("lerobot_engine.nodeik.NodeIK", return_value=SimpleNamespace(solve=solve, image_keys=[])) as constructor:
                adapter = RelativeEEFAdapter(directory, robot, "ffw_sg2_rev1", "nodeik", "/models/pose", "cpu")
            try:
                raw = np.zeros(22, dtype=np.float32)
                raw[names.index("lift_joint")] = -0.25
                adapter.measured = dict(zip(names, raw.tolist(), strict=True))
                adapter.encode_observation({"observation.state": torch.tensor(np.stack([raw, raw])[None])})
                anchors = adapter.transform.poses(raw, "observation.state")[0]
                absolute = np.repeat(anchors[None], 2, axis=0)
                absolute[0, 0, 0, 3] += .01
                absolute[1, 1, 1, 3] -= .02
                chunk = adapter.transform.encode(absolute, np.array([[-.4, .2], [.6, -.8]]), anchors)
                commands = adapter.decode_action(chunk)
                np.testing.assert_allclose(targets, absolute, atol=1e-7)
                np.testing.assert_allclose(commands[:, :7], np.tile(np.arange(1,8)*.01, (2,1)))
                np.testing.assert_allclose(commands[:, 8:15], np.tile(-np.arange(1,8)*.02, (2,1)))
                np.testing.assert_allclose(commands[:, [7,15]], [[.3,.6],[.8,.1]], atol=1e-7)
                np.testing.assert_allclose(commands[:,16], -.2)
                self.assertEqual(commands.shape, (2,17))
                self.assertEqual(constructor.call_args.args[0], "/models/pose")
            finally:
                adapter.close()
