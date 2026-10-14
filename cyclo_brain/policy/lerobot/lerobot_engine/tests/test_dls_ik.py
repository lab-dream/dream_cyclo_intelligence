"""DLS through the existing relative EEF adapter and checkpoint gate."""

import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lerobot_engine.relative_eef import RelativeEEFAdapter
from lerobot_engine.loading import LoadingMixin
from lerobot_engine.tests import test_relative_eef as native
from robot_client.robot_client import robot_schema


class DLSRelativeEEFTest(unittest.TestCase):
    # Reuse the native regression fixture and independent FK assertion.
    chunk = native.SharedLiftIKTest.chunk
    assert_targets = native.SharedLiftIKTest.assert_targets

    def setUp(self):
        section = robot_schema.load_robot_section("ffw_sg2_rev1")
        self.names = section["observation"]["state"]["upper_body"]["joint_names"]
        self.state_names = self.names + ["linear_x", "linear_y", "angular_z"]
        self.model_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.model_dir.cleanup)
        metadata = Path(self.model_dir.name) / "dataset_meta"
        metadata.mkdir()
        feature = {"shape": [22], "names": self.state_names}
        (metadata / "info.json").write_text(json.dumps({"features": {
            "observation.state": feature, "action": feature}}))
        (metadata / "stats.json").write_text(json.dumps({key: {
            "min": [0.] * 22, "max": [1.] * 22} for key in ("observation.state", "action")}))
        self.robot = SimpleNamespace(_action_groups=robot_schema.get_action_groups(section),
                                     get_joint_names=lambda group: self.names)
        self.measured = dict.fromkeys(self.state_names, 0.)
        self.measured.update(arm_l_joint2=.5, arm_r_joint2=-.5, arm_l_joint4=-1.2,
                             arm_r_joint4=-1.2, lift_joint=-.25,
                             gripper_l_joint1=.3, gripper_r_joint1=.4)
        self.adapter = RelativeEEFAdapter(self.model_dir.name, self.robot, "ffw_sg2_rev1", "dls")
        self.addCleanup(self.adapter.close)
        raw = self.adapter.raw_state({"follower_upper_body": np.array([self.measured[name] for name in self.names])})
        self.adapter.encode_observation({"observation.state": torch.tensor(np.stack([raw, raw])[None])})
        self.anchor = self.adapter.transform.poses(raw, "observation.state")[0]

    def test_waypoints_reach_both_tcp_targets_and_keep_gripper_values(self):
        chunk, poses = self.chunk([([.01, 0, .01], [-.01, .005, .02]),
                                  ([.015, 0, .015], [-.015, .01, .025])])
        commands = self.adapter.decode_action(chunk)
        self.assertEqual(commands.shape, (2, 17))
        self.assert_targets(commands, poses)
        np.testing.assert_allclose(commands[:, [7, 15]], [[.3, .4], [.3, .4]], atol=1e-6)
        self.assertEqual(self.adapter.kinematics.names, self.adapter._active_names)
        reference = self.adapter.kinematics.reference.copy()
        self.adapter.decode_action(chunk)
        np.testing.assert_array_equal(reference, self.adapter.kinematics.reference)

    def test_unreachable_chunk_is_not_accepted(self):
        chunk, _ = self.chunk([([5, 0, 0], [0, 0, 0])])
        with self.assertRaisesRegex(ValueError, "DLS IK target did not converge"):
            self.adapter.decode_action(chunk)

    def test_nonrelative_checkpoints_reject_dls(self):
        loader = LoadingMixin()
        request = SimpleNamespace(ik_backend="dls")
        for config in ({"type": "act"}, {"type": "diffusion"},
                       {"type": "diffusion", "state": "joint"}):
            (Path(self.model_dir.name) / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "DLS IK requires a Diffusion"):
                loader._configure_relative_eef(self.model_dir.name, "ffw_sg2_rev1", request)


if __name__ == "__main__":
    unittest.main()
