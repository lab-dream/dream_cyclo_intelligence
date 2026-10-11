"""Relative EEF regression checks with in-memory sensors and no transport."""

from __future__ import annotations

import threading
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot_engine.relative_eef import RelativeEEFAdapter
from lerobot_engine._vendor.umi.real_inference_util import get_real_umi_action
from lerobot_engine._vendor.umi.pose_util import pose_to_mat
from robot_client import RobotClient
from robot_client.robot_client import derive_robot_config, robot_schema


class RelativeEEFTest(unittest.TestCase):
    def test_real_joint_callback_and_relative_raw_state_keep_grippers_in_order(self):
        section = robot_schema.load_robot_section("ffw_sg2_rev1")
        client = RobotClient.__new__(RobotClient)
        client._closed = True
        client._config = derive_robot_config(section)
        client._lock = threading.Lock()
        for field in ("_joint_positions", "_joint_velocities", "_joint_efforts",
                      "_joint_timestamps", "_joint_positions_by_name",
                      "_joint_position_timestamps_by_name"):
            setattr(client, field, {})
        parent = "follower_upper_body"
        client._joint_children = {parent: [name for name, cfg in client._config["joint_groups"].items()
                                          if cfg.get("parent") == parent]}
        names = client.get_joint_names(parent)
        received = list(reversed(names)) + [f"wheel_{i}" for i in range(6)]
        client._update_joint(parent, SimpleNamespace(name=received, position=list(range(25)), velocity=[], effort=[]))
        adapter = RelativeEEFAdapter.__new__(RelativeEEFAdapter)
        adapter.robot = client
        adapter.names = names + ["linear_x", "linear_y", "angular_z"]
        adapter.transform = SimpleNamespace(indices={"observation.state": {
            "left": [names.index(f"arm_l_joint{i}") for i in range(1, 8)],
            "right": [names.index(f"arm_r_joint{i}") for i in range(1, 8)],
            "lift": names.index("lift_joint"),
            "grippers": [names.index("gripper_l_joint1"), names.index("gripper_r_joint1")],
        }})
        state = adapter.raw_state(client.get_joint_positions())
        np.testing.assert_array_equal(state[:19], [received.index(name) for name in names])
        self.assertEqual(adapter.measured["gripper_l_joint1"], received.index("gripper_l_joint1"))
        self.assertEqual(state.shape, (22,))
        # Missing a required joint must fail instead of reusing the earlier lift.
        incomplete = [name for name in received if name != "lift_joint"]
        client._update_joint(parent, SimpleNamespace(name=incomplete, position=list(range(24)), velocity=[], effort=[]))
        with self.assertRaisesRegex(ValueError, "Missing measured joints.*lift_joint"):
            adapter.raw_state(client.get_joint_positions())

    def test_history_and_action_share_latest_measured_tcp_anchor(self):
        poses = np.tile(np.eye(4), (2, 2, 1, 1))
        for frame, side, angles, position in (
            (0, 0, [15, -20, 40], [0.1, 0.2, 0.3]),
            (1, 0, [-10, 30, 90], [0.3, -0.1, 0.4]),
            (0, 1, [35, 10, -15], [-0.2, 0.4, 0.1]),
            (1, 1, [0, -45, 60], [-0.1, 0.1, 0.5]),
        ):
            poses[frame, side, :3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            poses[frame, side, :3, 3] = position
        grippers = np.array([[-0.8, 0.1], [-0.2, 0.7]])
        adapter = RelativeEEFAdapter.__new__(RelativeEEFAdapter)
        adapter.transform = SimpleNamespace(poses=lambda joints, key: poses, grippers=lambda joints, key: grippers)
        adapter.shape_meta = {"obs": {
            f"robot{side}_{key}": {"type": "low_dim", "shape": [width]}
            for side in range(2)
            for key, width in (("eef_pos", 3), ("eef_rot_axis_angle", 6), ("gripper_width", 1))
        }}
        batch = {"observation.state": torch.zeros((1, 2, 22), dtype=torch.float64)}
        state = adapter.encode_observation(batch)["observation.state"].numpy()[0]
        self.assertEqual(state.shape, (2, 20))
        for side in range(2):
            relative = np.linalg.inv(poses[-1, side]) @ poses[:, side]
            expected = np.concatenate([relative[:, :3, 3], relative[:, :2, :3].reshape(2, 6), grippers[:, side:side + 1]], axis=-1)
            np.testing.assert_allclose(state[:, side * 10:(side + 1) * 10], expected, atol=1e-12)
            np.testing.assert_allclose(state[-1, side * 10:side * 10 + 9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-12)
        # Decode relative history as hypothetical actions to independently
        # verify that the same latest anchor restores both absolute poses.
        decoded = get_real_umi_action(state, adapter.env_obs, action_pose_repr="relative")
        for side in range(2):
            np.testing.assert_allclose(pose_to_mat(decoded[:, side * 7:side * 7 + 6]), poses[:, side], atol=1e-12)


class SharedLiftIKTest(unittest.TestCase):
    def setUp(self):
        section = robot_schema.load_robot_section("ffw_sg2_rev1")
        self.names = section["observation"]["state"]["upper_body"]["joint_names"]
        self.state_names = self.names + ["linear_x", "linear_y", "angular_z"]
        self.model_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.model_dir.cleanup)
        metadata = Path(self.model_dir.name) / "dataset_meta"
        metadata.mkdir()
        feature = {"shape": [22], "names": self.state_names}
        (metadata / "info.json").write_text(json.dumps({"features": {"observation.state": feature, "action": feature}}))
        stats = {key: {"min": [0.0] * 22, "max": [1.0] * 22} for key in ("observation.state", "action")}
        (metadata / "stats.json").write_text(json.dumps(stats))
        robot = SimpleNamespace(
            _action_groups=robot_schema.get_action_groups(section),
            get_joint_names=lambda group: self.names,
        )
        self.robot = robot
        # Keep the original high-lift cases as a regression reference.
        with patch.dict(os.environ, {"CYCLO_IK_LIFT_WEIGHT": "1e-6"}):
            self.adapter = RelativeEEFAdapter(self.model_dir.name, robot, "ffw_sg2_rev1")
        self.addCleanup(self.adapter.close)
        self.measured = dict.fromkeys(self.state_names, 0.0)
        self.measured.update(arm_l_joint2=0.5, arm_r_joint2=-0.5,
                             arm_l_joint4=-1.2, arm_r_joint4=-1.2,
                             gripper_l_joint1=0.3, gripper_r_joint1=0.4, lift_joint=-0.25)
        raw = self.adapter.raw_state({"follower_upper_body": np.array([self.measured[name] for name in self.names])})
        self.adapter.encode_observation({"observation.state": torch.tensor(np.stack([raw, raw])[None])})
        self.anchor = self.adapter.transform.poses(raw, "observation.state")[0]

    def chunk(self, offsets):
        poses = np.repeat(self.anchor[None], len(offsets), axis=0)
        for index, (left, right) in enumerate(offsets):
            poses[index, 0, :3, 3] += left
            poses[index, 1, :3, 3] += right
        grippers = np.tile([-0.4, -0.2], (len(offsets), 1))
        return self.adapter.transform.encode(poses, grippers, self.anchor), poses

    def assert_targets(self, commands, targets):
        for command, target in zip(commands, targets, strict=True):
            values = dict(self.measured)
            for side, offset in (("left", 0), ("right", 8)):
                for name, value in zip(self.adapter._arm_names[side], command[offset:offset + 7], strict=True):
                    values[name] = value
            values["lift_joint"] = command[16]
            joints = np.array([values[name] for name in self.state_names])
            reached = self.adapter.transform.poses(joints, "action")[0]
            np.testing.assert_allclose(reached[:, :3, 3], target[:, :3, 3], atol=1e-4)
            np.testing.assert_allclose(reached[:, :3, :3], target[:, :3, :3], atol=1e-3)

    def test_identity_target_preserves_arms_grippers_and_lift_in_native_units(self):
        chunk, poses = self.chunk([([0, 0, 0], [0, 0, 0])])
        commands = self.adapter.decode_action(chunk)
        self.assertEqual(self.adapter.action_keys, ["arm_left", "arm_right", "lift"])
        self.assertEqual(commands.shape, (1, 17))
        expected = [*[self.measured[name] for name in self.names[:16]], -0.25]
        np.testing.assert_allclose(commands[0], expected, atol=1e-6)
        self.assert_targets(commands, poses)

    def test_shared_vertical_targets_move_one_lift_and_reach_both_tcp_poses(self):
        chunk, poses = self.chunk([([0, 0, 0.03], [0, 0, 0.03]),
                                  ([0, 0, 0.06], [0, 0, 0.06])])
        commands = self.adapter.decode_action(chunk)
        self.assertGreater(commands[0, 16], -0.23)
        self.assertGreater(commands[1, 16], commands[0, 16])
        np.testing.assert_allclose(commands[:, 16], [-0.22, -0.19], atol=1e-3)
        self.assert_targets(commands, poses)

    def test_different_arm_targets_are_solved_with_the_same_lift(self):
        chunk, poses = self.chunk([([0, 0, 0], [0.01, 0, 0.02])])
        commands = self.adapter.decode_action(chunk)
        self.assertTrue(np.isfinite(commands).all())
        self.assert_targets(commands, poses)

    def test_default_weight_reduces_lift_motion_while_reaching_both_targets(self):
        chunk, poses = self.chunk([([0, 0, 0.03], [0, 0, 0.03])])
        previous = self.adapter.decode_action(chunk)
        with patch.dict(os.environ):
            os.environ.pop("CYCLO_IK_LIFT_WEIGHT", None)
            reduced = RelativeEEFAdapter(self.model_dir.name, self.robot, "ffw_sg2_rev1")
        self.addCleanup(reduced.close)
        self.assertEqual(reduced.lift_weight, 10.0)
        reduced.measured = dict(self.adapter.measured)
        reduced.env_obs = self.adapter.env_obs
        commands = reduced.decode_action(chunk)
        self.assertLess(abs(commands[0, 16] + 0.25), abs(previous[0, 16] + 0.25) * 0.25)
        self.assert_targets(commands, poses)

    def test_invalid_lift_weights_are_rejected(self):
        for value in ("0", "-0.01", "nan", "inf", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {"CYCLO_IK_LIFT_WEIGHT": value}):
                with self.assertRaises(ValueError):
                    RelativeEEFAdapter(self.model_dir.name, self.robot, "ffw_sg2_rev1")

    def test_lift_limit_is_respected_when_targets_request_more_travel(self):
        chunk, _ = self.chunk([([0, 0, 0.30], [0, 0, 0.30])])
        commands = self.adapter.decode_action(chunk)
        self.assertTrue(np.isfinite(commands).all())
        self.assertLessEqual(commands[0, 16], 0.0)
        self.assertGreaterEqual(commands[0, 16], -0.5)

    def test_nonfinite_action_is_rejected_before_solving(self):
        with self.assertRaisesRegex(ValueError, "finite relative EEF"):
            self.adapter.decode_action(np.full((1, 20), np.nan))


if __name__ == "__main__":
    unittest.main()
