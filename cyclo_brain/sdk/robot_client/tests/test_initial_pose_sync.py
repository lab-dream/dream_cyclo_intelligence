#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, **kwargs) -> None:
        self.messages.append(kwargs)


class FakeMessage:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


zenoh_stub = types.ModuleType("zenoh_ros2_sdk")
zenoh_stub.ROS2Publisher = object
zenoh_stub.ROS2Subscriber = object
zenoh_stub.get_message_class = lambda _name: FakeMessage
sys.modules.setdefault("zenoh_ros2_sdk", zenoh_stub)

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robot_client"
    / "robot_client.py"
)
ROBOT_CONFIG_PATH = MODULE_PATH.parents[4] / "shared" / "shared" / "robot_configs"
sys.path.insert(0, str(ROBOT_CONFIG_PATH))
spec = importlib.util.spec_from_file_location("robot_client_impl", MODULE_PATH)
robot_client_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robot_client_impl)
RobotClient = robot_client_impl.RobotClient


class InitialPoseSyncCommandTest(unittest.TestCase):
    def _make_client(self, robot_type: str) -> RobotClient:
        section = robot_client_impl.robot_schema.load_robot_section(robot_type)
        action_groups = robot_client_impl.robot_schema.get_action_groups(section)
        client = RobotClient.__new__(RobotClient)
        client._config = robot_client_impl._build_runtime_config(section)
        client._action_groups = action_groups
        client._action_keys = sorted(action_groups)
        client._command_publishers = {}
        client._command_joint_names = {}
        client._joint_positions_by_name = {}
        client._joint_position_timestamps_by_name = {}
        client._joint_positions = {}
        client._joint_velocities = {}
        client._joint_efforts = {}
        client._joint_timestamps = {}
        client._joint_children = {}
        client._lock = threading.Lock()
        client._cmd_vel_linear_deadband = 0.0
        client._cmd_vel_angular_deadband = 0.0
        client._initial_pose_sync_state_max_age_s = 1.0
        client._closed = True
        for key, cfg in action_groups.items():
            publisher_key = f"leader_{key}"
            client._command_publishers[publisher_key] = FakePublisher()
            client._command_joint_names[publisher_key] = list(
                cfg.get("joint_names", [])
            )
        return client

    def test_joint_state_max_age_environment_override_and_fallback(self) -> None:
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            with mock.patch.dict(
                os.environ,
                {"INITIAL_POSE_SYNC_STATE_MAX_AGE_S": "2.5"},
            ):
                client = RobotClient("omy_f3m")
                self.assertEqual(client._initial_pose_sync_state_max_age_s, 2.5)

            for invalid in ("0", "-1", "nan", "invalid"):
                with self.subTest(invalid=invalid), mock.patch.dict(
                    os.environ,
                    {"INITIAL_POSE_SYNC_STATE_MAX_AGE_S": invalid},
                ):
                    client = RobotClient("omy_f3m")
                    self.assertEqual(client._initial_pose_sync_state_max_age_s, 1.0)

    @staticmethod
    def _action_dimension(client: RobotClient, action_keys: list[str]) -> int:
        total = 0
        for key in action_keys:
            cfg = client._action_groups[key]
            total += (
                3
                if cfg["msg_type"] == "geometry_msgs/msg/Twist"
                else len(cfg["joint_names"])
            )
        return total

    @staticmethod
    def _seed_joint_state(client: RobotClient) -> None:
        value = 0.0
        received_at = time.monotonic()
        for cfg in client._action_groups.values():
            if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                continue
            for name in reversed(cfg["joint_names"]):
                client._joint_positions_by_name[name] = value
                client._joint_position_timestamps_by_name[name] = received_at
                value += 0.1

    def test_supported_robot_layouts_publish_all_position_groups_and_zero_mobile(self):
        for robot_type in ("ffw_sg2_rev1", "ffw_sh5_rev1", "omy_f3m"):
            with self.subTest(robot_type=robot_type):
                client = self._make_client(robot_type)
                action_keys = list(client._action_keys)
                self._seed_joint_state(client)
                action = np.arange(
                    self._action_dimension(client, action_keys),
                    dtype=np.float64,
                )

                client.publish_initial_pose_sync(
                    action,
                    action_keys,
                    duration_s=5.5,
                )

                for key in action_keys:
                    publisher = client._command_publishers[f"leader_{key}"]
                    self.assertEqual(len(publisher.messages), 1)
                    message = publisher.messages[0]
                    cfg = client._action_groups[key]
                    if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                        self.assertEqual(message["linear"].x, 0.0)
                        self.assertEqual(message["linear"].y, 0.0)
                        self.assertEqual(message["angular"].z, 0.0)
                    else:
                        point = message["points"][0]
                        self.assertEqual(point.time_from_start.sec, 5)
                        self.assertEqual(point.time_from_start.nanosec, 500_000_000)
                        self.assertEqual(message["joint_names"], cfg["joint_names"])

    def test_current_pose_hold_uses_joint_names_in_config_order(self):
        client = self._make_client("omy_f3m")
        action_keys = list(client._action_keys)
        joint_names = client._action_groups["arm"]["joint_names"]
        reversed_names = list(reversed(joint_names))
        client._update_joint(
            "follower_arm",
            SimpleNamespace(
                name=reversed_names,
                position=[float(index + 1) for index in range(len(reversed_names))],
                velocity=[],
                effort=[],
            ),
        )

        client.publish_current_pose_hold(action_keys, duration_s=0.1)

        message = client._command_publishers["leader_arm"].messages[0]
        expected = [client._joint_positions_by_name[name] for name in joint_names]
        np.testing.assert_allclose(message["points"][0].positions, expected)
        self.assertEqual(message["points"][0].time_from_start.nanosec, 100_000_000)

    def test_relative_eef_action_publishes_one_lift_position_in_meters(self):
        client = self._make_client("ffw_sg2_rev1")
        action = np.concatenate([np.arange(16) * 0.01, [-0.25]])
        client.publish_action(action, ["arm_left", "arm_right", "lift"])
        for key, offset in (("arm_left", 0), ("arm_right", 8), ("lift", 16)):
            messages = client._command_publishers[f"leader_{key}"].messages
            self.assertEqual(len(messages), 1)
            width = 1 if key == "lift" else 8
            np.testing.assert_allclose(messages[0]["points"][0].positions, action[offset:offset + width])
        self.assertEqual(client._command_publishers["leader_lift"].messages[0]["joint_names"], ["lift_joint"])
        self.assertFalse(client._command_publishers["leader_head"].messages)
        self.assertFalse(client._command_publishers["leader_mobile"].messages)

    def test_relative_eef_initial_pose_sync_includes_the_shared_lift(self):
        client = self._make_client("ffw_sg2_rev1")
        self._seed_joint_state(client)
        action = np.concatenate([np.arange(16) * 0.01, [-0.25]])
        client.publish_initial_pose_sync(
            action, ["arm_left", "arm_right", "lift"], duration_s=5.0
        )
        message = client._command_publishers["leader_lift"].messages[0]
        self.assertEqual(message["joint_names"], ["lift_joint"])
        np.testing.assert_allclose(message["points"][0].positions, [-0.25])
        self.assertEqual(message["points"][0].time_from_start.sec, 5)

    def test_missing_current_joint_state_prevents_any_sync_command(self):
        client = self._make_client("omy_f3m")
        action_keys = list(client._action_keys)
        action = np.zeros(self._action_dimension(client, action_keys))

        with self.assertRaisesRegex(RuntimeError, "current joint state unavailable"):
            client.publish_initial_pose_sync(action, action_keys, duration_s=5.0)

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_stale_current_joint_state_prevents_any_sync_command(self):
        client = self._make_client("omy_f3m")
        action_keys = list(client._action_keys)
        self._seed_joint_state(client)
        stale_at = time.monotonic() - 1.1
        client._joint_position_timestamps_by_name = {
            name: stale_at for name in client._joint_positions_by_name
        }
        action = np.zeros(self._action_dimension(client, action_keys))

        with self.assertRaisesRegex(RuntimeError, "current joint state stale"):
            client.publish_initial_pose_sync(action, action_keys, duration_s=5.0)

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_one_stale_joint_prevents_hold_and_twist_commands(self):
        client = self._make_client("ffw_sg2_rev1")
        action_keys = list(client._action_keys)
        self._seed_joint_state(client)
        stale_name = next(iter(client._joint_positions_by_name))
        client._joint_position_timestamps_by_name[stale_name] = time.monotonic() - 1.1

        with self.assertRaisesRegex(RuntimeError, stale_name):
            client.publish_current_pose_hold(action_keys, duration_s=0.1)

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_invalid_action_layout_is_rejected_before_publish(self):
        client = self._make_client("omy_f3m")
        action_keys = list(client._action_keys)
        self._seed_joint_state(client)
        expected_dim = self._action_dimension(client, action_keys)

        invalid_actions = [
            np.zeros(expected_dim - 1),
            np.zeros(expected_dim + 1),
            np.full(expected_dim, np.nan),
            np.full(expected_dim, np.inf),
        ]
        for action in invalid_actions:
            with self.subTest(size=len(action)):
                with self.assertRaises(ValueError):
                    client.publish_initial_pose_sync(
                        action,
                        action_keys,
                        duration_s=5.0,
                    )

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_unknown_action_key_is_rejected_before_publish(self):
        client = self._make_client("omy_f3m")
        self._seed_joint_state(client)

        with self.assertRaisesRegex(ValueError, "unknown action key"):
            client.publish_initial_pose_sync(
                np.zeros(1),
                ["missing_action_group"],
                duration_s=5.0,
            )

        self.assertTrue(
            all(not publisher.messages for publisher in client._command_publishers.values())
        )

    def test_embedded_inference_request_matches_native_service_field_order(self):
        messages_path = MODULE_PATH.parent / "messages" / "__init__.py"
        messages_spec = importlib.util.spec_from_file_location(
            "robot_client_messages_impl",
            messages_path,
        )
        messages = importlib.util.module_from_spec(messages_spec)
        messages_spec.loader.exec_module(messages)

        native_request = (
            MODULE_PATH.parents[4] / "interfaces" / "srv" / "InferenceCommand.srv"
        ).read_text(encoding="utf-8").split("---", maxsplit=1)[0]

        def serialized_fields(definition: str) -> list[str]:
            fields = []
            for raw_line in definition.splitlines():
                line = raw_line.split("#", maxsplit=1)[0].strip()
                if not line or "=" in line:
                    continue
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_/\[\]]*\s+[a-z][a-z0-9_]*", line):
                    fields.append(line)
            return fields

        self.assertEqual(
            serialized_fields(messages.INFERENCE_COMMAND_REQUEST_DEF),
            serialized_fields(native_request),
        )


class JointStateMappingTest(unittest.TestCase):
    def setUp(self):
        self.client = InitialPoseSyncCommandTest()._make_client("ffw_sg2_rev1")
        self.parent = "follower_upper_body"
        groups = self.client._config["joint_groups"]
        self.children = [name for name, cfg in groups.items() if cfg.get("parent") == self.parent]
        self.client._joint_children = {self.parent: self.children}
        self.names = (
            [f"arm_l_joint{i}" for i in range(1, 8)]
            + [f"arm_r_joint{i}" for i in range(1, 8)]
            + ["gripper_l_joint1", "gripper_r_joint1", "head_joint1", "head_joint2",
               "left_wheel_drive", "left_wheel_steer", "lift_joint", "rear_wheel_drive",
               "rear_wheel_steer", "right_wheel_drive", "right_wheel_steer"]
        )

    def update(self, names, position=None):
        self.client._update_joint(
            self.parent,
            SimpleNamespace(
                name=names,
                position=list(range(len(names))) if position is None else position,
                velocity=[100 + i for i in range(len(names))],
                effort=[200 + i for i in range(len(names))],
            ),
        )

    def test_extra_wheels_and_reordered_joints_match_every_configured_view(self):
        self.update(self.names)
        self.assertEqual(len(self.names), 25)
        self.assertEqual(len(self.client.get_joint_positions(self.parent)), 19)
        for view in [self.parent, *self.children]:
            indices = [self.names.index(name) for name in self.client.get_joint_names(view)]
            np.testing.assert_array_equal(self.client.get_joint_positions(view), indices)
            np.testing.assert_array_equal(self.client.get_joint_velocities(view), np.array(indices) + 100)
            np.testing.assert_array_equal(self.client.get_joint_efforts(view), np.array(indices) + 200)
        self.assertEqual(self.client._joint_positions_by_name["gripper_l_joint1"], 14)
        self.assertIn("left_wheel_drive", self.client._joint_positions_by_name)
        self.assertTrue(all(not publisher.messages for publisher in self.client._command_publishers.values()))

    def test_same_length_message_also_uses_names_instead_of_array_order(self):
        names = list(reversed(self.client.get_joint_names(self.parent)))
        self.update(names)
        np.testing.assert_array_equal(self.client.get_joint_positions(self.parent), list(reversed(range(19))))

    def test_missing_joint_removes_incomplete_views_instead_of_retaining_old_pose(self):
        self.update(self.names)
        self.update([name for name in self.names if name != "lift_joint"])
        self.assertNotIn(self.parent, self.client.get_joint_positions())
        self.assertNotIn("follower_lift", self.client.get_joint_positions())
        self.assertNotIn(self.parent, self.client._joint_timestamps)
        self.assertEqual(len(self.client.get_joint_positions("follower_arm_left")), 8)

    def test_malformed_positions_do_not_create_truncated_or_stale_pose(self):
        self.update(self.names)
        timestamps = dict(self.client._joint_position_timestamps_by_name)
        self.update(self.names, position=list(range(24)))
        self.assertEqual(self.client.get_joint_positions(), {})
        self.assertEqual(self.client._joint_position_timestamps_by_name, timestamps)

    def test_duplicate_or_absent_names_cannot_be_used_as_joint_positions(self):
        for names in (["arm_l_joint1"] * 25, []):
            with self.subTest(names=names):
                self.update(self.names)
                self.update(names, position=list(range(25)))
                self.assertEqual(self.client.get_joint_positions(), {})


if __name__ == "__main__":
    unittest.main()
