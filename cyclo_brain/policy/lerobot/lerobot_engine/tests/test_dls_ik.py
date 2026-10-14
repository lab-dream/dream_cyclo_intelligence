"""DLS through the existing relative EEF adapter and checkpoint gate."""

import json
import threading
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from lerobot_engine.engine import LeRobotEngine
from lerobot_engine.relative_eef import RelativeEEFAdapter
from lerobot_engine.loading import LoadingMixin
from lerobot_engine.io_mapping import IoMappingMixin
from lerobot_engine.tests import test_relative_eef as native
from robot_client import RobotClient
from robot_client.robot_client import derive_robot_config, robot_schema


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
        self.assertEqual(commands.shape, (2, 16))
        self.assert_targets(np.c_[commands, np.full(len(commands), self.measured['lift_joint'])], poses)
        np.testing.assert_allclose(commands[:, [7, 15]], [[.3, .4], [.3, .4]], atol=1e-6)
        self.assertEqual(self.adapter.kinematics.names, self.adapter._active_names)
        reference = self.adapter.kinematics.reference.copy()
        self.adapter.decode_action(chunk)
        np.testing.assert_array_equal(reference, self.adapter.kinematics.reference)

    def test_selected_arm_reuses_dls_and_locks_the_other_arm(self):
        for side, index in (("left", 0), ("right", 1)):
            with self.subTest(side=side):
                self.adapter.command_arms = (side,)
                offsets = [[5, 0, 0], [5, 0, 0]]
                offsets[index] = [.01, 0, .01]
                chunk, poses = self.chunk([offsets, offsets])
                commands = self.adapter.decode_action(chunk)
                other = 1 - index
                names = self.adapter._arm_names['right' if side == 'left' else 'left']
                np.testing.assert_array_equal(commands[:, other*8:other*8+7],
                    np.tile([self.measured[name] for name in names], (2, 1)))
                poses[:, other] = self.anchor[other]
                self.assert_targets(np.c_[commands, np.full(len(commands), self.measured['lift_joint'])], poses)
                self.assertTrue(self.adapter.kinematics._free_joints.all())

    def test_unreachable_chunk_is_not_accepted(self):
        chunk, _ = self.chunk([([5, 0, 0], [0, 0, 0])])
        with self.assertRaisesRegex(ValueError, "DLS IK target did not converge"):
            self.adapter.decode_action(chunk)

    def test_bimanual_right_ignores_left_policy_target_and_selects_right_gripper(self):
        engine = IoMappingMixin()
        engine._policy = SimpleNamespace(config=SimpleNamespace(type="diffusion"))
        engine._robot = self.robot
        engine._relative_eef = self.adapter
        engine._action_keys = list(self.adapter.action_keys)
        engine._configure_arm_selection(SimpleNamespace(arm_selection="bimanual_right"))
        self.assertEqual(engine._action_keys, ["arm_right"])
        self.assertEqual(engine._arm_selection, "bimanual_right")
        self.assertEqual(self.adapter.command_arms, ("right",))

        # The unused left prediction is unreachable; only the right TCP moves.
        chunk, poses = self.chunk([([5, 0, 0], [-.01, .005, .02]),
                                  ([5, 0, 0], [-.015, .01, .025])])
        chunk[:, 19] = .5  # Right gripper in the original 20-D policy output.
        full = self.adapter.decode_action(chunk)
        commands = engine._select_action_chunk(full)
        self.assertEqual(commands.shape, (2, 8))
        np.testing.assert_allclose(commands[:, 7], .75, atol=1e-6)
        left = [self.measured[name] for name in self.adapter._arm_names["left"]]
        np.testing.assert_array_equal(full[:, :7], np.tile(left, (2, 1)))
        poses[:, 0] = self.anchor[0]
        self.assert_targets(np.c_[full, np.full(2, self.measured["lift_joint"])], poses)

        # Changing every left action component cannot change a right command.
        chunk[:, :10] = [10, -10, 10, 0, 1, 0, 0, 0, 1, 1]
        np.testing.assert_allclose(
            engine._select_action_chunk(self.adapter.decode_action(chunk)), commands,
            atol=1e-10,
        )

    def test_bimanual_right_requires_relative_dls(self):
        for adapter in (None, SimpleNamespace(ik_backend="nodeik"),
                        SimpleNamespace(ik_backend="cyclo_control")):
            with self.subTest(adapter=adapter):
                engine = IoMappingMixin()
                engine._policy = SimpleNamespace(config=SimpleNamespace(type="diffusion"))
                engine._relative_eef = adapter
                engine._action_keys = ["arm_left", "arm_right"]
                with self.assertRaisesRegex(ValueError, "bimanual_right requires.*DLS"):
                    engine._configure_arm_selection(SimpleNamespace(arm_selection="bimanual_right"))

    def test_nonrelative_checkpoints_reject_dls(self):
        loader = LoadingMixin()
        request = SimpleNamespace(ik_backend="dls")
        for config in ({"type": "act"}, {"type": "diffusion"},
                       {"type": "diffusion", "state": "joint"}):
            (Path(self.model_dir.name) / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "DLS IK requires a Diffusion"):
                loader._configure_relative_eef(self.model_dir.name, "ffw_sg2_rev1", request)

    def wrist_engine(self, side):
        """Exercise the live RobotClient callback without opening robot transport."""
        section = robot_schema.load_robot_section("ffw_sg2_rev1")
        robot = RobotClient.__new__(RobotClient)
        robot._closed = True
        robot._config = derive_robot_config(section)
        robot._action_groups = robot_schema.get_action_groups(section)
        robot._lock = threading.Lock()
        robot._images = {}
        robot._image_timestamps = {}
        robot._joint_positions = {"follower_upper_body": np.array([self.measured[name] for name in self.names])}
        images = {}
        source = np.arange(240 * 424 * 3, dtype=np.uint8).reshape(240, 424, 3)
        for camera_side, rgb in (("left", source), ("right", 255 - source)):
            camera = f"cam_{camera_side}_wrist"
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.assertTrue(ok)
            robot._update_image(camera, SimpleNamespace(data=encoded.tobytes()))
            images[camera] = rgb
        key = f"observation.images.rgb.cam_{side}_wrist"
        engine = LeRobotEngine()
        engine._robot = robot
        engine._relative_eef = self.adapter
        engine._device = torch.device("cpu")
        engine._policy = SimpleNamespace(config=SimpleNamespace(
            type="diffusion", n_obs_steps=2, resize_shape=None, crop_shape=None,
            input_features={"observation.state": SimpleNamespace(shape=(20,)),
                            key: SimpleNamespace(shape=(3, 424, 240))}))
        engine._cameras = engine._resolve_camera_mappings(robot.camera_names, {key})
        engine._image_resize = engine._infer_image_resize(engine._policy)
        engine._action_keys = list(self.adapter.action_keys)
        engine._observation_interval_s = 0
        return engine, key, images

    def test_single_wrist_live_observation_keeps_recorded_pixels_for_each_selected_arm(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                engine, key, images = self.wrist_engine(side)
                engine._configure_arm_selection(SimpleNamespace(arm_selection=side))
                self.assertEqual(engine._image_preprocessing, {"resize_mode": "native"})
                batch = engine._build_policy_observation("")
                self.assertEqual(set(batch), {key, "observation.state", "task"})
                self.assertEqual(batch[key].shape, (1, 2, 3, 424, 240))
                self.assertEqual(batch["observation.state"].shape, (1, 2, 20))
                # Recorder's transpose=2: one counter-clockwise turn, no other pixel changes.
                recorded = np.rot90(images[f"cam_{side}_wrist"], k=1).copy()
                expected = torch.from_numpy(recorded).permute(2, 0, 1).float() / 255
                torch.testing.assert_close(batch[key][0], expected.repeat(2, 1, 1, 1), rtol=0, atol=0)
                self.assertEqual(engine._action_keys, [f"arm_{side}"])

    def test_native_wrist_rejects_wrong_live_resolution_and_missing_selected_camera(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                engine, _, _ = self.wrist_engine(side)
                engine._configure_arm_selection(SimpleNamespace(arm_selection=side))
                camera = f"cam_{side}_wrist"
                engine._robot._images[camera] = np.zeros((480, 640, 3), dtype=np.uint8)
                result = engine._build_policy_observation("")
                self.assertFalse(result["success"])
                self.assertIn("Native image shape mismatch", result["message"])
                del engine._robot._images[camera]
                result = engine._build_policy_observation("")
                self.assertFalse(result["success"])
                self.assertIn(f"Missing camera frame: {camera}", result["message"])

    def test_single_wrist_checkpoint_must_match_selected_arm_and_keep_native_encoder(self):
        for side in ("left", "right"):
            engine, _, _ = self.wrist_engine(side)
            other = "right" if side == "left" else "left"
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, "requires a .*wrist checkpoint"):
                engine._configure_arm_selection(SimpleNamespace(arm_selection=other))
            for field, value in (("resize_shape", (256, 256)), ("crop_shape", (200, 200))):
                engine, _, _ = self.wrist_engine(side)
                setattr(engine._policy.config, field, value)
                with self.subTest(side=side, field=field), self.assertRaisesRegex(ValueError, "no image resize or crop"):
                    engine._configure_arm_selection(SimpleNamespace(arm_selection=side))

    def test_existing_multi_camera_and_bimanual_image_recipes_are_preserved(self):
        for selection in ("both", "bimanual_right", "left", "right"):
            with self.subTest(selection=selection):
                engine, key, _ = self.wrist_engine("right")
                if selection in {"left", "right"}:
                    engine._policy.config.input_features["observation.images.rgb.cam_left_wrist"] = (
                        SimpleNamespace(shape=(3, 256, 256)))
                    engine._policy.config.input_features[key].shape = (3, 256, 256)
                recipe = {"resize_mode": "letterbox", "padding_alignment": "center"}
                engine._image_preprocessing = recipe.copy()
                engine._configure_arm_selection(SimpleNamespace(arm_selection=selection))
                self.assertEqual(engine._image_preprocessing, recipe)


if __name__ == "__main__":
    unittest.main()
