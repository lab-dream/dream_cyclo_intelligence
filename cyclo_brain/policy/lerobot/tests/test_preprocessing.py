#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


ENGINE_DIR = Path(__file__).resolve().parents[1] / "lerobot_engine"
package = types.ModuleType("lerobot_engine")
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault("lerobot_engine", package)

constants_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.constants",
    ENGINE_DIR / "constants.py",
)
constants = importlib.util.module_from_spec(constants_spec)
sys.modules[constants_spec.name] = constants
constants_spec.loader.exec_module(constants)

image_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.image_preprocessing",
    ENGINE_DIR / "image_preprocessing.py",
)
image_preprocessing = importlib.util.module_from_spec(image_spec)
sys.modules[image_spec.name] = image_preprocessing
image_spec.loader.exec_module(image_preprocessing)

preprocessing_spec = importlib.util.spec_from_file_location(
    "lerobot_engine.preprocessing",
    ENGINE_DIR / "preprocessing.py",
)
preprocessing = importlib.util.module_from_spec(preprocessing_spec)
sys.modules[preprocessing_spec.name] = preprocessing
preprocessing_spec.loader.exec_module(preprocessing)

PreprocessingMixin = preprocessing.PreprocessingMixin
STATE_KEY = constants.STATE_KEY


class FakeRobot:
    _config = {"cameras": {}}

    def __init__(self, positions):
        self._positions = positions

    def get_images(self, format="rgb"):
        return {"unused": np.zeros((2, 2, 3), dtype=np.uint8)}

    def get_joint_positions(self):
        return {"follower_arm": self._positions}


class Preprocessor(PreprocessingMixin):
    def __init__(self, positions, expected):
        self._robot = FakeRobot(positions)
        self._cameras = {}
        self._state_modalities = ["arm"]
        self._image_resize = {}
        self._device = torch.device("cpu")
        feature = SimpleNamespace(shape=(expected,))
        config = SimpleNamespace(input_features={STATE_KEY: feature})
        self._policy = SimpleNamespace(config=config)

    def _fail(self, message):
        return {"error": message}


class PreprocessingTest(unittest.TestCase):
    def test_diffusion_captures_time_axis_at_dataset_fps(self):
        preprocessor = Preprocessor([1.0, 2.0], expected=2)
        preprocessor._policy.config.type = "diffusion"
        preprocessor._policy.config.n_obs_steps = 2
        preprocessor._observation_interval_s = 1.0 / 30.0
        original_build = preprocessor._build_observation
        frames = []

        def capture(task):
            frame = original_build(task)
            frames.append(frame)
            preprocessor._robot._positions = [3.0, 4.0]
            return frame

        preprocessor._build_observation = capture
        with patch.object(preprocessing.time, "monotonic", side_effect=[0.0, 0.01, 1.0]), \
                patch.object(preprocessing.time, "sleep") as sleep:
            batch = preprocessor._build_policy_observation("task")

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 1.0 / 30.0 - 0.01)
        np.testing.assert_allclose(batch[STATE_KEY], [[[1., 2.], [3., 4.]]])
        self.assertEqual(batch["task"], ["task"])
        self.assertEqual(len(frames), 2)

    def test_single_step_diffusion_keeps_time_axis(self):
        preprocessor = Preprocessor([1.0, 2.0], expected=2)
        preprocessor._policy.config.type = "diffusion"
        preprocessor._policy.config.n_obs_steps = 1
        preprocessor._observation_interval_s = 1.0 / 30.0
        self.assertEqual(
            tuple(preprocessor._build_policy_observation("")[STATE_KEY].shape),
            (1, 1, 2),
        )

    def test_other_policies_keep_single_observation(self):
        preprocessor = Preprocessor([1.0, 2.0], expected=2)
        self.assertEqual(
            tuple(preprocessor._build_policy_observation("")[STATE_KEY].shape),
            (1, 2),
        )

    def test_diffusion_propagates_sensor_failure(self):
        preprocessor = Preprocessor([1.0, 2.0], expected=2)
        preprocessor._policy.config.type = "diffusion"
        preprocessor._policy.config.n_obs_steps = 2
        preprocessor._build_observation = lambda task: {"success": False, "message": "Missing camera"}
        self.assertFalse(preprocessor._build_policy_observation("")["success"])

    def test_pads_short_state_to_policy_shape(self):
        preprocessor = Preprocessor([1.0, 2.0], expected=4)

        batch = preprocessor._build_observation("task")

        np.testing.assert_allclose(
            batch[STATE_KEY].numpy(),
            np.asarray([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32),
        )

    def test_truncates_long_state_to_policy_shape(self):
        preprocessor = Preprocessor([1.0, 2.0, 3.0, 4.0], expected=2)

        batch = preprocessor._build_observation("task")

        np.testing.assert_allclose(
            batch[STATE_KEY].numpy(),
            np.asarray([[1.0, 2.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
