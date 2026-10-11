"""Closed-loop dataset observations use IK state rather than recorded future joints."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from robot_client.robot_client import robot_schema

from lerobot_engine.dataset_simulation import DatasetSimulation
from lerobot_engine.preprocessing import PreprocessingMixin
from lerobot_engine.relative_eef import RelativeEEFAdapter
from lerobot_engine._vendor.dream.relative_eef_pose import RelativeEEFTransform


@pytest.fixture
def simulation(tmp_path):
    section = robot_schema.load_robot_section("ffw_sg2_rev1")
    names = section["observation"]["state"]["upper_body"]["joint_names"] + ["linear_x", "linear_y", "angular_z"]
    features = {key: {"shape": [22], "names": names} for key in ("observation.state", "action")}
    stats = {key: {"min": [-1.0] * 22, "max": [1.0] * 22} for key in features}
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"robot_type": "ffw_sg2_rev1", "total_episodes": 2,
                                              "fps": 30, "features": features}))
    (meta / "stats.json").write_text(json.dumps(stats))
    samples = [{"observation.state": torch.full((22,), index / 100),
                **{f"observation.images.rgb.{name}": torch.full((3, 12, 8), index / 10)
                   for name in robot_schema.get_image_topics(section)}} for index in range(6)]
    with patch("lerobot_engine.dataset_simulation.LeRobotDataset", return_value=samples) as reader, \
         patch("robot_client.RobotClient.__init__", side_effect=AssertionError("No robot transport")):
        robot = DatasetSimulation("ffw_sg2_rev1", tmp_path, 1, 1, 2)
    reader.assert_called_once_with(tmp_path.name, root=tmp_path, episodes=[1], video_backend="pyav")
    adapter = RelativeEEFAdapter.__new__(RelativeEEFAdapter)
    adapter.robot = robot
    adapter.names = names
    adapter.transform = RelativeEEFTransform(features, stats)
    adapter.env_obs = None
    adapter.measured = {}
    adapter.shape_meta = {"obs": {
        f"robot{side}_{key}": {"shape": [width], "type": "low_dim"}
        for side in range(2) for key, width in (("eef_pos", 3), ("eef_rot_axis_angle", 6), ("gripper_width", 1))}}
    engine = PreprocessingMixin()
    engine._robot = robot
    engine._relative_eef = adapter
    engine._dataset_simulation = robot
    engine._device = torch.device("cpu")
    engine._cameras = {name: f"observation.images.rgb.{name}" for name in robot.camera_names}
    engine._image_resize = {key: (8, 8) for key in engine._cameras.values()}
    engine._action_keys = ["arm_left", "arm_right", "lift"]
    engine._policy = SimpleNamespace(config=SimpleNamespace(type="diffusion", n_obs_steps=2,
                                                           input_features={"observation.state": None, **dict.fromkeys(engine._cameras.values())}))
    engine._fail = lambda message: {"success": False, "message": message}
    return robot, engine, samples


def test_ik_actions_feed_next_window_and_images_advance_without_transport(simulation):
    robot, engine, samples = simulation
    first = engine._build_policy_observation("task")
    assert first["observation.state"].shape == (1, 2, 20)
    expected_first = engine._relative_eef.transform
    recorded = np.stack([samples[i]["observation.state"].numpy() for i in (0, 1)])
    torch.testing.assert_close(first["observation.state"][0], expected_first({"observation.state": recorded, "action": recorded})["observation.state"])
    commands = np.full((2, 17), 0.25)
    commands[1] = 0.5
    frames = robot.advance(commands, engine)
    assert [frame["frame"] for frame in frames] == [2, 3]
    assert all(src.startswith("data:image/jpeg;base64,") for src in frames[-1]["images"].values())
    for offset, (_, state) in enumerate(robot.history):
        np.testing.assert_allclose(state[:16], commands[offset, :16])
        assert state[robot.names.index("lift_joint")] == commands[offset, 16]
        # Head and odometry remain from the initial observation.
        assert state[robot.names.index("head_joint1")] == samples[1]["observation.state"][robot.names.index("head_joint1")]
    second = engine._build_policy_observation("task")
    predicted = np.stack([state for _, state in robot.history])
    expected = engine._relative_eef.transform({"observation.state": predicted, "action": predicted})["observation.state"]
    torch.testing.assert_close(second["observation.state"][0], expected)
    key = next(iter(engine._cameras.values()))
    torch.testing.assert_close(second[key][0, :, 0, 0, 0], torch.tensor([0.2, 0.3]), atol=1/255, rtol=0)
    assert set(robot._samples) == {2, 3}


def test_episode_end_does_not_read_or_apply_past_final_frame(simulation):
    robot, engine, _ = simulation
    frames = robot.advance(np.zeros((8, 17)), engine)
    assert [frame["frame"] for frame in frames] == [2, 3, 4, 5]
    assert robot.frame == 5
    assert not robot.advance(np.zeros((8, 17)), engine)


@pytest.mark.parametrize("rotation_deg", [0, 90, 180, 270])
def test_recorded_image_orientation_reaches_policy_unchanged(simulation, rotation_deg):
    robot, engine, samples = simulation
    # An asymmetric image detects flips that uniform-color fixtures cannot reveal.
    image = torch.arange(3 * 12 * 8, dtype=torch.float32).reshape(3, 12, 8) % 256 / 255
    for name, cfg in robot._config["cameras"].items():
        cfg["rotation_deg"] = rotation_deg
        key = f"observation.images.rgb.{name}"
        engine._image_resize[key] = (8, 12)
        for sample in samples:
            sample[key] = image
    batch = engine._build_policy_observation("")
    for key in engine._cameras.values():
        torch.testing.assert_close(batch[key][0], image.expand(2, -1, -1, -1), atol=1e-7, rtol=0)


def test_simulation_keeps_training_gripper_bounds_from_checkpoint(simulation, tmp_path):
    robot, _, _ = simulation
    model = tmp_path / "checkpoint"
    meta = model / "dataset_meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_bytes((robot.dataset_root / "meta/info.json").read_bytes())
    stats = {key: {"min": [0.0] * 22, "max": [1.0] * 22} for key in ("observation.state", "action")}
    (meta / "stats.json").write_text(json.dumps(stats))
    # Metadata selection is independent of solving IK; native solve is covered separately.
    with patch.dict(sys.modules, {"_cyclo_ik": SimpleNamespace(CycloIK=lambda *args: object())}):
        adapter = RelativeEEFAdapter(str(model), robot, "ffw_sg2_rev1")
    try:
        lower, upper = adapter.transform.gripper_bounds["observation.state"]
        np.testing.assert_allclose(lower, [0, 0])
        np.testing.assert_allclose(upper, [1, 1])
    finally:
        adapter.close()


def test_missing_checkpoint_metadata_requires_original_training_dataset(simulation, tmp_path):
    robot, _, _ = simulation
    model = tmp_path / "portable_checkpoint"
    model.mkdir()
    train = model / "train_config.json"
    train.write_text(json.dumps({"dataset": {"repo_id": robot.dataset_root.name}}))
    with patch.dict(sys.modules, {"_cyclo_ik": SimpleNamespace(CycloIK=lambda *args: object())}):
        adapter = RelativeEEFAdapter(str(model), robot, "ffw_sg2_rev1")
        adapter.close()
        train.write_text(json.dumps({"dataset": {"repo_id": "another_dataset"}}))
        with pytest.raises(ValueError, match="Bundle original training dataset_meta"):
            RelativeEEFAdapter(str(model), robot, "ffw_sg2_rev1")
