"""Closed-loop dataset observations use IK state rather than recorded future joints."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from robot_client.robot_client import robot_schema

from lerobot_engine.dataset_simulation import DatasetSimulation, configured_initial_state
from lerobot_engine.preprocessing import PreprocessingMixin
from lerobot_engine.relative_eef import RelativeEEFAdapter
from lerobot_engine._vendor.dream.relative_eef_pose import RelativeEEFTransform


@pytest.fixture
def simulation(tmp_path, request):
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
        options = {"use_initial_frame_state": request.param} if hasattr(request, "param") else {}
        robot = DatasetSimulation("ffw_sg2_rev1", tmp_path, 1, 1, 2, **options)
    reader.assert_called_once_with(tmp_path.name, root=tmp_path, episodes=[1], video_backend="pyav")
    adapter = RelativeEEFAdapter.__new__(RelativeEEFAdapter)
    adapter.ik_backend = "cyclo_control"
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


@pytest.mark.parametrize("simulation", [False], indirect=True)
def test_configured_pose_replaces_all_joint_history_but_keeps_dataset_images(simulation):
    robot, engine, samples = simulation
    assert robot.initial_state_source == "configured_initial_state"
    assert robot.use_initial_frame_state is False
    assert [frame for frame, _ in robot.history] == [0, 1]
    home = robot.current[1].copy()
    assert home[robot.names.index("lift_joint")] == pytest.approx(-.001771)
    assert home[robot.names.index("head_joint1")] == pytest.approx(.7773294300)
    assert home[robot.names.index("gripper_l_joint1")] == pytest.approx(.2148800281)
    assert home[robot.names.index("linear_x")] == pytest.approx(.0000211963, abs=1e-10)
    np.testing.assert_array_equal(robot.history[0][1], robot.history[1][1])
    assert robot.history[0][1] is not robot.history[1][1]
    # Verify the geometric constraints independently through the policy FK model.
    poses = engine._relative_eef.transform.poses(home, "observation.state")[0]
    np.testing.assert_allclose(poses[:, 0, 3], .48563202918324094, atol=1e-7)
    np.testing.assert_allclose(poses[:, 1, 3], [.17355220030191562, -.19497015368571618], atol=1e-7)
    np.testing.assert_allclose(poses[:, 2, 3], 1.0845264878214818, atol=1e-7)
    assert np.linalg.norm(poses[0, :3, 3] - poses[1, :3, 3]) == pytest.approx(.3685223539876318, abs=1e-7)
    model_robot = engine._relative_eef.transform._robot
    for side in ("left", "right"):
        camera = model_robot.model.getFrameId(f"camera_{side}_link")
        np.testing.assert_allclose(model_robot.data.oMf[camera].rotation[:, 0], [0, 0, -1], atol=1e-6)
    batch = engine._build_policy_observation("")
    for key in engine._cameras.values():
        torch.testing.assert_close(batch[key][0, :, 0, 0, 0], torch.tensor([0., .1]), atol=1/255, rtol=0)
    # First relative window has one identical home anchor across all history.
    np.testing.assert_allclose(batch["observation.state"][0, :, :3], 0, atol=1e-7)
    np.testing.assert_allclose(batch["observation.state"][0, :, 10:13], 0, atol=1e-7)
    commands = np.full((1, 17), .25)
    robot.advance(commands, engine)
    np.testing.assert_array_equal(robot.history[0][1], home)
    np.testing.assert_allclose(robot.current[1][:16], commands[0, :16])
    assert robot.current[1][robot.names.index("head_joint1")] == home[robot.names.index("head_joint1")]
    assert not np.array_equal(robot.current[1], samples[2]["observation.state"].numpy())


def test_configured_initial_state_maps_joint_and_base_means_by_name():
    names = ["angular_z", "gripper_r_joint1", "head_joint1", "linear_y"]
    actual = configured_initial_state("ffw_sg2_rev1", names)
    np.testing.assert_allclose(actual, [-.0000277439, .1027306928, .7773294300,
                                       -.0000001787], rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("invalid", ["false", 0, None])
def test_initial_frame_state_flag_must_be_boolean(tmp_path, invalid):
    with pytest.raises(ValueError, match="must be a boolean"):
        DatasetSimulation("ffw_sg2_rev1", tmp_path, 0, 0, 2, use_initial_frame_state=invalid)


def test_ik_actions_feed_next_window_and_images_advance_without_transport(simulation):
    robot, engine, samples = simulation
    assert robot.use_initial_frame_state is True
    assert robot.initial_state_source == "dataset_frame"
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


@pytest.mark.parametrize("backend", ["nodeik", "dls"])
def test_quintic_subframes_preserve_endpoint_feedback_and_dataset_images(simulation, backend):
    robot, engine, samples = simulation
    engine._relative_eef.ik_backend = backend
    engine._relative_eef.kinematics = SimpleNamespace(set_images=lambda *args: None)
    commands = np.full((2, 17), 0.25)
    commands[1] = 0.5
    initial = robot.current[1].copy()
    frames = robot.advance(commands, engine)
    assert len(frames) == 8  # 10, 20, 30, 33.333 ms per 30 Hz waypoint
    assert [frame["frame"] for frame in frames] == [1, 1, 1, 2, 2, 2, 2, 3]
    assert sum(frame["dt_s"] for frame in frames) == pytest.approx(2 / 30)
    state = np.array([frame["joint_state"]["position"] for frame in frames])
    # Independent known smoothstep value at 30% of the first interval.
    weight = 10 * 0.3**3 - 15 * 0.3**4 + 6 * 0.3**5
    expected = initial[:16] + weight * (commands[0, :16] - initial[:16])
    np.testing.assert_allclose(state[0, :16], expected, atol=1e-7)
    np.testing.assert_allclose(state[3, :16], commands[0, :16])
    np.testing.assert_allclose(state[7, :16], commands[1, :16])
    lift = robot.names.index("lift_joint")
    np.testing.assert_allclose(state[[3, 7], lift], commands[:, 16])
    assert [frame for frame, _ in robot.history] == [2, 3]
    np.testing.assert_allclose(robot.current[1], state[-1])
    assert frames[0]["images"] == frames[1]["images"] == frames[2]["images"]
    assert frames[2]["images"] != frames[3]["images"]
    assert set(robot._samples) == {2, 3}
    # Next policy window uses endpoint states, never interpolation subframes.
    batch = engine._build_policy_observation("task")
    endpoints = np.stack([state[3], state[7]])
    expected = engine._relative_eef.transform({"observation.state": endpoints,
                                              "action": endpoints})["observation.state"]
    torch.testing.assert_close(batch["observation.state"][0], expected)
    key = next(iter(engine._cameras.values()))
    torch.testing.assert_close(batch[key][0, :, 0, 0, 0], torch.tensor([0.2, 0.3]), atol=1/255, rtol=0)


@pytest.mark.parametrize("backend", ["nodeik", "dls"])
def test_quintic_subframes_stop_at_episode_end(simulation, backend):
    robot, engine, _ = simulation
    engine._relative_eef.ik_backend = backend
    frames = robot.advance(np.zeros((8, 17)), engine)
    assert len(frames) == 16
    assert robot.frame == 5
    assert frames[-1]["frame"] == 5
    assert not robot.advance(np.zeros((8, 17)), engine)


@pytest.mark.parametrize("simulation", [False], indirect=True)
@pytest.mark.parametrize("control_hz, expected_times", [(80, [1/80, 2/80, 1/30]), (10, [1/30])])
def test_dls_interpolates_configured_pose_at_selected_rate(simulation, control_hz, expected_times):
    robot, engine, _ = simulation
    engine._relative_eef.ik_backend = "dls"
    robot.control_hz = control_hz
    start = robot.current[1].copy()
    frames = robot.advance(np.full((1, 17), .25), engine)
    positions = np.array([frame["joint_state"]["position"] for frame in frames])
    times = np.cumsum([frame["dt_s"] for frame in frames])
    np.testing.assert_allclose(times, expected_times)
    assert all(frame["dt_s"] > 0 for frame in frames)
    target = robot.current[1]
    u = times * robot.fps
    weights = 10 * u**3 - 15 * u**4 + 6 * u**5
    np.testing.assert_allclose(positions, start + weights[:, None] * (target - start), atol=1e-7)
    for name in ("head_joint1", "head_joint2", "linear_x", "linear_y", "angular_z"):
        index = robot.names.index(name)
        np.testing.assert_array_equal(positions[:, index], np.repeat(start[index], len(frames)))
    np.testing.assert_array_equal(positions[-1], target)
    np.testing.assert_array_equal(robot.history[0][1], start)
    np.testing.assert_array_equal(robot.history[-1][1], target)


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
