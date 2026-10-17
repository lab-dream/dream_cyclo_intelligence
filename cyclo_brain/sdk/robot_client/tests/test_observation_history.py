"""Receive-time alignment, bounded history, and coherent callback snapshots."""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

sdk = types.ModuleType('zenoh_ros2_sdk')
sdk.ROS2Publisher = sdk.ROS2Subscriber = object
sdk.get_message_class = lambda name: SimpleNamespace
sys.modules.setdefault('zenoh_ros2_sdk', sdk)
p = Path(__file__).resolve().parents[1] / 'robot_client/robot_client.py'
sys.path.insert(0, str(p.parents[4] / 'shared/shared/robot_configs'))
spec = importlib.util.spec_from_file_location('history_robot_client', p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RobotClient = module.RobotClient


@pytest.fixture
def client():
    with patch.object(RobotClient, '_init_subscriptions'):
        robot = RobotClient('ffw_sg2_rev1')
    robot._joint_children = {'follower_upper_body': [name for name, cfg in robot._config['joint_groups'].items()
                                                  if cfg.get('parent') == 'follower_upper_body']}
    robot.enable_observation_history(['cam_right_wrist'])
    yield robot
    robot.close()


def image(robot, t, value, camera='cam_right_wrist'):
    bgr = np.full((4, 6, 3), [value, value + 1, value + 2], dtype=np.uint8)
    ok, encoded = cv2.imencode('.png', bgr)
    assert ok
    with patch.object(module.time, 'monotonic', return_value=t):
        robot._update_image(camera, SimpleNamespace(data=encoded.tobytes()))


def joints(robot, t, value):
    names = robot._config['joint_groups']['follower_upper_body']['joint_names']
    msg = SimpleNamespace(name=list(reversed(names)), position=[value] * len(names))
    with patch.object(module.time, 'monotonic', return_value=t):
        robot._update_joint('follower_upper_body', msg)


def window(robot, now, steps=2, interval=1/30, **kwargs):
    with patch.object(module.time, 'monotonic', return_value=now):
        return robot.get_observation_window(steps, interval, **kwargs)


def test_jittered_frames_align_states_and_keep_snapshots_after_eviction(client):
    for t, value in [(1., 10), (1.033, 20), (1.0665, 30)]:
        image(client, t, value)
        joints(client, t - .002, value)
    snapshots, meta = window(client, 1.07)
    assert meta['stream_timestamps_s']['image:cam_right_wrist'] == [1.033, 1.0665]
    assert meta['max_alignment_error_ms'] < 3.
    assert meta['latest_age_ms'] == pytest.approx(3.5)
    assert [s['images']['cam_right_wrist'][0, 0].tolist() for s in snapshots] == [[22, 21, 20], [32, 31, 30]]
    assert snapshots[-1]['joints']['follower_arm_right'][0] == 30
    for n in range(300):
        image(client, 2 + n / 1000, 40)
        joints(client, 2 + n / 1000, 40)
    assert len(client._observation_history[('image', 'cam_right_wrist')]) == 256
    assert snapshots[-1]['joints']['follower_arm_right'][0] == 30
    assert snapshots[-1]['images']['cam_right_wrist'][0, 0, 0] == 32
    assert ('image', 'cam_left_wrist') not in client._observation_history


def test_history_uses_training_spacing_even_when_requests_are_slow(client):
    for n in range(12):
        t = 1 + n / 30
        image(client, t, n)
        joints(client, t, n)
        if n in (4, 11):
            snapshots, meta = window(client, t + .001)
            assert [s['images']['cam_right_wrist'][0, 0, 2] for s in snapshots] == [n-1, n]
            assert np.diff(meta['stream_timestamps_s']['image:cam_right_wrist'])[0] == pytest.approx(1/30)
    snapshots, _ = window(client, 1 + 11/30 + .001, steps=5)
    assert [s['images']['cam_right_wrist'][0, 0, 2] for s in snapshots] == list(range(7, 12))


def test_missing_duplicate_stale_and_dropped_frames_are_rejected_without_wait(client):
    with pytest.raises(ValueError, match='warming up'):
        window(client, 1.)
    image(client, 1., 10)
    joints(client, 1., 10)
    with pytest.raises(ValueError, match='gap|distinct'):
        window(client, 1.001)
    image(client, 1 + 1/30, 20)
    joints(client, 1 + 1/30, 20)
    with pytest.raises(ValueError, match='Stale'):
        window(client, 2.)
    image(client, 1.2, 30)
    joints(client, 1.2, 30)
    with pytest.raises(ValueError, match='gap|distinct'):
        window(client, 1.201)


def test_joint_samples_after_camera_anchor_are_not_used(client):
    for t in (1., 1 + 1/30):
        image(client, t, 10)
        joints(client, t - .001, 10)
    joints(client, 1.04, 99)
    snapshots, meta = window(client, 1.041)
    assert snapshots[-1]['joints']['follower_arm_right'][0] == 10
    assert max(meta['stream_timestamps_s']['joint:follower_upper_body']) <= meta['target_timestamps_s'][-1]


def test_multiple_cameras_are_aligned_to_one_window(client):
    client.enable_observation_history(['cam_left_wrist', 'cam_right_wrist'])
    for t, value in [(1., 10), (1 + 1/30, 20)]:
        image(client, t - .006, value, 'cam_left_wrist')
        image(client, t, value, 'cam_right_wrist')
        joints(client, t, value)
    snapshots, meta = window(client, 1.04)
    assert meta['max_alignment_error_ms'] == pytest.approx(6.)
    assert snapshots[-1]['images']['cam_left_wrist'][0, 0, 2] == 20
    assert snapshots[-1]['images']['cam_right_wrist'][0, 0, 2] == 20


def test_sensor_history_is_optional_and_aligned_when_requested(client):
    for t in (1., 1 + 1/30):
        image(client, t, 10)
        joints(client, t, 10)
    window(client, 1.04)  # Relative EEF does not require odometry.
    with pytest.raises(ValueError, match='missing sensor:odom'):
        window(client, 1.04, sensor_names=('odom',))
    with client._lock:
        for t in (1., 1 + 1/30):
            client._append_observation('sensor', 'odom', t, {'linear_velocity': np.array([t, 0, 0])})
    snapshots, _ = window(client, 1.04, sensor_names=('odom',))
    assert snapshots[-1]['sensors']['odom']['linear_velocity'][0] == pytest.approx(1 + 1/30)


def test_late_callback_does_not_replace_history_and_reload_clears_it(client):
    image(client, 1., 10)
    image(client, .9, 99)
    assert client._observation_history[('image', 'cam_right_wrist')][-1][0] == 1.
    client.enable_observation_history(['cam_left_wrist'])
    assert ('image', 'cam_right_wrist') not in client._observation_history
    assert not client._observation_history[('image', 'cam_left_wrist')]
