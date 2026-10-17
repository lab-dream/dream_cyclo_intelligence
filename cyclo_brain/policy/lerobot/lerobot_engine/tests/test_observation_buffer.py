"""Live windows consume the SDK snapshot, never reread newer sensor caches."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot_engine.engine import LeRobotEngine


@pytest.fixture
def engine():
    engine = LeRobotEngine()
    engine._device = torch.device('cpu')
    engine._state_modalities = ['arm']
    engine._cameras = {'wrist': 'observation.images.wrist'}
    engine._policy = SimpleNamespace(config=SimpleNamespace(type='diffusion',n_obs_steps=2,
        input_features={'observation.state':SimpleNamespace(shape=[2]),'observation.images.wrist':None}))
    snapshots=[dict(images={'wrist':np.full((4,6,3),value,np.uint8)},
                    joints={'follower_arm':np.array([value,value+1],np.float32)},sensors={}) for value in (10,20)]
    engine._robot = SimpleNamespace(_config={'cameras':{'wrist':{}}},
        get_observation_window=Mock(return_value=(snapshots,{'latest_age_ms':2.})),
        get_images=Mock(side_effect=AssertionError('Do not reread live images')),
        get_joint_positions=Mock(side_effect=AssertionError('Do not reread live joints')))
    yield engine
    engine._robot = None


def test_input_window_uses_fixed_history_samples(engine):
    batch=engine._build_policy_observation('task')
    assert batch['observation.images.wrist'].shape==(1,2,3,4,6)
    torch.testing.assert_close(batch['observation.state'],torch.tensor([[[10.,11.],[20.,21.]]]))
    torch.testing.assert_close(batch['observation.images.wrist'][0,:,0,0,0],torch.tensor([10.,20.])/255)
    assert engine._observation_window=={'latest_age_ms':2.}
    engine._robot.get_observation_window.assert_called_once_with(2,1/30,sensor_names=())


def test_unavailable_history_fails_before_encoding_or_prediction(engine):
    engine._robot.get_observation_window.side_effect=ValueError('Stale observation stream')
    engine._observation_window={'old':True}
    result=engine._build_policy_observation('')
    assert not result['success'] and 'Stale' in result['message']
    assert engine._observation_window is None


def test_act_still_uses_the_existing_latest_observation_path(engine):
    engine._policy.config.type='act'
    engine._build_observation=Mock(return_value={'existing':'act'})
    assert engine._build_policy_observation('task')=={'existing':'act'}
    engine._build_observation.assert_called_once_with('task')
    engine._robot.get_observation_window.assert_not_called()
