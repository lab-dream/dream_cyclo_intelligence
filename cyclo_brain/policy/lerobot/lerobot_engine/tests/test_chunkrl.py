"""Per-arm selection, coordinate conversion, and failure recovery without transport."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot_engine.chunkrl import BimanualChunkRL, configure_chunkrl
from lerobot_engine.engine import LeRobotEngine
from lerobot_engine._vendor.chunk_rl.decider_wrapper import ChunkDeciderWrapper
from lerobot_engine._vendor.chunk_rl.model import ChunkUpdateConfig, ChunkUpdateDecider
from lerobot_engine._vendor.umi.pose_util import mat_to_rot6d, pose_to_mat, rot6d_to_mat


class DecisionActor(torch.nn.Module):
    def __init__(self, updates, index):
        super().__init__()
        self.config = ChunkUpdateConfig(L=3, H=2, d_a=10)
        self.updates = iter(updates)
        self.index = index
        self.inputs = []

    def forward(self, new_chunk, ctx, mask):
        self.inputs.append((new_chunk.clone(), ctx.clone(), mask.clone()))
        logits = torch.full((1, 3), -100.0)
        logits[0, self.index] = 100
        return {"update_logit": torch.tensor([[100.0 if next(self.updates) else -100.0]]),
                "index_logits": logits}


def manager(updates_left=(True, False), updates_right=(True, False)):
    result = BimanualChunkRL.__new__(BimanualChunkRL)
    result.arms = [ChunkDeciderWrapper(DecisionActor(updates_left, 1), relative_action=True, device='cpu'),
                   ChunkDeciderWrapper(DecisionActor(updates_right, 0), relative_action=True, device='cpu')]
    result.length = 3
    result.last_decisions = []
    return result


def proposal():
    result = np.zeros((3, 20), dtype=np.float32)
    for side in range(2):
        result[:, side*10:side*10+3] = np.array([[.01, .02, .03], [.02, .04, .06], [.03, .06, .09]])
        result[:, side*10+3:side*10+9] = mat_to_rot6d(Rotation.from_euler('xyz', [20, -30, 40], degrees=True).as_matrix())
        result[:, side*10+9] = [-.4, .2, .8]
    return result


def observation(offset=0):
    return {f'robot{side}_{key}': np.asarray([values])
            for side in range(2) for key, values in [
                ('eef_pos', [offset, side*.3, .5]),
                ('eef_rot_axis_angle', [.2, -.3, .4])]}


def absolute_output(selected, obs, side):
    local = np.eye(4)
    local[:3, :3] = rot6d_to_mat(selected[0, side*10+3:side*10+9])
    local[:3, 3] = selected[0, side*10:side*10+3]
    anchor = pose_to_mat(np.concatenate([obs[f'robot{side}_eef_pos'][-1], obs[f'robot{side}_eef_rot_axis_angle'][-1]]))
    return anchor @ local


def test_independent_indices_and_non_identity_row_column_rotation_roundtrip():
    deciders = manager()
    chunk = proposal()
    result = deciders.select(chunk, observation())
    np.testing.assert_allclose(result[0, :10], chunk[1, :10], atol=2e-6)
    np.testing.assert_allclose(result[0, 10:], chunk[0, 10:], atol=2e-6)
    assert [d['index'] for d in deciders.last_decisions] == [1, 0]
    assert deciders.arms[0].past_actions is not deciders.arms[1].past_actions
    actor_input = deciders.arms[0].actor.inputs[0][0][0, 0, 3:9].numpy()
    expected = rot6d_to_mat(chunk[0, 3:9])[:, :2].T.reshape(6)
    np.testing.assert_allclose(actor_input, expected, atol=1e-6)


def test_kept_chunks_remain_at_original_world_targets_after_anchor_moves():
    deciders = manager()
    chunk = proposal()
    old_obs, new_obs = observation(), observation(.08)
    deciders.select(chunk, old_obs)
    next_selected = deciders.select(chunk, new_obs)
    assert [d['index'] for d in deciders.last_decisions] == [2, 1]
    assert not any(d['switched'] for d in deciders.last_decisions)
    for side, index in [(0, 2), (1, 1)]:
        np.testing.assert_allclose(absolute_output(next_selected, new_obs, side),
                                  absolute_output(chunk[index:index+1], old_obs, side), atol=2e-6)


def test_right_arm_failure_rolls_back_left_arm_history():
    deciders = manager()
    chunk = proposal()
    chunk[:, 13:19] = 0
    with pytest.raises(ValueError, match='degenerate'):
        deciders.select(chunk, observation())
    assert all(a.current_chunk is None and len(a.past_actions) == 0 for a in deciders.arms)


def test_exhaustion_forces_switch_and_reset_clears_both_histories():
    deciders = manager((False, False, False, False), (False, False, False, False))
    for _ in range(4):
        deciders.select(proposal(), observation())
    assert all(d['switched'] for d in deciders.last_decisions)
    deciders.reset()
    assert all(a.current_chunk is None and a.chunk_step == 0 and not a.past_actions for a in deciders.arms)


def test_load_restores_diffusion_window_and_rejects_joint_checkpoint(tmp_path):
    config = ChunkUpdateConfig(L=3, H=2, d_a=10)
    checkpoint = tmp_path/'actor.pt'
    torch.save({'actor_config': vars(config), 'actor': ChunkUpdateDecider(config).state_dict()}, checkpoint)
    engine = SimpleNamespace(_policy=SimpleNamespace(config=SimpleNamespace(type='diffusion', horizon=16, n_obs_steps=2, n_action_steps=8), reset=lambda: None),
                             _relative_eef=object(), _device='cpu')
    configure_chunkrl(engine, SimpleNamespace(action_request_mode='chunkrl', chunkrl_checkpoint=str(checkpoint)))
    assert engine._policy.config.n_action_steps == 3
    assert engine._chunkrl.arms[0].actor is engine._chunkrl.arms[1].actor
    configure_chunkrl(engine, SimpleNamespace(action_request_mode='sync'))
    assert engine._policy.config.n_action_steps == 8 and engine._chunkrl is None
    engine._relative_eef = None
    with pytest.raises(ValueError, match="state='relative'"):
        configure_chunkrl(engine, SimpleNamespace(action_request_mode='chunkrl'))


def test_failed_ik_does_not_commit_decider_state():
    engine = LeRobotEngine.__new__(LeRobotEngine)
    engine._chunkrl = manager()
    engine._policy = object()
    engine._robot = object()
    engine._preprocessor = engine._postprocessor = lambda x: x
    engine._build_policy_observation = lambda _: {}
    engine._predict_chunk = lambda _: torch.from_numpy(proposal()).unsqueeze(0)
    def fail(_):
        raise ValueError('IK failure')
    engine._relative_eef = SimpleNamespace(env_obs=observation(), decode_action=fail)
    result = engine.get_action_chunk(SimpleNamespace(task_instruction=''))
    assert not result['success'] and 'IK failure' in result['message']
    assert all(a.current_chunk is None and not a.past_actions for a in engine._chunkrl.arms)
