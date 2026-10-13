"""ChunkRL preserves selected endpoints and resets after discarded requests."""
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
robot_stub = types.ModuleType("robot_client")
robot_stub.RobotClient = object
sys.modules.setdefault("robot_client", robot_stub)
from main_runtime.control_loop import ControlLoop, normalize_action_request_mode
from engine_process.protocol import EngineCommandRequest, request_from_message, request_to_message_kwargs
from main_runtime.inference_requester import InferenceRequester
from action_chunk_processing import ActionChunkProcessor


def test_selected_waypoint_reaches_endpoint_without_alignment_or_blending():
    processor = ActionChunkProcessor(inference_hz=30, control_hz=100, interpolation_method='quintic')
    processor.push_waypoint(np.array([0., 1.]))
    while processor.pop_action() is not None:
        pass
    count = processor.push_waypoint(np.array([1., 2.]))
    outputs = np.asarray([processor.pop_action() for _ in range(count)])
    assert count == 3
    np.testing.assert_array_equal(outputs[-1], [1., 2.])
    assert (outputs >= [0., 1.]).all() and (outputs <= [1., 2.]).all()
    assert processor.buffer_size == 0


def test_chunkrl_refill_waits_for_selected_waypoint_and_passes_reset_flag():
    calls = []
    def get_action(_, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=True, chunk_size=1, action_dim=2, action_list=[.2, .4])
    loop = ControlLoop(SimpleNamespace(get_action=get_action), action_request_mode='chunkrl')
    loop._processor = ActionChunkProcessor(inference_hz=30, control_hz=100)
    loop._running = True
    loop._action_request_mode = 'chunkrl'
    loop._request_and_buffer('', loop._generation, 'chunkrl')
    assert calls[-1] == {'reset_policy_state': True}
    assert not loop._should_request_actions(loop._processor)
    while loop._processor.pop_action() is not None:
        pass
    assert loop._should_request_actions(loop._processor)
    loop._request_and_buffer('', loop._generation, 'chunkrl')
    assert calls[-1] == {'reset_policy_state': False}
    loop._processor.clear()
    loop._requester.get_action = lambda *args, **kwargs: SimpleNamespace(success=False, message='timeout')
    loop._request_and_buffer('', loop._generation, 'chunkrl')
    assert loop._chunkrl_reset_pending
    loop._processor = None


def test_chunkrl_wire_fields_survive_roundtrip_and_load_forwarding():
    request = EngineCommandRequest(0, action_request_mode='chunkrl', chunkrl_checkpoint='/model/actor.pt', reset_policy_state=True)
    decoded = request_from_message(SimpleNamespace(**request_to_message_kwargs(request)))
    assert decoded == request
    calls = []
    class Client:
        def call(self, req, **kwargs):
            calls.append(req)
            return SimpleNamespace(success=True, seq_id=req.seq_id)
    requester = InferenceRequester(Client())
    assert requester.load_policy(request).success
    assert calls[-1].action_request_mode == 'chunkrl'
    assert calls[-1].chunkrl_checkpoint == '/model/actor.pt'
    assert requester.get_action('', reset_policy_state=True).success
    assert calls[-1].reset_policy_state
    assert normalize_action_request_mode(' ChunkRL ') == 'chunkrl'


def test_waypoint_ticks_preserve_30fps_over_a_full_second():
    processor = ActionChunkProcessor(inference_hz=30, control_hz=100)
    counts = [processor.push_waypoint(np.array([float(i)])) for i in range(30)]
    assert sum(counts) == 100
    assert set(counts) == {3, 4}
    processor.clear()
    assert processor.push_waypoint(np.array([0.])) == 3
