"""Telemetry reports accepted requests and failures without rejected action poses."""
import json
import time
import sys
import types
from types import SimpleNamespace

import numpy as np

robot_stub = types.ModuleType('robot_client')
robot_stub.RobotClient = object
sys.modules.setdefault('robot_client', robot_stub)

from inference_telemetry import InferenceTrace
from engine_process.protocol import EngineCommandRequest, response_from_message, response_to_message_kwargs
from engine_process.worker import EngineWorker
from main_runtime.control_loop import ControlLoop
from action_chunk_processing import ActionChunkProcessor


def test_stages_and_engine_response_survive_wire_roundtrip():
    class Engine:
        def get_action_chunk(self, request):
            trace = InferenceTrace()
            with trace.stage('observation'):
                time.sleep(.001)
            self._last_inference_telemetry = trace.finish(eef={'anchor_poses': []})
            return dict(success=True, chunk_size=1, action_dim=2, action_chunk=np.array([.2, .4]))
    response = EngineWorker(Engine()).handle(EngineCommandRequest(command=1, seq_id=5))
    restored = response_from_message(SimpleNamespace(**response_to_message_kwargs(response)))
    data = json.loads(restored.telemetry)
    assert restored.success and restored.seq_id == 5
    assert data['timings_ms']['observation'] > 0
    assert data['timings_ms']['engine_request'] >= data['timings_ms']['engine']


def test_main_reports_transport_and_buffer_time_and_clears_rejected_action_pose():
    payload = dict(timings_ms=dict(policy=2.0), eef={'observation_poses': [[0]*14], 'action_poses': [[1]*14]})
    response = SimpleNamespace(success=True, telemetry=json.dumps(payload), chunk_size=1,
                               action_dim=2, action_list=[.2, .4])
    reports = []
    loop = ControlLoop(SimpleNamespace(get_action=lambda *a, **k: response),
                       telemetry_callback=reports.append, action_request_mode='chunkrl')
    loop._processor = ActionChunkProcessor(inference_hz=30, control_hz=100)
    loop._running = True
    loop._request_and_buffer('', loop._generation, 'chunkrl')
    assert reports[-1]['success'] and reports[-1]['eef'] == payload['eef']
    timing = reports[-1]['timings_ms']
    assert timing['buffer'] >= 0
    assert timing['total'] == timing['request'] + timing['buffer']
    response.success = False
    response.message = 'IK target rejected'
    loop._request_and_buffer('', loop._generation, 'chunkrl')
    assert not reports[-1]['success'] and not reports[-1]['eef']['action_poses']
    assert reports[-1]['eef']['observation_poses'] == payload['eef']['observation_poses']
    assert reports[-1]['message'] == response.message
    loop._processor = None


def test_stale_response_after_unload_does_not_restore_telemetry():
    reports = []
    loop = ControlLoop(None, telemetry_callback=reports.append)
    def stale(*args, **kwargs):
        loop._generation += 1
        return SimpleNamespace(success=False, message='old request')
    loop._requester = SimpleNamespace(get_action=stale)
    loop._request_and_buffer('', loop._generation, 'chunkrl')
    assert reports == []
