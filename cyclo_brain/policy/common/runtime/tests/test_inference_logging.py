"""Recorded events distinguish model latency, command starvation and robot output."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from inference_logging import InferenceEventPublisher, log_values, set_engine_logger
from inference_telemetry import InferenceTrace
from engine_process.worker import EngineWorker
from main_runtime.inference_requester import InferenceRequester
from main_runtime.control_loop import ControlLoop
from action_chunk_processing import ActionChunkProcessor


class Capture:
    def __init__(self):
        self.events = []

    def publish(self, *, data):
        self.events.append(json.loads(data))


def logger(process, capture):
    return InferenceEventPublisher('lerobot', process, publisher=capture)


def test_disabled_logging_never_copies_tensors_or_publishes():
    capture = Capture()
    events = logger('engine', capture)
    set_engine_logger(events)
    class Tensor:
        def detach(self):
            raise AssertionError('must not copy when disabled')
    log_values('observation_state', Tensor())
    with InferenceTrace().stage('policy'):
        pass
    assert capture.events == []
    set_engine_logger(None)


@pytest.mark.parametrize('mode', ['async', 'sync', 'chunkrl'])
@pytest.mark.parametrize('robot_mode', [False, True])
def test_request_to_command_timeline_correlates_across_processes(mode, robot_mode):
    capture = Capture()
    main_log, engine_log = logger('main', capture), logger('engine', capture)
    class Engine:
        _action_keys = ['arm_right']
        def load_policy(self, request):
            return dict(success=True, action_keys=self._action_keys)
        def get_action_chunk(self, request):
            trace = InferenceTrace()
            for stage in ('observation', 'preprocess', 'policy', 'postprocess'):
                with trace.stage(stage):
                    if stage == 'observation':
                        log_values('observation_state', np.zeros((1, 2, 10)))
            self._last_inference_telemetry = trace.finish()
            return dict(success=True, chunk_size=1, action_dim=2, action_chunk=np.array([.2, .4]))
    worker = EngineWorker(Engine())
    worker._event_publisher = engine_log
    requester = InferenceRequester(SimpleNamespace(call=lambda req, **kw: worker.handle(req)),
                                   event_logger=main_log)
    assert requester.load_policy(SimpleNamespace(record_inference_mode=True)).success
    published, previews = [], []
    loop = ControlLoop(requester, event_logger=main_log, action_request_mode=mode)
    loop._robot = SimpleNamespace(publish_action=lambda a, k: published.append(a.copy()),
        publish_action_preview=lambda a, k: previews.append(a.copy()), publish_idle_action=lambda k: None)
    loop._processor = ActionChunkProcessor(inference_hz=30, control_hz=100)
    loop._action_keys = ['arm_right']
    loop._publish_to_robot = robot_mode
    loop._running = True
    loop._should_request_actions = lambda processor: False
    loop._request_and_buffer('', loop._generation, mode)
    while loop._processor.buffer_size:
        loop.tick()
    loop.tick()  # starvation is observable even though there is no new request
    assert loop.pause()
    events = capture.events
    request = next(e for e in events if e['event'] == 'request_started' and e['command'] == 1)
    seq = request['seq_id']
    related = [e for e in events if e.get('seq_id') == seq]
    names = [e['event'] for e in related]
    assert names[0] == 'request_started'
    assert names.index('engine_request_finished') < names.index('request_finished') < names.index('buffer_finished')
    assert len({e['session_id'] for e in events}) == 1
    assert len({e['clock_id'] for e in events}) == 1
    assert all(e['wall_time_ns'] > 10**18 and e['monotonic_ns'] > 0 for e in events)
    for stage in ('observation', 'preprocess', 'policy', 'postprocess'):
        start, end = [e for e in related if e.get('stage') == stage]
        assert start['event'] == 'stage_started' and end['event'] == 'stage_finished'
        assert end['monotonic_ns'] >= start['monotonic_ns']
    assert next(e for e in related if e['event'] == 'observation_state')['values'] == np.zeros((1, 2, 10)).tolist()
    assert next(e for e in related if e['event'] == 'action_chunk')['values'] == [.2, .4]
    ticks = [e for e in events if e['event'] == 'control_tick']
    assert ticks[-1]['state'] == 'buffer_empty' and not ticks[-1]['command_published']
    assert all(e['command_published'] == robot_mode for e in ticks[:-1])
    assert len(published) == (len(previews) if robot_mode else 0)
    assert [e['state'] for e in events if e['event'] == 'control_state'] == ['executing', 'buffer_empty', 'paused']
    for process in ('main', 'engine'):
        indices = [e['event_index'] for e in events if e['process'] == process]
        assert indices == sorted(set(indices))
    loop._robot = loop._processor = None
    set_engine_logger(None)


def test_timeout_is_recorded_as_failure_without_an_action():
    capture = Capture()
    events = logger('main', capture)
    events.configure(True, 'timeout-session')
    def timeout(*a, **kw):
        raise TimeoutError()
    requester = InferenceRequester(SimpleNamespace(call=timeout), event_logger=events)
    response = requester.get_action('')
    assert not response.success
    assert [e['event'] for e in capture.events] == ['request_started', 'request_finished']
    assert not capture.events[-1]['success']
    assert capture.events[0]['seq_id'] == capture.events[-1]['seq_id']


def test_logging_failure_does_not_break_control_publication():
    def unavailable(**kw):
        raise RuntimeError('transport unavailable')
    events = logger('main', SimpleNamespace(publish=unavailable))
    events.configure(True, 'failed-logger')
    events.emit('control_tick', values=[1.0])  # no exception reaches control loop


def test_buffering_failure_has_a_terminal_event_and_no_command():
    capture = Capture()
    events = logger('main', capture)
    events.configure(True, 'bad-buffer')
    response = SimpleNamespace(success=True, seq_id=42, chunk_size=1, action_dim=2,
                               action_list=[.1, .2], telemetry='{}')
    loop = ControlLoop(SimpleNamespace(get_action=lambda *a, **kw: response), event_logger=events)
    def reject(*a, **kw):
        raise ValueError('alignment failed')
    loop._processor = SimpleNamespace(buffer_size=0, output_hz=100, push_actions=reject)
    loop._running = True
    loop._request_and_buffer('', loop._generation)
    assert [e['event'] for e in capture.events] == ['buffer_started', 'buffer_finished']
    assert capture.events[-1]['seq_id'] == 42
    assert not capture.events[-1]['accepted'] and capture.events[-1]['reason'] == 'alignment failed'
    assert loop._chunkrl_reset_pending
    loop._processor = None
