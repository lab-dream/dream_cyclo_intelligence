"""Opt-in, timestamped inference events recorded by the dataset rosbag recorder.

Wall time aligns events with ROS messages. Monotonic time measures durations
without NTP jumps; compare it only when clock_id matches. This module never
opens files or blocks waiting for a recorder.
"""

import contextvars
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

_request_id = contextvars.ContextVar("inference_request_id", default=None)
_engine_logger = None


def clock_stamp():
    return dict(wall_time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())


def set_engine_logger(logger):
    global _engine_logger
    _engine_logger = logger


def engine_event(event, **fields):
    if _engine_logger is not None:
        _engine_logger.emit(event, seq_id=_request_id.get(), **fields)


def log_values(event, values, **fields):
    """Copy small state/action arrays only when recording is enabled. No images."""
    if _engine_logger is None or not _engine_logger.enabled:
        return
    try:
        def convert(value):
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            return value.tolist() if hasattr(value, "tolist") else value
        engine_event(event, values=convert(values), **fields)
    except Exception:
        logging.getLogger(__name__).warning("Cannot encode inference values", exc_info=True)


class InferenceEventPublisher:
    def __init__(self, backend, process, publisher=None, **connection):
        if publisher is None:
            from zenoh_ros2_sdk import ROS2Publisher
            publisher = ROS2Publisher(topic=f"/{backend}/inference_events",
                msg_type="std_msgs/msg/String", msg_definition="string data\n", **connection)
        self.publisher = publisher
        self.backend = backend
        self.process = process
        self.process_id = uuid.uuid4().hex
        try:
            self.clock_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            self.clock_id = self.process_id
        self.pid = os.getpid()
        self.session_id = ""
        self.enabled = False
        self._sequence = 0
        self._lock = threading.Lock()

    def configure(self, enabled, session_id=""):
        self.enabled = bool(enabled)
        self.session_id = str(session_id)

    def emit(self, event, *, stamp=None, **fields):
        if not self.enabled:
            return
        try:
            with self._lock:
                stamp = stamp or clock_stamp()
                self._sequence += 1
                value = dict(schema_version=1, backend=self.backend, process=self.process,
                    process_id=self.process_id, pid=self.pid, clock_id=self.clock_id,
                    session_id=self.session_id, event_index=self._sequence,
                    event=event, **stamp, **fields)
                self.publisher.publish(data=json.dumps(value, allow_nan=False))
        except Exception:
            # Logging failures must not change robot commands. Index gaps let
            # offline analysis detect missing records instead of assuming idle.
            logging.getLogger(__name__).warning("Cannot publish inference event", exc_info=True)

    def close(self):
        self.publisher.close()
