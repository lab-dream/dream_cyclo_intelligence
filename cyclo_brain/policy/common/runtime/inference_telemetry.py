"""Per-request inference timings and pose previews shared by policy backends."""

import json
import logging
import sys
import time
from contextlib import contextmanager


class InferenceTrace:
    def __init__(self):
        self.started = time.perf_counter()
        self.timings = {}

    @staticmethod
    def synchronize():
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name):
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - started) * 1000

    def finish(self, **extra):
        return dict(timings_ms={**self.timings, "engine": (time.perf_counter() - self.started) * 1000}, **extra)


class TelemetryPublisher:
    """JSON over std_msgs/String keeps the GUI independent of policy message types."""

    def __init__(self, backend, **connection):
        from zenoh_ros2_sdk import ROS2Publisher

        self.backend = backend
        self.publisher = ROS2Publisher(topic=f"/{backend}/inference_telemetry",
                                       msg_type="std_msgs/msg/String", msg_definition="string data\n",
                                       **connection)

    def publish(self, value):
        try:
            value = dict(value, backend=self.backend, timestamp_ms=time.time() * 1000)
            self.publisher.publish(data=json.dumps(value, allow_nan=False))
        except Exception:
            # Diagnostics must never prevent action processing or publication.
            logging.getLogger(__name__).warning("Cannot publish inference telemetry", exc_info=True)

    def close(self):
        self.publisher.close()
