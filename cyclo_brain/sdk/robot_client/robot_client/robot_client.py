#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

"""
RobotClient - High-level abstraction for robot sensor data and control.

Provides simple Python API over zenoh_ros2_sdk, hiding all Zenoh/ROS2 details.
Users only need to specify robot type to get automatic topic subscription.
"""
import os
import sys
import time
import threading
import logging
import math
from bisect import bisect_left
from collections import deque
from pathlib import Path
from typing import Optional, Union

import numpy as np
import cv2

# Add zenoh_ros2_sdk to path if not already available
_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "")
if _SDK_PATH and _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

from zenoh_ros2_sdk import ROS2Publisher, ROS2Subscriber, get_message_class  # noqa: E402


# -- robot config schema helper -----------------------------------------------
# shared/robot_configs/ is bind-mounted into the policy container at
# /orchestrator_config/, so schema.py lands beside the per-robot yamls.
# The module is intentionally self-contained (no `shared` package
# imports) so it can be picked up as a standalone file from that mount.
_SCHEMA_DIR = os.environ.get("ORCHESTRATOR_CONFIG_PATH", "/orchestrator_config")
if os.path.isdir(_SCHEMA_DIR) and _SCHEMA_DIR not in sys.path:
    sys.path.insert(0, _SCHEMA_DIR)
try:
    import schema as robot_schema  # type: ignore[import-not-found]
except ImportError:
    _src = Path(__file__).resolve()
    for _parent in _src.parents:
        _cand = _parent / "shared" / "robot_configs"
        if _cand.is_dir():
            sys.path.insert(0, str(_cand))
            break
    import schema as robot_schema  # type: ignore[import-not-found]


logger = logging.getLogger("robot_client")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _deadband(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def _build_runtime_config(section: dict) -> dict:
    """Translate the VLA-semantic schema into the cameras/joint_groups/
    sensors shape the RobotClient + inference engines consume.

    * ``observation.images``  → ``cameras``.
    * ``observation.state.<g>`` with JointState msg_type → physical
      follower joint group named ``follower_<g>``.
    * ``observation.state.<g>`` with Odometry msg_type   → ``sensors["odom"]``
      (treated as a sensor-backed state modality by GR00T).
    * Each ``action.<modality>`` (excluding mobile/Twist) gets a SYNTHETIC
      ``follower_<modality>`` joint_group with ``parent`` pointing at the
      first physical follower; ``_update_joint`` slices the parent's
      message by the action's ``joint_names`` to populate it.
    """
    cameras = robot_schema.get_image_topics(section)
    state_groups = robot_schema.get_state_groups(section)
    action_groups = robot_schema.get_action_groups(section)

    joint_groups: dict = {}
    sensors: dict = {}
    physical_follower_name: Optional[str] = None

    for name, cfg in state_groups.items():
        msg_type = cfg["msg_type"]
        if msg_type == "sensor_msgs/msg/JointState":
            group_name = f"follower_{name}"
            joint_groups[group_name] = {
                "topic": cfg["topic"],
                "msg_type": msg_type,
                "role": "follower",
                "joint_names": list(cfg["joint_names"]),
            }
            if physical_follower_name is None:
                physical_follower_name = group_name
        elif msg_type == "nav_msgs/msg/Odometry":
            sensors["odom"] = {
                "topic": cfg["topic"],
                "msg_type": msg_type,
            }
        else:
            # Unknown state msg_type — keep it in joint_groups for
            # diagnostics; subscribers will pick it up via the generic
            # JointState callback unless a more specific shape lands.
            joint_groups[f"follower_{name}"] = {
                "topic": cfg["topic"],
                "msg_type": msg_type,
                "role": "follower",
                "joint_names": list(cfg["joint_names"]),
            }

    if physical_follower_name is not None:
        for modality, cfg in action_groups.items():
            if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                # action.mobile is command-only; observation RobotClient
                # instances stay read-only.
                continue
            child_name = f"follower_{modality}"
            if child_name in joint_groups:
                # A physical follower already covers this modality.
                continue
            joint_groups[child_name] = {
                "parent": physical_follower_name,
                "role": "follower",
                "joint_names": list(cfg["joint_names"]),
            }

    return {
        "cameras": cameras,
        "joint_groups": joint_groups,
        "sensors": sensors,
    }


# Compatibility re-export for older engine code. Same VLA-semantic section in,
# same runtime-config dict out.
def derive_robot_config(section: dict) -> dict:
    return _build_runtime_config(section)


class RobotClient:
    """High-level robot interface over zenoh_ros2_sdk.

    Usage:
        robot = RobotClient("ffw_sg2_rev1")
        robot.wait_for_ready(timeout=10.0)
        images = robot.get_images()
        joints = robot.get_joint_positions()
    """

    def __init__(
        self,
        robot_type: str,
        sync_check: bool = False,
        sync_threshold_ms: float = 33.0,
        router_ip: str = "127.0.0.1",
        router_port: int = 7447,
        domain_id: Optional[int] = None,
        enable_command_publishers: bool = False,
        enable_preview_publisher: bool = False,
        command_action_keys: Optional[list[str]] = None,
    ):
        section = robot_schema.load_robot_section(robot_type)
        # Phase 4: yaml is VLA-semantic (observation.images / state +
        # action.<modality>). _build_runtime_config translates that into
        # the cameras / joint_groups / sensors shape RobotClient and the
        # downstream inference engines have always consumed.
        self._config = _build_runtime_config(section)

        self._robot_type = robot_type
        self._sync_check = sync_check
        self._sync_threshold_ms = sync_threshold_ms
        self._router_ip = router_ip
        self._router_port = router_port
        self._domain_id = domain_id
        self._enable_command_publishers = bool(enable_command_publishers)
        self._enable_preview_publisher = bool(enable_preview_publisher)
        self._action_groups = robot_schema.get_action_groups(section)

        # Thread-safe data stores
        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray] = {}
        self._image_timestamps: dict[str, float] = {}
        self._joint_positions: dict[str, np.ndarray] = {}
        self._joint_velocities: dict[str, np.ndarray] = {}
        self._joint_efforts: dict[str, np.ndarray] = {}
        self._joint_timestamps: dict[str, float] = {}
        self._joint_positions_by_name: dict[str, float] = {}
        self._joint_position_timestamps_by_name: dict[str, float] = {}
        self._sensors: dict[str, dict] = {}
        self._sensor_timestamps: dict[str, float] = {}
        self._observation_history = None
        self._task_instruction: str = ""

        self._subscribers: list = []
        self._command_publishers: dict[str, ROS2Publisher] = {}
        self._preview_publisher: Optional[ROS2Publisher] = None
        self._command_msg_types: dict[str, str] = {}
        self._command_joint_names: dict[str, list[str]] = {}
        self._action_keys = (
            sorted(self._action_groups) if command_action_keys is None
            else list(command_action_keys)
        )
        if command_action_keys is not None and (
            not self._action_keys or len(set(self._action_keys)) != len(self._action_keys)
        ):
            raise ValueError("command_action_keys must be non-empty and unique")
        unknown = set(self._action_keys) - set(self._action_groups)
        if unknown:
            raise ValueError(f"unknown command action keys: {sorted(unknown)}")
        self._cmd_vel_linear_deadband = max(
            0.0,
            _float_env("CMD_VEL_LINEAR_DEADBAND", 0.0),
        )
        self._cmd_vel_angular_deadband = max(
            0.0,
            _float_env("CMD_VEL_ANGULAR_DEADBAND", 0.0),
        )
        self._initial_pose_sync_state_max_age_s = _float_env(
            "INITIAL_POSE_SYNC_STATE_MAX_AGE_S",
            1.0,
        )
        if (
            not math.isfinite(self._initial_pose_sync_state_max_age_s)
            or self._initial_pose_sync_state_max_age_s <= 0.0
        ):
            logger.warning(
                "INITIAL_POSE_SYNC_STATE_MAX_AGE_S must be positive and finite; "
                "using 1.0"
            )
            self._initial_pose_sync_state_max_age_s = 1.0
        self._closed = False

        self._init_subscriptions()
        if self._enable_command_publishers:
            self._init_command_publishers()
            if self._cmd_vel_linear_deadband or self._cmd_vel_angular_deadband:
                logger.info(
                    "cmd_vel deadband enabled: linear=%s angular=%s",
                    self._cmd_vel_linear_deadband,
                    self._cmd_vel_angular_deadband,
                )
        if self._enable_preview_publisher:
            self._init_preview_publisher()
        logger.info(f"RobotClient initialized: {robot_type} "
                     f"({len(self._config.get('cameras', {}))} cameras, "
                     f"{len(self._config.get('joint_groups', {}))} joint groups)")

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _init_subscriptions(self):
        """Subscribe to all configured topics.

        Joint groups carrying a ``parent`` field have no physical topic of
        their own — they're synthetic per-modality views over a sibling
        group's data. ``_update_joint`` propagates from parent → children
        by name-based slicing inside the callback.
        """
        # Cameras
        for cam_name, cam_cfg in self._config.get("cameras", {}).items():
            sub = ROS2Subscriber(
                topic=cam_cfg["topic"],
                msg_type=cam_cfg["msg_type"],
                callback=lambda msg, name=cam_name: self._update_image(name, msg),
            )
            self._subscribers.append(sub)
            logger.debug(f"Subscribed camera: {cam_name} -> {cam_cfg['topic']}")

        # Index parent → list of child group names so the upper-body
        # callback knows which slices to populate per message.
        self._joint_children: dict[str, list[str]] = {}
        for child_name, child_cfg in self._config.get("joint_groups", {}).items():
            parent = child_cfg.get("parent")
            if parent:
                self._joint_children.setdefault(parent, []).append(child_name)

        # Joint groups — only those with their own physical topic.
        for group_name, group_cfg in self._config.get("joint_groups", {}).items():
            if group_cfg.get("parent"):
                logger.debug(
                    f"Skipped joint subscription: {group_name} "
                    f"(synthetic view of {group_cfg['parent']})"
                )
                continue
            sub = ROS2Subscriber(
                topic=group_cfg["topic"],
                msg_type=group_cfg["msg_type"],
                callback=lambda msg, name=group_name: self._update_joint(name, msg),
            )
            self._subscribers.append(sub)
            logger.debug(f"Subscribed joint: {group_name} -> {group_cfg['topic']}")

        # Additional sensors. ``sensor_cfg`` may carry an optional
        # ``type_hash`` override — escape hatch for messages where
        # zenoh_ros2_sdk's hash computation needs to be pinned to a known
        # wire hash. Default is auto-compute via the SDK.
        for sensor_name, sensor_cfg in self._config.get("sensors", {}).items():
            sub_kwargs = dict(
                topic=sensor_cfg["topic"],
                msg_type=sensor_cfg["msg_type"],
                callback=lambda msg, name=sensor_name: self._update_sensor(name, msg),
            )
            if sensor_cfg.get("type_hash"):
                sub_kwargs["type_hash"] = sensor_cfg["type_hash"]
            sub = ROS2Subscriber(**sub_kwargs)
            self._subscribers.append(sub)
            logger.debug(f"Subscribed sensor: {sensor_name} -> {sensor_cfg['topic']}")

    def _init_command_publishers(self):
        """Create publishers for configured action topics."""
        common = {
            "router_ip": self._router_ip,
            "router_port": self._router_port,
        }
        if self._domain_id is not None:
            common["domain_id"] = self._domain_id

        for action_key in self._action_keys:
            cfg = self._action_groups[action_key]
            publisher_key = f"leader_{action_key}"
            self._command_msg_types[publisher_key] = cfg["msg_type"]
            self._command_joint_names[publisher_key] = list(cfg.get("joint_names", []))
            self._command_publishers[publisher_key] = ROS2Publisher(
                topic=cfg["topic"],
                msg_type=cfg["msg_type"],
                **common,
            )
            logger.debug(
                "Command publisher: %s -> %s (%s)",
                publisher_key,
                cfg["topic"],
                cfg["msg_type"],
            )

    def _init_preview_publisher(self):
        """Create a unified trajectory preview publisher for the 3D viewer."""
        common = {
            "router_ip": self._router_ip,
            "router_port": self._router_port,
        }
        if self._domain_id is not None:
            common["domain_id"] = self._domain_id
        self._preview_publisher = ROS2Publisher(
            topic="/inference/trajectory_preview",
            msg_type="trajectory_msgs/msg/JointTrajectory",
            **common,
        )
        logger.debug("Action preview publisher: /inference/trajectory_preview")

    # ------------------------------------------------------------------ #
    # Callback handlers
    # ------------------------------------------------------------------ #

    def _update_image(self, cam_name: str, msg):
        """CompressedImage -> BGR numpy array."""
        try:
            received_at = time.monotonic()
            data = msg.data
            if isinstance(data, (list, tuple)):
                data = bytes(data)
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                with self._lock:
                    self._images[cam_name] = img
                    self._image_timestamps[cam_name] = time.time()
                    self._append_observation("image", cam_name, received_at, img)
        except Exception as e:
            logger.warning(f"Failed to decode image from {cam_name}: {e}")

    def _update_joint(self, group_name: str, msg):
        """JointState -> np.ndarray(float32).

        Both physical groups and synthetic children follow their configured
        ``joint_names`` order. A message may include extra joints (e.g. wheels)
        or a different order; its ``name`` array is the source of the mapping.
        Incomplete views are removed rather than retaining an older pose.
        """
        try:
            msg_names = list(getattr(msg, "name", []))
            position = list(getattr(msg, "position", []))
            velocity = list(getattr(msg, "velocity", []))
            effort = list(getattr(msg, "effort", []))
            now = time.time()
            received_monotonic = time.monotonic()
            valid_names = bool(msg_names) and len(set(msg_names)) == len(msg_names)
            valid_position = valid_names and len(position) == len(msg_names)
            name_to_idx = {name: index for index, name in enumerate(msg_names)}
            children = getattr(self, "_joint_children", {}).get(group_name, [])
            with self._lock:
                if valid_position:
                    self._joint_positions_by_name.update(
                        {
                            name: float(value)
                            for name, value in zip(msg_names, position, strict=True)
                        }
                    )
                    self._joint_position_timestamps_by_name.update(
                        {name: received_monotonic for name in msg_names}
                    )
                elif position:
                    logger.warning("Invalid JointState names/positions from %s", group_name)

                for view in [group_name, *children]:
                    cfg = self._config["joint_groups"].get(view, {})
                    wanted = cfg.get("joint_names", []) or msg_names
                    indices = [name_to_idx[name] for name in wanted if name in name_to_idx]
                    complete = valid_names and len(indices) == len(wanted)
                    for values, store in (
                        (position, self._joint_positions),
                        (velocity, self._joint_velocities),
                        (effort, self._joint_efforts),
                    ):
                        if complete and len(values) == len(msg_names):
                            store[view] = np.asarray(
                                [values[index] for index in indices], dtype=np.float32
                            )
                        else:
                            store.pop(view, None)
                    if complete and valid_position:
                        self._joint_timestamps[view] = now
                    else:
                        self._joint_timestamps.pop(view, None)
                self._append_observation("joint", group_name, received_monotonic, {
                    view: self._joint_positions[view]
                    for view in [group_name, *children] if view in self._joint_positions
                })
        except Exception as e:
            logger.warning(f"Failed to parse joint from {group_name}: {e}")

    def _update_sensor(self, sensor_name: str, msg):
        """Parse sensor messages (Odometry, Twist, etc.)."""
        try:
            received_at = time.monotonic()
            data = {}
            if sensor_name == "odom":
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                lin = msg.twist.twist.linear
                ang = msg.twist.twist.angular
                data = {
                    "position": np.array([pos.x, pos.y, pos.z], dtype=np.float32),
                    "orientation": np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float32),
                    "linear_velocity": np.array([lin.x, lin.y, lin.z], dtype=np.float32),
                    "angular_velocity": np.array([ang.x, ang.y, ang.z], dtype=np.float32),
                }
            elif sensor_name == "cmd_vel":
                data = {
                    "linear": np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32),
                    "angular": np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float32),
                }
            else:
                data = {"raw": str(msg)}

            with self._lock:
                self._sensors[sensor_name] = data
                self._sensor_timestamps[sensor_name] = time.time()
                self._append_observation("sensor", sensor_name, received_at, data)
        except Exception as e:
            logger.warning(f"Failed to parse sensor {sensor_name}: {e}")

    # ------------------------------------------------------------------ #
    # Observation history
    # ------------------------------------------------------------------ #

    def enable_observation_history(self, camera_names, horizon_s=0.5):
        """Opt in to bounded receive-time history; callbacks never do policy work.

        Entries reference newly decoded/parsed arrays, which callbacks replace
        rather than modify. Only policy cameras retain image history. All access
        to these rings uses the existing client lock.
        """
        if not math.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("Observation history horizon must be positive and finite")
        cameras = tuple(camera_names)
        if set(cameras) - self._config.get("cameras", {}).keys():
            raise ValueError("Unknown observation history camera")
        streams = [("image", name) for name in cameras]
        streams += [("joint", name) for name, cfg in self._config.get("joint_groups", {}).items()
                    if cfg.get("role") == "follower" and not cfg.get("parent")]
        streams += [("sensor", name) for name in self._config.get("sensors", {})]
        with self._lock:
            self._observation_history = {key: deque(maxlen=256) for key in streams}
            self._observation_history_horizon_s = horizon_s

    def _append_observation(self, kind, name, timestamp, value):
        """Caller holds _lock. Ignore late callbacks that finished out of order."""
        history = getattr(self, "_observation_history", None)
        ring = history.get((kind, name)) if history is not None else None
        if ring is None or (ring and timestamp <= ring[-1][0]):
            return
        ring.append((timestamp, value))
        cutoff = timestamp - self._observation_history_horizon_s
        while len(ring) > 1 and ring[0][0] < cutoff:
            ring.popleft()

    def get_observation_window(self, n_obs_steps, interval_s, *, sensor_names=()):
        """Return an immutable-in-time window immediately, or raise ValueError.

        Use the latest policy-camera timestamp as the anchor and choose nearest
        received samples within half an observation interval. Sampling is bounded
        by the anchor (no samples from beyond this window), and camera frames
        must be distinct. Receipt timestamps share the local monotonic clock;
        ROS header clocks may belong to another host or Gazebo simulation time.
        """
        if not isinstance(n_obs_steps, int) or n_obs_steps < 1 or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("Observation steps and interval must be positive")
        max_age_s = max(0.25, 3 * interval_s)
        tolerance = max(0.01, interval_s / 2)
        with self._lock:
            history = getattr(self, "_observation_history", None)
            if history is None:
                raise ValueError("Observation history is not enabled")
            keys = [key for key in history if key[0] != "sensor"]
            keys += [("sensor", name) for name in sensor_names]
            rings = {}
            now = time.monotonic()
            for key in keys:
                ring = history.get(key)
                if not ring:
                    raise ValueError(f"Observation history warming up: missing {key[0]}:{key[1]}")
                if now - ring[-1][0] > max_age_s:
                    raise ValueError(f"Stale observation stream: {key[0]}:{key[1]}")
                rings[key] = list(ring)
            camera_times = [ring[-1][0] for key, ring in rings.items() if key[0] == "image"]
            if not camera_times:
                raise ValueError("No policy camera in observation history")
            anchor = max(camera_times)
            targets = [anchor - (n_obs_steps - 1 - index) * interval_s for index in range(n_obs_steps)]
            selected = {}
            for key, ring in rings.items():
                ring = [item for item in ring if item[0] <= anchor + 1e-9]
                times = [item[0] for item in ring]
                samples = []
                for target in targets:
                    index = bisect_left(times, target)
                    candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
                    if not candidates:
                        raise ValueError(f"Observation history warming up: {key[0]}:{key[1]}")
                    index = min(candidates, key=lambda i: abs(times[i] - target))
                    if abs(times[index] - target) > tolerance + 1e-9:
                        raise ValueError(f"Observation history gap: {key[0]}:{key[1]}")
                    samples.append(ring[index])
                if key[0] == "image" and any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
                    raise ValueError(f"Observation history needs distinct camera frames: {key[1]}")
                selected[key] = samples
        # Data referenced by selected entries survives eviction. Convert/copy
        # outside the callback lock so inference never blocks sensor acquisition.
        observations = []
        for index in range(n_obs_steps):
            images, joints, sensors = {}, {}, {}
            for (kind, name), samples in selected.items():
                value = samples[index][1]
                if kind == "image":
                    images[name] = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                elif kind == "joint":
                    joints.update({group: values.copy() for group, values in value.items()})
                else:
                    sensors[name] = {key: item.copy() if isinstance(item, np.ndarray) else item
                                     for key, item in value.items()}
            observations.append(dict(images=images, joints=joints, sensors=sensors))
        metadata = dict(timestamp_basis="receive_monotonic", target_timestamps_s=targets,
                        latest_age_ms=max(0.0, now - anchor) * 1000,
                        max_alignment_error_ms=max(abs(samples[i][0] - target) * 1000
                            for samples in selected.values() for i, target in enumerate(targets)),
                        stream_timestamps_s={f"{kind}:{name}": [item[0] for item in samples]
                                             for (kind, name), samples in selected.items()})
        return observations, metadata

    def wait_for_observation_history(self, n_obs_steps, interval_s, *, sensor_names=(), timeout=10.0):
        """Warm up only during LOAD; normal inference requests never wait."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.get_observation_window(n_obs_steps, interval_s, sensor_names=sensor_names)
                return
            except ValueError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    # ------------------------------------------------------------------ #
    # Image API
    # ------------------------------------------------------------------ #

    @property
    def camera_names(self) -> list[str]:
        return list(self._config.get("cameras", {}).keys())

    def get_images(
        self,
        resize: Optional[tuple[int, int]] = None,
        format: str = "bgr",
    ) -> dict[str, np.ndarray]:
        """Get all camera images.

        Args:
            resize: Optional (width, height) tuple. None = original size.
            format: "bgr" (default) or "rgb".
        """
        with self._lock:
            result = {k: v.copy() for k, v in self._images.items()}
        if resize:
            result = {k: cv2.resize(v, resize) for k, v in result.items()}
        if format == "rgb":
            result = {k: cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for k, v in result.items()}
        return result

    def get_image(
        self,
        camera_name: str,
        resize: Optional[tuple[int, int]] = None,
        format: str = "bgr",
    ) -> Optional[np.ndarray]:
        """Get single camera image."""
        with self._lock:
            img = self._images.get(camera_name)
            if img is None:
                return None
            img = img.copy()
        if resize:
            img = cv2.resize(img, resize)
        if format == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def is_image_ready(self, camera_name: str) -> bool:
        with self._lock:
            return camera_name in self._images

    def get_image_timestamp(self, camera_name: str) -> Optional[float]:
        with self._lock:
            return self._image_timestamps.get(camera_name)

    # ------------------------------------------------------------------ #
    # Joint API
    # ------------------------------------------------------------------ #

    @property
    def joint_group_names(self) -> list[str]:
        return list(self._config.get("joint_groups", {}).keys())

    @property
    def total_dof(self) -> int:
        return self._config.get("total_dof", 0)

    def get_joint_names(self, group_name: str) -> list[str]:
        cfg = self._config.get("joint_groups", {}).get(group_name, {})
        return cfg.get("joint_names", [])

    def get_dof(self, group_name: str) -> int:
        cfg = self._config.get("joint_groups", {}).get(group_name, {})
        return cfg.get("dof", 0)

    def get_joint_positions(
        self, group: Optional[str] = None
    ) -> Union[dict[str, np.ndarray], np.ndarray]:
        """Get joint positions. Returns dict if no group, or np.ndarray for specific group."""
        with self._lock:
            if group:
                arr = self._joint_positions.get(group)
                return arr.copy() if arr is not None else np.array([], dtype=np.float32)
            return {k: v.copy() for k, v in self._joint_positions.items()}

    def get_joint_velocities(
        self, group: Optional[str] = None
    ) -> Union[dict[str, np.ndarray], np.ndarray]:
        with self._lock:
            if group:
                arr = self._joint_velocities.get(group)
                return arr.copy() if arr is not None else np.array([], dtype=np.float32)
            return {k: v.copy() for k, v in self._joint_velocities.items()}

    def get_joint_efforts(
        self, group: Optional[str] = None
    ) -> Union[dict[str, np.ndarray], np.ndarray]:
        with self._lock:
            if group:
                arr = self._joint_efforts.get(group)
                return arr.copy() if arr is not None else np.array([], dtype=np.float32)
            return {k: v.copy() for k, v in self._joint_efforts.items()}

    def is_joint_ready(self, group_name: str) -> bool:
        with self._lock:
            return group_name in self._joint_positions

    def get_joint_timestamp(self, group_name: str) -> Optional[float]:
        with self._lock:
            return self._joint_timestamps.get(group_name)

    # ------------------------------------------------------------------ #
    # Sensor API
    # ------------------------------------------------------------------ #

    def get_odom(self) -> Optional[dict]:
        with self._lock:
            return self._sensors.get("odom")

    def is_sensor_ready(self, sensor_name: str) -> bool:
        with self._lock:
            return sensor_name in self._sensors

    # ------------------------------------------------------------------ #
    # Command API
    # ------------------------------------------------------------------ #

    @property
    def action_keys(self) -> list[str]:
        return list(self._action_keys)

    def publish_action(self, action: np.ndarray, action_keys: Optional[list[str]] = None) -> None:
        """Publish one flat action vector to the robot command topics.

        Main process control loops use this method. Engine process instances keep
        ``enable_command_publishers=False`` and remain read-only.
        """
        # Validate the entire command before any group is sent, including the
        # session's allowed modalities and the exact flat-vector dimension.
        segments = self._build_action_segments(action, action_keys)
        for publisher, joint_names, segment, msg_type in segments:
            if msg_type == "geometry_msgs/msg/Twist":
                self._publish_twist(publisher, segment)
            else:
                self._publish_joint_trajectory(
                    publisher,
                    joint_names,
                    segment,
                )

    def publish_idle_action(self, action_keys: Optional[list[str]] = None) -> None:
        """Publish safe idle commands for velocity-like action topics.

        Position trajectory controllers hold their last target when the action
        buffer is empty. Twist command topics do not have that same semantics,
        so publish an explicit zero velocity for any commanded Twist modality.
        """
        if not self._command_publishers:
            raise RuntimeError("RobotClient command publishers are not enabled")

        keys = self._validated_command_keys(action_keys)
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None or cfg["msg_type"] != "geometry_msgs/msg/Twist":
                continue
            publisher = self._command_publishers.get(f"leader_{publish_key}")
            if publisher is None:
                continue
            self._publish_twist(publisher, np.zeros(3, dtype=np.float64))

    def publish_initial_pose_sync(
        self,
        action: np.ndarray,
        action_keys: Optional[list[str]] = None,
        duration_s: float = 5.0,
    ) -> None:
        """Publish one validated action as a slow position-only transition.

        Twist modalities are forced to zero. A complete current-position
        snapshot is required before any target is published so an interrupted
        transition can be replaced with a hold trajectory.
        """
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be a positive finite value")

        segments = self._build_action_segments(action, action_keys)
        hold_segments = self._build_current_position_segments(action_keys)
        if not hold_segments:
            raise ValueError("initial pose sync requires a position action group")

        try:
            for publisher, _joint_names, _values, msg_type in segments:
                if msg_type == "geometry_msgs/msg/Twist":
                    self._publish_twist(publisher, np.zeros(3, dtype=np.float64))
            for publisher, joint_names, values, msg_type in segments:
                if msg_type != "geometry_msgs/msg/Twist":
                    self._publish_joint_trajectory(
                        publisher,
                        joint_names,
                        values,
                        time_from_start_s=duration_s,
                    )
        except Exception:
            try:
                self._publish_position_segments(hold_segments, duration_s=0.1)
                self.publish_idle_action(action_keys)
            except Exception as hold_error:
                logger.error(
                    "failed to hold current pose after initial sync publish error: %s",
                    hold_error,
                )
            raise

    def publish_current_pose_hold(
        self,
        action_keys: Optional[list[str]] = None,
        duration_s: float = 0.1,
    ) -> None:
        """Replace active position trajectories with the latest joint pose."""
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be a positive finite value")
        segments = self._build_current_position_segments(action_keys)
        if not segments:
            raise ValueError("current pose hold requires a position action group")
        self.publish_idle_action(action_keys)
        self._publish_position_segments(segments, duration_s)

    def _build_action_segments(
        self,
        action: np.ndarray,
        action_keys: Optional[list[str]],
    ) -> list[tuple[ROS2Publisher, list[str], np.ndarray, str]]:
        if not self._command_publishers:
            raise RuntimeError("RobotClient command publishers are not enabled")

        keys = self._validated_command_keys(action_keys)
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("action contains non-finite values")

        segments = []
        offset = 0
        resolved_keys: set[str] = set()
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            if publish_key in resolved_keys:
                raise ValueError(f"duplicate action key: {action_key}")
            resolved_keys.add(publish_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None:
                raise ValueError(f"unknown action key: {action_key}")
            publisher_key = f"leader_{publish_key}"
            publisher = self._command_publishers.get(publisher_key)
            if publisher is None:
                raise RuntimeError(f"publisher unavailable for action key: {action_key}")
            msg_type = cfg["msg_type"]
            joint_names = list(self._command_joint_names.get(publisher_key, []))
            width = 3 if msg_type == "geometry_msgs/msg/Twist" else len(joint_names)
            if width <= 0:
                raise ValueError(f"action key has no configured dimensions: {action_key}")
            segment = values[offset:offset + width]
            if len(segment) != width:
                raise ValueError(
                    f"action dimension too small for {action_key}: "
                    f"expected {width}, got {len(segment)}"
                )
            offset += width
            segments.append((publisher, joint_names, segment.copy(), msg_type))

        if offset != len(values):
            raise ValueError(
                f"action dimension mismatch: expected {offset}, got {len(values)}"
            )
        return segments

    def _build_current_position_segments(
        self,
        action_keys: Optional[list[str]],
    ) -> list[tuple[ROS2Publisher, list[str], np.ndarray]]:
        keys = self._validated_command_keys(action_keys)
        planned = []
        missing = []
        stale = []
        now = time.monotonic()
        with self._lock:
            positions_by_name = dict(self._joint_positions_by_name)
            timestamps_by_name = dict(self._joint_position_timestamps_by_name)
            max_age_s = self._initial_pose_sync_state_max_age_s
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None:
                raise ValueError(f"unknown action key: {action_key}")
            if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                continue
            publisher_key = f"leader_{publish_key}"
            publisher = self._command_publishers.get(publisher_key)
            if publisher is None:
                raise RuntimeError(f"publisher unavailable for action key: {action_key}")
            joint_names = list(self._command_joint_names.get(publisher_key, []))
            group_missing = [
                name
                for name in joint_names
                if name not in positions_by_name or name not in timestamps_by_name
            ]
            missing.extend(group_missing)
            if group_missing:
                continue
            group_stale = [
                (name, max(0.0, now - timestamps_by_name[name]))
                for name in joint_names
                if max(0.0, now - timestamps_by_name[name]) > max_age_s
            ]
            stale.extend(group_stale)
            if group_stale:
                continue
            planned.append(
                (
                    publisher,
                    joint_names,
                    np.asarray(
                        [positions_by_name[name] for name in joint_names],
                        dtype=np.float64,
                    ),
                )
            )
        if missing:
            missing_names = ", ".join(sorted(set(missing)))
            raise RuntimeError(f"current joint state unavailable: {missing_names}")
        if stale:
            stale_by_name = {name: age for name, age in stale}
            stale_details = ", ".join(
                f"{name}={age:.3f}s"
                for name, age in sorted(stale_by_name.items())
            )
            raise RuntimeError(
                "current joint state stale: "
                f"{stale_details} (max {max_age_s:.3f}s)"
            )
        return planned

    def _publish_position_segments(
        self,
        segments: list[tuple[ROS2Publisher, list[str], np.ndarray]],
        duration_s: float,
    ) -> None:
        for publisher, joint_names, values in segments:
            self._publish_joint_trajectory(
                publisher,
                joint_names,
                values,
                time_from_start_s=duration_s,
            )

    def build_action_preview(
        self,
        action: np.ndarray,
        action_keys: Optional[list[str]] = None,
    ) -> tuple[list[str], np.ndarray]:
        """Build a single joint trajectory point for preview-only consumers."""
        keys = list(action_keys) if action_keys else self._action_keys
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        joint_names: list[str] = []
        positions: list[float] = []
        offset = 0
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None:
                continue
            msg_type = cfg["msg_type"]
            width = 3 if msg_type == "geometry_msgs/msg/Twist" else len(cfg["joint_names"])
            segment = values[offset:offset + width]
            offset += width
            if msg_type == "geometry_msgs/msg/Twist":
                continue
            names = list(cfg.get("joint_names", []))
            joint_names.extend(names)
            positions.extend(float(v) for v in segment[:len(names)])
        return joint_names, np.asarray(positions, dtype=np.float64)

    def publish_action_preview(
        self,
        action: np.ndarray,
        action_keys: Optional[list[str]] = None,
    ) -> None:
        """Publish preview-only action data for the 3D viewer."""
        if self._preview_publisher is None:
            return
        joint_names, positions = self.build_action_preview(action, action_keys)
        if not joint_names or len(joint_names) != len(positions):
            return
        self._publish_joint_trajectory(self._preview_publisher, joint_names, positions)

    def _resolve_action_key(self, action_key: str) -> str:
        if action_key in self._action_groups:
            return action_key
        if action_key == "odometry" and "mobile" in self._action_groups:
            return "mobile"
        return action_key

    def _validated_command_keys(self, action_keys: Optional[list[str]]) -> list[str]:
        keys = list(self._action_keys if action_keys is None else action_keys)
        resolved = [self._resolve_action_key(key) for key in keys]
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("command action keys must be non-empty and unique")
        unknown = set(resolved) - set(self._action_groups)
        if unknown:
            raise ValueError(f"unknown action key: {sorted(unknown)}")
        disabled = set(resolved) - set(self._action_keys)
        if disabled:
            raise ValueError(f"action keys are not enabled for this session: {sorted(disabled)}")
        return resolved

    def _publish_twist(self, publisher: ROS2Publisher, values: np.ndarray) -> None:
        Vector3 = get_message_class("geometry_msgs/msg/Vector3")
        linear_x = float(values[0]) if len(values) > 0 else 0.0
        linear_y = float(values[1]) if len(values) > 1 else 0.0
        angular_z = float(values[2]) if len(values) > 2 else 0.0
        filtered_linear_x = _deadband(linear_x, self._cmd_vel_linear_deadband)
        filtered_linear_y = _deadband(linear_y, self._cmd_vel_linear_deadband)
        filtered_angular_z = _deadband(angular_z, self._cmd_vel_angular_deadband)

        linear = Vector3(
            x=filtered_linear_x,
            y=filtered_linear_y,
            z=0.0,
        )
        angular = Vector3(
            x=0.0,
            y=0.0,
            z=filtered_angular_z,
        )
        publisher.publish(linear=linear, angular=angular)

    def _publish_joint_trajectory(
        self,
        publisher: ROS2Publisher,
        joint_names: list[str],
        values: np.ndarray,
        time_from_start_s: float = 0.0,
    ) -> None:
        Header = get_message_class("std_msgs/msg/Header")
        Time = get_message_class("builtin_interfaces/msg/Time")
        Duration = get_message_class("builtin_interfaces/msg/Duration")
        JointTrajectoryPoint = get_message_class(
            "trajectory_msgs/msg/JointTrajectoryPoint"
        )
        duration_s = float(time_from_start_s)
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("time_from_start_s must be finite and non-negative")
        duration_sec = int(math.floor(duration_s))
        duration_nanosec = int(round((duration_s - duration_sec) * 1_000_000_000))
        if duration_nanosec >= 1_000_000_000:
            duration_sec += 1
            duration_nanosec -= 1_000_000_000
        point = JointTrajectoryPoint(
            positions=np.asarray(values, dtype=np.float64),
            velocities=np.zeros(0, dtype=np.float64),
            accelerations=np.zeros(0, dtype=np.float64),
            effort=np.zeros(0, dtype=np.float64),
            time_from_start=Duration(sec=duration_sec, nanosec=duration_nanosec),
        )
        publisher.publish(
            header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
            joint_names=list(joint_names),
            points=[point],
        )

    # ------------------------------------------------------------------ #
    # Task instruction
    # ------------------------------------------------------------------ #

    def set_task_instruction(self, instruction: str):
        self._task_instruction = instruction

    @property
    def task_instruction(self) -> str:
        return self._task_instruction

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def get_observation(
        self,
        resize: Optional[tuple[int, int]] = None,
        format: str = "bgr",
    ) -> Optional[dict]:
        """Get full observation for inference.

        Returns:
            Dict with images, joint_positions, task_instruction.
            None if sync_check is enabled and data is out of sync.
        """
        if self._sync_check and not self._check_sync():
            return None
        return {
            "images": self.get_images(resize=resize, format=format),
            "joint_positions": self.get_joint_positions(),
            "task_instruction": self._task_instruction,
        }

    def _check_sync(self) -> bool:
        """Check if image and joint timestamps are within threshold."""
        threshold_s = self._sync_threshold_ms / 1000.0
        with self._lock:
            if not self._image_timestamps or not self._joint_timestamps:
                return False
            img_times = list(self._image_timestamps.values())
            jnt_times = list(self._joint_timestamps.values())

        latest_img = max(img_times) if img_times else 0
        latest_jnt = max(jnt_times) if jnt_times else 0
        return abs(latest_img - latest_jnt) < threshold_s

    # ------------------------------------------------------------------ #
    # Readiness / waiting
    # ------------------------------------------------------------------ #

    def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Wait until at least one frame from all sensors is received."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._all_ready():
                logger.info("All sensors ready")
                return True
            time.sleep(0.1)
        # Log what's missing
        missing = self._get_missing()
        logger.warning(f"Timeout waiting for sensors. Missing: {missing}")
        return False

    def wait_for_image(self, camera_name: str, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_image_ready(camera_name):
                return True
            time.sleep(0.1)
        return False

    def wait_for_joint(self, group_name: str, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_joint_ready(group_name):
                return True
            time.sleep(0.1)
        return False

    def _all_ready(self) -> bool:
        with self._lock:
            for cam in self._config.get("cameras", {}):
                if cam not in self._images:
                    return False
            for group in self._config.get("joint_groups", {}):
                if group not in self._joint_positions:
                    return False
            return True

    def _get_missing(self) -> list[str]:
        missing = []
        with self._lock:
            for cam in self._config.get("cameras", {}):
                if cam not in self._images:
                    missing.append(f"camera:{cam}")
            for group in self._config.get("joint_groups", {}):
                if group not in self._joint_positions:
                    missing.append(f"joint:{group}")
        return missing

    # ------------------------------------------------------------------ #
    # Info / diagnostics
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """Get current status of all subscriptions."""
        with self._lock:
            return {
                "robot_type": self._robot_type,
                "cameras": {
                    name: {
                        "ready": name in self._images,
                        "shape": self._images[name].shape if name in self._images else None,
                        "timestamp": self._image_timestamps.get(name),
                    }
                    for name in self._config.get("cameras", {})
                },
                "joint_groups": {
                    name: {
                        "ready": name in self._joint_positions,
                        "dof": len(self._joint_positions[name]) if name in self._joint_positions else 0,
                        "timestamp": self._joint_timestamps.get(name),
                    }
                    for name in self._config.get("joint_groups", {})
                },
                "sensors": {
                    name: {
                        "ready": name in self._sensors,
                        "timestamp": self._sensor_timestamps.get(name),
                    }
                    for name in self._config.get("sensors", {})
                },
            }

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def close(self):
        """Close all subscriptions and command publishers."""
        if hasattr(self, '_closed') and self._closed:
            return
        self._closed = True
        for sub in self._subscribers:
            try:
                sub.close()
            except Exception as e:
                logger.debug(f"Error closing subscriber: {e}")
        self._subscribers.clear()
        for pub in self._command_publishers.values():
            try:
                pub.close()
            except Exception as e:
                logger.debug(f"Error closing command publisher: {e}")
        self._command_publishers.clear()
        if self._preview_publisher is not None:
            try:
                self._preview_publisher.close()
            except Exception as e:
                logger.debug(f"Error closing preview publisher: {e}")
            self._preview_publisher = None
        with self._lock:
            self._observation_history = None
        logger.info("RobotClient closed")

    def __del__(self):
        self.close()
