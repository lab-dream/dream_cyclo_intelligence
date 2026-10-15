"""Dataset images + predicted IK state, using the existing inference/UMI path."""

from __future__ import annotations

import base64
import json
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from action_chunk_processing.spline import quintic_spline_multi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robot_client import RobotClient
from robot_client.robot_client import derive_robot_config, robot_schema

from ._vendor.umi.pose_util import pose_to_mat
from ._vendor.umi.real_inference_util import get_real_umi_action
from .image_preprocessing import apply_rotation


def configured_initial_state(robot_type, names):
    """Read the configured initial pose by name, including base state values."""
    if robot_type != "ffw_sg2_rev1":
        raise ValueError("The configured initial state requires AI Worker ffw_sg2_rev1.")
    sdk_path = Path(os.environ.get("DLS_IK_SDK_PATH", "/opt/cyclo_sdk/dls_ik"))
    if not sdk_path.is_dir():
        sdk_path = Path(__file__).resolve().parents[3] / "sdk/dls_ik"
    path = sdk_path / "config/initial_state.json"
    document = json.loads(path.read_text())
    if document["robot_type"] != robot_type or len(set(document["names"])) != len(document["names"]):
        raise ValueError("Configured initial state has an invalid robot type or duplicate joint names.")
    positions = dict(zip(document["names"], document["state"], strict=True))
    missing = set(names) - positions.keys()
    if missing:
        raise ValueError(f"Configured initial state does not define dataset joints: {sorted(missing)}")
    state = np.asarray([positions[name] for name in names], dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Configured initial state must contain finite joint values.")
    return state


class DatasetSimulation(RobotClient):
    """Reuse RobotClient's named groups without creating any robot transport."""

    def __init__(self, robot_type, dataset_path, episode, frame, n_obs_steps, control_hz=100,
                 use_initial_frame_state=True):
        self._closed = True  # RobotClient's destructor must not close nonexistent transport.
        if not isinstance(use_initial_frame_state, bool):
            raise ValueError("use_initial_frame_state must be a boolean.")
        self.use_initial_frame_state = use_initial_frame_state
        self.initial_state_source = "dataset_frame" if use_initial_frame_state else "configured_initial_state"
        self.dataset_root = Path(dataset_path)
        info = json.loads((self.dataset_root / "meta/info.json").read_text())
        if info.get("robot_type") != robot_type:
            raise ValueError("Dataset robot_type does not match the selected robot.")
        if not 0 <= episode < info["total_episodes"] or frame < 0:
            raise ValueError("Episode/frame is outside the dataset.")
        section = robot_schema.load_robot_section(robot_type)
        self._config = derive_robot_config(section)
        self._action_groups = robot_schema.get_action_groups(section)
        self.names = info["features"]["observation.state"]["names"]
        self.fps = info["fps"]
        self.control_hz = float(control_hz)
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be positive and finite.")
        self.dataset = LeRobotDataset(self.dataset_root.name, root=self.dataset_root,
                                     episodes=[episode], video_backend="pyav")
        self.length = len(self.dataset)
        if frame >= self.length:
            raise ValueError("Initial frame is outside the selected episode.")
        self.episode = episode
        self.frame = frame
        self.history = deque(maxlen=n_obs_steps)
        self._samples = {}
        home = None if use_initial_frame_state else configured_initial_state(robot_type, self.names)
        for index in range(frame - n_obs_steps + 1, frame + 1):
            index = max(0, index)
            state = self.sample(index)["observation.state"].numpy() if home is None else home
            self.history.append((index, state.copy()))
        self.current = self.history[-1]
        self.relative_action = None
        self.relative_observation = None

    def sample(self, frame):
        if frame not in self._samples:
            self._samples[frame] = self.dataset[frame]
        return self._samples[frame]

    def get_images(self, format="rgb"):
        sample = self.sample(self.current[0])
        images = {}
        for name, cfg in self._config["cameras"].items():
            key = f"observation.images.rgb.{name}"
            if key not in sample:
                continue
            image = np.rint(sample[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Undo the recorder's rotation; the normal inference preprocessor applies it once.
            images[name] = apply_rotation(image, -int(cfg.get("rotation_deg", 0))).copy()
        return images

    def get_joint_positions(self, group=None):
        measured = dict(zip(self.names, self.current[1], strict=True))
        groups = {
            key: np.asarray([measured[name] for name in cfg["joint_names"]])
            for key, cfg in self._config["joint_groups"].items()
        }
        return groups if group is None else groups[group]

    def wait_for_ready(self, timeout=10.0):
        return True

    def get_odom(self):
        measured = dict(zip(self.names, self.current[1], strict=True))
        return {"linear_velocity": np.array([measured["linear_x"], measured["linear_y"], 0.]),
                "angular_velocity": np.array([0., 0., measured["angular_z"]])}

    def close(self):
        getattr(self, "_samples", {}).clear()

    def build_observation(self, engine, instruction):
        observations = []
        try:
            for self.current in self.history:
                obs = engine._build_observation(instruction)
                if "success" in obs:
                    return obs
                observations.append(obs)
        finally:
            self.current = self.history[-1]
        batch = dict(observations[-1])
        for key in engine._policy.config.input_features:
            batch[key] = torch.stack([obs[key] for obs in observations], dim=1)
        if engine._relative_eef is not None:
            batch = engine._relative_eef.encode_observation(batch)
            self.relative_observation = batch["observation.state"][0].detach().cpu().numpy().copy()
        return batch

    def frame_state(self, engine):
        images = {}
        sample = self.sample(self.current[0])
        for camera, key in engine._cameras.items():
            rgb = np.rint(sample[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise RuntimeError(f"Cannot encode dataset image: {camera}")
            images[camera] = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
        return {"frame": self.current[0], "joint_state": {
            "name": self.names, "position": self.current[1].tolist()}, "images": images}

    def advance(self, commands, engine):
        frames = []
        interpolate_q = engine._relative_eef is not None and engine._relative_eef.ik_backend in {"nodeik", "dls"}
        for command in commands[:self.length - 1 - self.frame]:
            names, values = self.build_action_preview(command, engine._action_keys)
            state = self.history[-1][1].copy()
            for name, value in zip(names, values, strict=True):
                state[self.names.index(name)] = value
            if interpolate_q:
                # Only endpoint states enter policy history; images follow dataset time.
                duration = 1.0 / self.fps
                times = np.append(np.arange(1.0 / self.control_hz, duration,
                                            1.0 / self.control_hz), duration)
                previous_t = 0.0
                zero = np.zeros_like(state)
                for t in times:
                    intermediate = quintic_spline_multi(
                        t, 0.0, duration, self.history[-1][1], zero, zero,
                        state, zero, zero)[0]
                    endpoint = t == duration
                    self.current = (self.frame + int(endpoint), state if endpoint else intermediate)
                    frames.append({**self.frame_state(engine), "dt_s": float(t - previous_t)})
                    previous_t = t
            self.frame += 1
            self.history.append((self.frame, state))
            self.current = self.history[-1]
            if not interpolate_q:
                frames.append(self.frame_state(engine))
        # Keep only the current history's images, not a whole decoded episode.
        keep = {frame for frame, _ in self.history}
        self._samples = {frame: sample for frame, sample in self._samples.items() if frame in keep}
        return frames


def dataset_simulation_command(engine, request):
    """Called by EngineWorker under its command lock; Main never publishes these actions."""
    if request.command == 5:
        if getattr(engine, "_dataset_simulation", None) is not None:
            engine.cleanup()
        return {"unloaded": True}
    if request.command == 3:
        if engine.is_ready:
            raise RuntimeError("Unload the current inference session before loading dataset simulation.")
        model = Path(engine._resolve_model_dir(request.model_path))
        config = json.loads((model / "config.json").read_text())
        if config.get("type") != "diffusion":
            raise ValueError("Dataset simulation requires a Diffusion policy.")
        settings = json.loads(request.simulation_config)
        robot = DatasetSimulation(request.robot_type, settings["dataset_path"],
                                  int(settings.get("episode", 0)), int(settings.get("frame", 0)),
                                  config["n_obs_steps"], settings.get("control_hz", 100),
                                  use_initial_frame_state=settings.get("use_initial_frame_state", True))
        loaded = engine.load_policy(SimpleNamespace(**vars(request), dataset_robot=robot))
        if not loaded["success"]:
            robot.close()
            raise RuntimeError(loaded["message"])
        try:
            obs = engine._build_policy_observation(request.task_instruction)
            if "success" in obs:
                raise RuntimeError(obs["message"])
            frames = [robot.frame_state(engine)]
        except Exception:
            engine.cleanup()
            raise
    elif request.command == 4:
        robot = getattr(engine, "_dataset_simulation", None)
        if robot is None:
            raise RuntimeError("Load dataset simulation first.")
        if robot.frame >= robot.length - 1:
            raise RuntimeError("Dataset episode finished. Unload and load to reset.")
        result = engine.get_action_chunk(request)
        if not result["success"]:
            raise RuntimeError(result["message"])
        commands = result["action_chunk"].reshape(result["chunk_size"], result["action_dim"])
        frames = robot.advance(commands, engine)
    else:
        raise ValueError("Unknown dataset simulation command.")

    adapter = engine._relative_eef
    # Both histories use the same most recent observed TCP anchor as UMI decoding.
    anchors = []
    for side in range(2) if adapter is not None else ():
        pose = np.concatenate([adapter.env_obs[f"robot{side}_eef_pos"][-1],
                               adapter.env_obs[f"robot{side}_eef_rot_axis_angle"][-1]])
        anchors.append(pose_to_mat(pose).tolist())
    relative_action = robot.relative_action if request.command == 4 and adapter is not None else None
    relative_observation = robot.relative_observation if adapter is not None else None
    return {
        "episode": robot.episode, "frame": robot.frame, "length": robot.length,
        "fps": robot.fps, "ik_backend": adapter.ik_backend if adapter is not None else "",
        "arm_selection": getattr(engine, "_arm_selection", "both"),
        "action_keys": list(engine._action_keys),
        "use_initial_frame_state": robot.use_initial_frame_state,
        "initial_state_source": robot.initial_state_source,
        "action_request_mode": "chunkrl" if getattr(engine, "_chunkrl", None) is not None else str(getattr(request, "action_request_mode", "") or "sync"),
        "chunkrl_decisions": engine._chunkrl.last_decisions if getattr(engine, "_chunkrl", None) is not None else [],
        "finished": robot.frame >= robot.length - 1, "frames": frames,
        "observation_relative": relative_observation.tolist() if relative_observation is not None else [],
        "action_relative": relative_action.tolist() if relative_action is not None else [],
        "observation_poses": get_real_umi_action(adapter.bimanual_relative(relative_observation), adapter.env_obs,
                                                action_pose_repr="relative").tolist() if adapter is not None else [],
        "action_poses": get_real_umi_action(adapter.bimanual_relative(relative_action), adapter.env_obs,
                                           action_pose_repr="relative").tolist() if relative_action is not None else [],
        "anchor_poses": anchors, "hardware_publish": False,
    }
