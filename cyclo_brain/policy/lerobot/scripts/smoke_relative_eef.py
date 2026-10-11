#!/usr/bin/env python3
"""Replay dataset sensors through the real EngineWorker and IK, without hardware."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / path)
    for path in (
        "cyclo_brain/policy/lerobot/lerobot/src",
        "cyclo_brain/policy/lerobot",
        "cyclo_brain/policy/common/runtime",
        "cyclo_brain/sdk/robot_client",
        "cyclo_brain/sdk/zenoh_ros2_sdk",
        "shared",
        "shared/shared/robot_configs",
    )
]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from engine_process.protocol import (  # noqa: E402
    CMD_GET_ACTION,
    CMD_LOAD_POLICY,
    CMD_UNLOAD_POLICY,
    EngineCommandRequest,
)
from engine_process.worker import EngineWorker  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot_engine import (
    LeRobotEngine,  # noqa: E402
    io_mapping,  # noqa: E402
)
from robot_client import RobotClient  # noqa: E402
from robot_client.robot_client import derive_robot_config  # noqa: E402
from shared.robot_configs import schema  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_path", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    dataset = LeRobotDataset(
        args.dataset_root.name,
        root=args.dataset_root,
        episodes=[0],
        video_backend="pyav",
    )
    samples = [dataset[index] for index in (50, 51)]
    state_names = info["features"]["observation.state"]["names"]

    class ReplayRobot(RobotClient):
        """Keep RobotClient's command slicing, replacing its transport and sensors."""

        def __init__(self, robot_type: str) -> None:
            section = schema.load_robot_section(robot_type)
            self._config = derive_robot_config(section)
            self._action_groups = schema.get_action_groups(section)
            self._action_keys = list(self._action_groups)
            self._command_publishers = {
                f"leader_{key}": key for key in self._action_groups
            }
            self._command_joint_names = {
                f"leader_{key}": cfg["joint_names"]
                for key, cfg in self._action_groups.items()
            }
            self.index = 0
            self.sent = []

        def get_images(self, format="rgb"):
            sample = samples[self.index % len(samples)]
            images = {}
            for camera, cfg in self._config["cameras"].items():
                image = (
                    sample[f"observation.images.rgb.{camera}"].permute(1, 2, 0).numpy()
                )
                # Dataset wrist frames already carry the recorder's rotation.
                images[camera] = np.rot90(
                    np.rint(image * 255).astype(np.uint8),
                    -int(cfg.get("rotation_deg", 0)) // 90,
                ).copy()
            return images

        def get_joint_positions(self, group=None):
            sample = samples[self.index % len(samples)]
            self.index += 1
            measured = dict(
                zip(state_names, sample["observation.state"].numpy(), strict=True)
            )
            groups = {
                key: np.asarray([measured[name] for name in cfg["joint_names"]])
                for key, cfg in self._config["joint_groups"].items()
            }
            return groups if group is None else groups[group]

        def get_odom(self):
            return {"linear_velocity": [0, 0, 0], "angular_velocity": [0, 0, 0]}

        def wait_for_ready(self, timeout=10.0):
            return True

        def _publish_joint_trajectory(
            self, publisher, joint_names, values, time_from_start_s=0.0
        ):
            self.sent.append((publisher, joint_names, np.asarray(values).copy()))

        def close(self):
            pass

    io_mapping.RobotClient = ReplayRobot
    engine = LeRobotEngine()
    worker = EngineWorker(engine)
    request = EngineCommandRequest(
        command=CMD_LOAD_POLICY,
        model_path=str(args.policy_path.resolve()),
        robot_type="ffw_sg2_rev1",
        inference_hz=30,
    )
    loaded = worker.handle(request)
    assert loaded.success, loaded.message
    assert loaded.action_keys == ["arm_left", "arm_right", "lift"], loaded

    # Compare the copied UMI observation conversion with Dream's training transform.
    adapter = engine._relative_eef
    joints = np.stack([sample["observation.state"].numpy() for sample in samples])
    expected = adapter.transform({"observation.state": joints, "action": joints})[
        "observation.state"
    ]
    batch = engine._build_policy_observation("")
    torch.testing.assert_close(batch["observation.state"][0].cpu(), expected)

    # A zero relative pose must keep the measured arms and the shared lift.
    from lerobot_engine._vendor.dream.math import normalize

    identity = expected[-1].numpy().copy()
    grippers = np.asarray(
        [adapter.measured[f"gripper_{side}_joint1"] for side in ("l", "r")]
    )
    identity[[9, 19]] = normalize(grippers, *adapter.transform.gripper_bounds["action"])
    restored = adapter.decode_action(identity[None])[0]
    for index, side in enumerate(("left", "right")):
        np.testing.assert_allclose(
            restored[index * 8 : index * 8 + 7],
            [adapter.measured[name] for name in adapter._arm_names[side]],
            atol=1e-6,
        )
    np.testing.assert_allclose(restored[16], adapter.measured["lift_joint"], atol=1e-6)
    np.testing.assert_allclose(restored[[7, 15]], grippers, atol=1e-6)

    chunks = []
    for index in range(args.requests):
        response = worker.handle(
            EngineCommandRequest(command=CMD_GET_ACTION, seq_id=index + 1)
        )
        assert response.success, response.message
        chunk = np.asarray(response.action_list).reshape(
            response.chunk_size, response.action_dim
        )
        assert chunk.shape == (engine._policy.config.n_action_steps, 17), chunk.shape
        assert np.isfinite(chunk).all()
        for action in chunk:
            engine._robot.publish_action(action, loaded.action_keys)
        chunks.append(list(chunk.shape))
    sent = engine._robot.sent
    assert len(sent) == sum(shape[0] for shape in chunks) * 3
    assert {topic for topic, _, _ in sent} == {"arm_left", "arm_right", "lift"}
    assert all(len(names) == len(values) == (1 if topic == "lift" else 8) for topic, names, values in sent)
    cached = worker.handle(request)
    assert cached.success and "cached" in cached.message, cached
    after_reload = worker.handle(EngineCommandRequest(command=CMD_GET_ACTION))
    assert after_reload.success, after_reload.message
    unloaded = worker.handle(EngineCommandRequest(command=CMD_UNLOAD_POLICY))
    assert unloaded.success and not engine.is_ready
    report = {
        "success": True,
        "policy_path": str(args.policy_path.resolve()),
        "training_steps": 100,
        "observation_matches_training": True,
        "identity_pose_ik_roundtrip": True,
        "action_keys": loaded.action_keys,
        "chunks": chunks,
        "captured_arm_commands": sum(topic != "lift" for topic, _, _ in sent),
        "captured_lift_commands": sum(topic == "lift" for topic, _, _ in sent),
        "cached_reload": True,
        "unload": True,
        "hardware_publish": False,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
