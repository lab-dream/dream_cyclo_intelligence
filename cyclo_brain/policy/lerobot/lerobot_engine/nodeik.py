"""Thin bridge to the pinned NodeIK worker; keep its CNF/encoder unchanged."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ._vendor import dream
from ._vendor.dream import ai_worker, math
from .image_preprocessing import prepare_policy_image

logger = logging.getLogger("lerobot_engine")


class NodeIK:
    def __init__(self, model_path, urdf_path, active_names, device):
        # The worker imports Dream's FK/math under lerobot.state_free. These
        # unchanged files are already vendored here for relative EEF policies.
        sys.modules.setdefault("lerobot.state_free", dream)
        sys.modules.setdefault("lerobot.state_free.ai_worker", ai_worker)
        sys.modules.setdefault("lerobot.state_free.math", math)
        source = os.environ.get("NODEIK_SOURCE_PATH", "/opt/cyclo_sdk/nodeik")
        if not Path(source).is_dir():
            source = str(Path(__file__).resolve().parents[3] / "sdk/nodeik")
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            from nodeik.ai_worker import encode_poses, fk_poses, load_model, predict
        except ImportError as error:
            raise RuntimeError("Initialize sdk/nodeik submodule and install torchdiffeq==0.2.5.") from error

        self.model_path = self.resolve_model_path(model_path)
        self.device = torch.device(device)
        self.model, self.robot, self.vision, saved = load_model(self.model_path, self.device, urdf_path)
        self.names = saved["joint_names"]
        if self.names != active_names:
            raise ValueError("NodeIK checkpoint joints do not match relative EEF arm/lift joints.")
        self.encode_poses, self.fk_poses, self.predict = encode_poses, fk_poses, predict
        mapping = [self.robot._joint_q_index(name) for name in self.names]
        self.lower = self.robot.model.lowerPositionLimit[mapping]
        self.upper = self.robot.model.upperPositionLimit[mapping]
        self.visual_features = None
        self.image_keys = list(self.vision.keys) if self.vision else []
        logger.info("NodeIK: %s (step=%s, mode=%s, latent=0)", self.model_path,
                    saved.get("step"), "vision" if self.vision else "pose")

    @staticmethod
    def resolve_model_path(model_path):
        root = Path(str(model_path or "").strip()).expanduser()
        if not str(model_path or "").strip():
            raise ValueError("Select a NodeIK model folder or .pt checkpoint.")
        if root.is_file() and root.suffix == ".pt":
            return root
        # A compare run has both pose/ and vision/. Prefer pose IK when the
        # entire run is selected; selecting vision/ explicitly enables RGB IK.
        for candidate in (root / "best.pt", root / "pose/best.pt"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"NodeIK best.pt not found in {root}.")

    def set_images(self, images, camera_config):
        if self.vision is None:
            return
        self.visual_features = None
        tensors = {}
        for key in self.image_keys:
            camera = key.removeprefix("observation.images.rgb.")
            if camera not in images:
                raise ValueError(f"NodeIK vision requires camera {camera}.")
            rgb = prepare_policy_image(images[camera],
                                       rotation_deg=camera_config.get(camera, {}).get("rotation_deg", 0))
            tensors[key] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.visual_features = self.vision(tensors)

    def solve(self, seed, left, right):
        targets = np.stack([left, right])[None]
        condition = torch.from_numpy(self.encode_poses(targets)).to(self.device)
        if self.vision is not None:
            if self.visual_features is None:
                raise RuntimeError("Capture NodeIK camera observations before solving IK.")
            condition = torch.cat([condition, self.visual_features], dim=-1)
        # Keep one deterministic latent across all waypoints and feedback
        # chunks, following NodeIK's trajectory inference convention.
        joints = self.predict(self.model, condition, latent="0")[0, 0]
        limited = np.clip(joints, self.lower, self.upper)
        if not np.array_equal(joints, limited):
            logger.debug("Clipped NodeIK output to the training URDF joint limits")
        actual = self.fk_poses(self.robot, limited)[0]
        position = np.linalg.norm(actual[:, :3, 3] - targets[0, :, :3, 3], axis=-1).max()
        rotation = np.swapaxes(targets[0, :, :3, :3], -1, -2) @ actual[:, :3, :3]
        angle = np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1)).max()
        return dict(zip(self.names, limited.tolist(), strict=True)), float(position), float(angle)
