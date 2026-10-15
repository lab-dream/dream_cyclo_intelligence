#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot engine loading helpers (LoadingMixin).

Extracted from ``engine.py`` to keep the core ``LeRobotEngine`` class
focused on the ``InferenceEngine`` API. Mixed into the engine via
multiple inheritance; bind-mounted into the policy container as part
of the ``/app/lerobot_engine/`` package.

Owns:
- ``_resolve_model_dir``: auto-descend lerobot training-output roots.
- ``_load_policy_assets``: load weights + stored pre/post processors.
- ``_infer_image_resize``: read per-input-image shape hints off the policy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple

import draccus
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy

from .image_preprocessing import infer_image_resize_targets

logger = logging.getLogger("lerobot_engine")


class LoadingMixin:
    """Policy load helpers — weights, processors, resize hint."""

    @staticmethod
    def _resolve_model_dir(model_path: str) -> str:
        """Auto-descend lerobot training-output roots.

        Users frequently paste the training-output root which contains
        ``pretrained_model/`` next to ``training_state/``. Strip that
        wrapper if needed so ``from_pretrained`` finds ``config.json``.
        """
        selected = (model_path or "").strip()
        if not selected:
            raise ValueError("Select a LeRobot checkpoint directory containing config.json.")
        root = Path(selected)
        nested = root / "pretrained_model"
        if not (root / "config.json").is_file() and (nested / "config.json").is_file():
            logger.info("Descending into pretrained_model: %s", nested)
            return str(nested)
        if root.is_dir() and not (root / "config.json").is_file():
            candidates = sorted(
                str(path.parent) for path in root.rglob("config.json")
                if path.is_file() and path.parent.name == "pretrained_model"
            )
            hint = " Available checkpoints: " + "; ".join(candidates[:10]) if candidates else ""
            raise FileNotFoundError(
                f"LeRobot config.json not found in {root}. Select a specific checkpoint "
                f"directory containing config.json, rather than the model collection root.{hint}"
            )
        return str(root)

    @staticmethod
    def _load_policy_assets(
        model_path: str, device: torch.device
    ) -> tuple[PreTrainedPolicy, Any, Any]:
        """Load policy weights + saved pre/post processors."""
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)
                policy_type = saved_config.get("type", "act")
        else:
            # ACT was the original default; fall back to it for
            # checkpoints saved before ``type`` started being recorded.
            policy_type = "act"

        logger.info("Policy type: %s", policy_type)
        PolicyClass = get_policy_class(policy_type)

        # FastWAM's text encoder must stay on the CPU. Its default config can
        # auto-select CUDA inside ``from_pretrained`` and exhaust VRAM before
        # the offload hook runs, so pin only this policy's initial load to CPU.
        if policy_type == "fastwam":
            policy_config = PreTrainedConfig.from_pretrained(model_path)
            policy_config.device = "cpu"
            policy = PolicyClass.from_pretrained(model_path, config=policy_config)
        elif policy_type == "diffusion" and config_path.exists():
            # Dream adds training representation/checkpointing/dtype fields.
            # Older Cyclo images need only the model fields for inference;
            # the saved `state` is handled separately by the EEF bridge.
            supported = {field.name for field in fields(PolicyClass.config_class)}
            for name in ("state", "hand", "gradient_checkpointing", "dtype"):
                if name not in supported:
                    saved_config.pop(name, None)
            policy_config = draccus.decode(PreTrainedConfig, saved_config)
            policy_config.device = str(device)
            policy = PolicyClass.from_pretrained(model_path, config=policy_config, strict=True)
        else:
            policy = PolicyClass.from_pretrained(model_path)

        # MolmoAct2 errors out unless the action mode is set. We run the
        # continuous (flow matching) head; a checkpoint that names one keeps it.
        if policy_type == "molmoact2" and not getattr(
            policy.config, "inference_action_mode", None
        ):
            policy.config.inference_action_mode = "continuous"

        if policy_type == "fastwam":
            policy = policy.eval()
            logger.info("FastWAM weights loaded on CPU for selective offload")
        else:
            policy = policy.to(device).eval()
            logger.info("Policy weights loaded on %s", device)

        # Stored processor pipelines include the dataset-time normalizer
        # stats and image transforms so we don't re-derive (and de-sync)
        # them. Falling through to the default factory here would wipe
        # those stats and produce garbage actions.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
            },
        )
        logger.info("Pre/post processors loaded")
        return policy, preprocessor, postprocessor

    def _configure_relative_eef(self, model_path: str, robot_type: str, request=None) -> None:
        """Select the inference representation from the saved checkpoint, never its width."""
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            return
        saved = json.loads(config_path.read_text())
        ik_backend = str(getattr(request, "ik_backend", "") or "cyclo_control").strip().lower()
        if ik_backend == "dls" and (saved.get("type") != "diffusion" or saved.get("state") != "relative"):
            raise ValueError("DLS IK requires a Diffusion checkpoint with state='relative'.")
        if saved.get("type") != "diffusion":
            return
        state = saved.get("state")
        if state is None:
            logger.info("Joint-action checkpoint; using the existing inference flow")
            return
        if state != "relative":
            raise ValueError(f"Unsupported Diffusion state representation: {state!r}.")
        hand = saved.get("hand", "both")
        if hand not in {"both", "left", "right"}:
            raise ValueError(f"Unsupported relative EEF hand: {hand!r}.")
        width = 20 if hand == "both" else 10
        for key, features in (("observation.state", saved["input_features"]), ("action", saved["output_features"])):
            if features[key]["shape"] != [width]:
                raise ValueError(f"Relative EEF hand={hand} requires a {width}-dimensional {key}.")
        from .relative_eef import RelativeEEFAdapter

        self._relative_eef = RelativeEEFAdapter(
            model_path, self._robot, robot_type,
            ik_backend=getattr(request, "ik_backend", "") or "cyclo_control",
            ik_model_path=getattr(request, "ik_model_path", ""), device=str(getattr(self, "_device", None) or "cpu"),
        )
        self._action_keys = list(self._relative_eef.action_keys)

    def _infer_image_resize(self, policy: PreTrainedPolicy) -> Dict[str, Tuple[int, int]]:
        """Best-effort per-policy-key target ``(W, H)`` from config.

        Many lerobot policies advertise the expected image shape under
        ``input_features['observation.images.<cam>'].shape = (C, H, W)``.
        Pre-resizing on the host keeps mixed camera shapes aligned with the
        dataset metadata. Missing keys mean: leave that camera at native size.
        """
        try:
            features = getattr(policy.config, "input_features", {}) or {}
            return infer_image_resize_targets(features)
        except Exception:
            pass
        return {}
