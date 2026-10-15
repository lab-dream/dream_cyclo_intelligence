#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot engine I/O mapping helpers (IoMappingMixin).

Extracted from ``engine.py`` to keep the core ``LeRobotEngine`` class
focused on the ``InferenceEngine`` API. Mixed into the engine via
multiple inheritance; bind-mounted into the policy container as part
of the ``/app/lerobot_engine/`` package.

Owns:
- ``_init_robot``: create RobotClient + resolve camera / state mappings.
- ``_teardown_robot``: release the RobotClient.
- ``_policy_image_keys``: read the policy's expected image input keys.
"""

from __future__ import annotations

import logging
import re
import numpy as np
from typing import Dict, Iterable

from .constants import IMAGE_KEY_PREFIX as _IMAGE_KEY_PREFIX

from robot_client import RobotClient


logger = logging.getLogger("lerobot_engine")


_CAMERA_SEMANTIC_RE = re.compile(
    r"^cam_(?P<a>left|right|head|wrist)_(?P<b>left|right|head|wrist)$"
)


class IoMappingMixin:
    """Robot wiring — camera / state modality resolution and teardown."""

    def _configure_arm_selection(self, request) -> None:
        """Project existing full model/IK output through RobotClient group order."""
        self._action_indices = None
        self._arm_selection = "both"
        if self._policy.config.type != "diffusion":
            return
        selection = str(getattr(request, "arm_selection", "both") or "right").strip().lower()
        if selection not in {"both", "left", "right", "bimanual_right"}:
            raise ValueError("arm_selection must be both, left, right, or bimanual_right")
        adapter = self._relative_eef
        hand = getattr(adapter, "policy_hand", "both")
        if hand != "both":
            if selection == "both":
                selection = hand
            if selection != hand:
                raise ValueError(f"The hand={hand} checkpoint requires arm_selection={hand}.")
        if selection == "bimanual_right" and (adapter is None or adapter.ik_backend != "dls"):
            raise ValueError("bimanual_right requires a relative EEF Diffusion checkpoint with DLS IK")
        if adapter is not None and adapter.ik_backend == "dls" and selection in {"left", "right"}:
            image_keys = self._policy_image_keys()
            wrist_keys = (self._camera_policy_key_candidates("cam_left_wrist")
                          | self._camera_policy_key_candidates("cam_right_wrist"))
            if len(image_keys) == 1 and image_keys <= wrist_keys:
                image_key = next(iter(image_keys))
                if image_key not in self._camera_policy_key_candidates(f"cam_{selection}_wrist"):
                    raise ValueError(f"arm_selection={selection} requires a {selection}-wrist checkpoint; "
                                     f"policy expects {image_key}")
                # The extracted tape datasets store the recorder's rotated native frames.
                # Keep their pixels even when image_preprocessing.json was not transferred.
                config = self._policy.config
                image_shape = tuple(config.input_features[image_key].shape)
                if image_shape == (3, 424, 240):
                    resize = getattr(config, "resize_shape", None)
                    if ((resize is not None and tuple(resize) != image_shape[1:])
                            or getattr(config, "crop_shape", None) is not None):
                        raise ValueError("Native wrist DLS inference requires no image resize or crop in the checkpoint")
                    self._image_preprocessing = {"resize_mode": "native"}
                    logger.info("Native %s wrist observation: CHW=%s, recorder rotation only", selection, image_shape)
        command_selection = "right" if selection == "bimanual_right" else selection
        keys = list(self._action_keys)
        if not set(keys) & {"arm_left", "arm_right"}:
            return  # Existing single-arm robot groups do not need projection.
        selected = [key for key in keys if key in (
            {"arm_left", "arm_right"} if command_selection == "both" else {f"arm_{command_selection}"})]
        if not selected:
            raise ValueError(f"Robot has no command group for arm_selection={selection}")
        if adapter is not None:
            adapter.command_arms = tuple(key.removeprefix("arm_") for key in selected)
            # Shared-lift IK must publish its lift solution with either arm.
            if adapter.ik_backend != "dls" and "lift" in keys:
                selected.append("lift")
        indices = []
        offset = 0
        for key in keys:
            width = len(self._robot._action_groups[key]["joint_names"])
            if key in selected:
                indices.extend(range(offset, offset + width))
            offset += width
        self._full_action_dim = offset
        self._action_indices = indices
        self._action_keys = selected
        self._arm_selection = selection
        logger.info("Arm selection=%s command groups=%s", selection, selected)

    def _select_action_chunk(self, chunk):
        indices = getattr(self, "_action_indices", None)
        if indices is None:
            return chunk
        if chunk.ndim != 2 or chunk.shape[1] != self._full_action_dim or not np.isfinite(chunk).all():
            raise ValueError("Invalid action chunk for the configured command groups")
        return np.ascontiguousarray(chunk[:, indices])

    def _init_robot(self, robot_type: str, robot=None) -> None:
        """Create RobotClient + resolve camera / state mappings."""
        self._robot = robot if robot is not None else RobotClient(robot_type)

        # Cameras: only those that match a policy input key
        # ``observation.images.<cam>``. Cameras advertised by the robot
        # but not consumed by the policy are silently ignored — same
        # behavior as GR00TInference.
        policy_image_keys = self._policy_image_keys()
        active = self._resolve_camera_mappings(
            self._robot.camera_names,
            policy_image_keys,
        )
        if not active and policy_image_keys:
            raise RuntimeError(
                "No cameras match the policy's expected input keys: "
                f"policy needs {sorted(policy_image_keys)}, robot has "
                f"{self._robot.camera_names}"
            )
        self._cameras = active

        # State modalities: sorted follower joint groups. We follow the
        # same convention groot uses (sorted modality names map to the
        # training-time concat order). Synthetic per-modality views (with
        # ``parent``) win over their leaf physical group; otherwise the
        # leaf group is used directly.
        groups = self._robot._config.get("joint_groups", {})
        parents = {cfg.get("parent") for cfg in groups.values() if cfg.get("parent")}
        modality_groups = []
        for name, cfg in groups.items():
            if cfg.get("role") != "follower" or not name.startswith("follower_"):
                continue
            if cfg.get("parent"):
                modality_groups.append(name)
            elif name not in parents:
                modality_groups.append(name)
        modalities = sorted(name[len("follower_"):] for name in modality_groups)
        if not modalities:
            raise RuntimeError(
                f"No follower joint groups in robot_type={robot_type}"
            )

        # Mobile is sourced from sensors["odom"] in the new schema —
        # bridge it into observation.state alongside the joint states so
        # policies trained on the legacy physical_ai_server pipeline
        # (with mobile as a 3-vector modality) still see it.
        sensors = self._robot._config.get("sensors", {})
        self._has_mobile_state = "odom" in sensors
        if self._has_mobile_state:
            modalities = sorted(set(modalities) | {"mobile"})

        self._state_modalities = modalities
        self._action_keys = list(modalities)

        # Block until at least one frame from each sensor lands. 10 s is
        # generous — typical hardware comes up in <2 s.
        self._robot.wait_for_ready(timeout=10.0)
        logger.info(
            "Robot ready: cameras=%s state_modalities=%s",
            list(self._cameras.keys()),
            self._state_modalities,
        )

    def _teardown_robot(self) -> None:
        self._dataset_simulation = None
        if getattr(self, "_relative_eef", None) is not None:
            self._relative_eef.close()
            self._relative_eef = None
        if self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
            self._robot = None

    def _policy_image_keys(self) -> set:
        try:
            features = getattr(self._policy.config, "input_features", {}) or {}
            return {k for k in features.keys() if k.startswith(_IMAGE_KEY_PREFIX)}
        except Exception:
            return set()

    @classmethod
    def _resolve_camera_mappings(
        cls,
        robot_camera_names: Iterable[str],
        policy_image_keys: set,
    ) -> Dict[str, str]:
        """Map RobotClient camera names to policy image feature keys.

        The canonical Cyclo camera names are ``cam_<side>_<part>`` such as
        ``cam_left_head``. Some runtime configs or checkpoints may expose
        ``rgb.`` prefixes. Older single-head checkpoints may use
        ``cam_head`` for the left head camera. Exact matches remain preferred.
        """
        camera_names = list(robot_camera_names)
        if not policy_image_keys:
            return {cam: f"{_IMAGE_KEY_PREFIX}{cam}" for cam in camera_names}

        active: Dict[str, str] = {}
        used_policy_keys = set()
        for cam in camera_names:
            exact = f"{_IMAGE_KEY_PREFIX}{cam}"
            candidates = cls._camera_policy_key_candidates(cam)
            matches = sorted(policy_image_keys & candidates)
            if not matches:
                continue

            if exact in matches:
                chosen = exact
            elif len(matches) == 1:
                chosen = matches[0]
            else:
                raise RuntimeError(
                    f"Ambiguous camera mapping for {cam}: matches {matches}"
                )

            if chosen in used_policy_keys:
                raise RuntimeError(
                    f"Policy camera key {chosen} matched multiple robot cameras"
                )
            active[cam] = chosen
            used_policy_keys.add(chosen)

        missing = sorted(policy_image_keys - used_policy_keys)
        if missing:
            raise RuntimeError(
                "Missing camera mappings for policy input keys: "
                f"{missing}; robot has {camera_names}; matched {active}"
            )
        return active

    @staticmethod
    def _camera_policy_key_candidates(camera_name: str) -> set:
        aliases = {camera_name}
        parts = camera_name.split(".")
        suffix = parts[-1]
        prefixes = parts[:-1]
        aliases.add(suffix)

        semantic_names = {suffix}
        if suffix == "cam_left_head":
            semantic_names.add("cam_head")
            # AI Worker datasets record the left head camera as the scene view.
            semantic_names.add("scene")

        match = _CAMERA_SEMANTIC_RE.match(suffix)
        if match:
            first = match.group("a")
            second = match.group("b")
            side = first if first in {"left", "right"} else second
            part = first if first in {"head", "wrist"} else second
            if side in {"left", "right"} and part in {"head", "wrist"}:
                semantic_names.add(f"cam_{side}_{part}")
                semantic_names.add(f"cam_{part}_{side}")
                # Datasets often drop the cam_ prefix, e.g. wrist_left.
                semantic_names.add(f"{part}_{side}")
                semantic_names.add(f"{side}_{part}")

        for name in semantic_names:
            aliases.add(name)
            aliases.add(f"rgb.{name}")
            if prefixes:
                aliases.add(".".join([*prefixes, name]))

        return {f"{_IMAGE_KEY_PREFIX}{alias}" for alias in aliases}
