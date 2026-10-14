"""Weeks 5–6: body pose error -> DLS -> null-space posture -> bounded update.

Modern Robotics twists use [omega, v]. Pinocchio uses [v, omega], so the
URDF adapter explicitly swaps the rows. No ROS, policy, camera, or transport
dependency belongs in this module. Angles are radians; lift is meters.
"""

from dataclasses import dataclass
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pinocchio as pin


def dls_step(J: np.ndarray, V: np.ndarray, lam: float = 0.01) -> np.ndarray:
    """Week 6, p48: J.T @ inv(J @ J.T + lam**2 I) @ V.

    ``solve`` evaluates the same expression without explicitly inverting M.
    J may contain one TCP (6 rows) or both TCPs (12 rows).
    """
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("DLS damping must be positive and finite")
    M = J @ J.T + lam**2 * np.eye(J.shape[0])
    return J.T @ np.linalg.solve(M, V)


def nullspace_projector(J: np.ndarray) -> np.ndarray:
    """Week 6, p49: use the true pinv, since a damped inverse leaks task motion."""
    return np.eye(J.shape[1]) - np.linalg.pinv(J, rcond=1e-8) @ J


@dataclass(frozen=True)
class DLSOptions:
    damping: float = 0.01
    max_iterations: int = 100
    position_tolerance: float = 1e-4
    rotation_tolerance: float = 1e-3
    max_joint_step: float = 0.1
    nullspace_gain: float = 0.15
    elbow_position_weight: float = 4.0
    elbow_angle_weight: float = 1.0
    posture_weight: float = 0.02
    limit_weight: float = 0.002
    posture_iterations: int = 6

    def __post_init__(self):
        positive = ("damping", "position_tolerance", "rotation_tolerance", "max_joint_step")
        nonnegative = ("nullspace_gain", "elbow_position_weight",
                       "elbow_angle_weight", "posture_weight", "limit_weight")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Invalid DLS option {name}: {value}")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(self.posture_iterations, int) or self.posture_iterations < 0:
            raise ValueError("posture_iterations must be a nonnegative integer")


class DLSIK:
    """Solve the 14 arm joints; keep the measured lift fixed during each solve.

    ``solve`` has the same return shape as CycloIK and NodeIK:
    (named active joints, largest TCP position error, largest angle error).
    The first seed becomes the posture reference for the whole session.
    Waypoint seeds warm-start IK without silently moving that reference.
    """

    def __init__(self, urdf_path, active_names=None, ee_frames=None, options=None):
        self.options = options or DLSOptions()
        self.names = list(active_names) if active_names is not None else [
            *(f"arm_l_joint{i}" for i in range(1, 8)),
            *(f"arm_r_joint{i}" for i in range(1, 8)),
        ]
        if len(set(self.names)) != len(self.names):
            raise ValueError("Duplicate active joint names")
        arm_names = {f"arm_{side}_joint{i}" for side in ("l", "r") for i in range(1, 8)}
        if set(self.names) != arm_names:
            raise ValueError("DLS IK solves only the 14 arm joints; lift_joint must not be active")
        path = Path(urdf_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model = pin.buildModelFromUrdf(str(path))
        self.data = self.model.createData()
        self.q_indices, self.v_indices = [], []
        for name in self.names:
            if not self.model.existJointName(name):
                raise ValueError(f"Unknown active joint: {name}")
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"Active joint must be scalar: {name}")
            self.q_indices.append(joint.idx_q)
            self.v_indices.append(joint.idx_v)
        self.q_indices = np.array(self.q_indices)
        self.v_indices = np.array(self.v_indices)
        self.lower = self.model.lowerPositionLimit[self.q_indices].copy()
        self.upper = self.model.upperPositionLimit[self.q_indices].copy()
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("Active joints require finite URDF position limits")
        frames = list(ee_frames) if ee_frames is not None else [
            "end_effector_l_link", "end_effector_r_link",
        ]
        if len(frames) != 2:
            raise ValueError("Exactly two TCP frames are required")
        self.tcp_ids = [self._frame_id(name) for name in frames]
        self.elbow_ids = [self._frame_id(f"arm_{side}_link4") for side in ("l", "r")]
        self.shoulder_ids = [self._frame_id(f"arm_{side}_link1") for side in ("l", "r")]
        self.elbow_indices = [self.names.index(f"arm_{side}_joint4") for side in ("l", "r")]
        self.step_bounds = np.full(len(self.names), self.options.max_joint_step)
        self.reference = None
        self.reference_elbows = None
        self.last_diagnostics = {}

    def _frame_id(self, name):
        if not self.model.existFrame(name):
            raise ValueError(f"Unknown frame: {name}")
        return int(self.model.getFrameId(name))

    def _configuration(self, seed):
        missing = (set(self.names) | {"lift_joint"}) - set(seed)
        if missing:
            raise ValueError(f"Missing seed joints: {sorted(missing)}")
        if not np.isfinite(list(seed.values())).all():
            raise ValueError("Seed joints must be finite")
        q = pin.neutral(self.model)
        for name, value in seed.items():
            if self.model.existJointName(name):
                joint = self.model.joints[self.model.getJointId(name)]
                if joint.nq == 1:
                    q[joint.idx_q] = value
        q[self.q_indices] = np.clip(q[self.q_indices], self.lower, self.upper)
        return q

    def _update(self, q):
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def _jacobian(self, frame, reference=pin.LOCAL):
        J = pin.getFrameJacobian(self.model, self.data, frame, reference)
        return J[[3, 4, 5, 0, 1, 2]][:, self.v_indices].copy()

    def forward(self, seed):
        """Return left/right 4x4 TCP transforms without changing the reference."""
        self._update(self._configuration(seed))
        return tuple(self.data.oMf[frame].homogeneous.copy() for frame in self.tcp_ids)

    def elbow_positions(self, seed):
        """Elbow locations relative to shoulder origins, in base-aligned axes."""
        self._update(self._configuration(seed))
        return self._elbows()

    def _elbows(self):
        return np.array([
            self.data.oMf[elbow].translation - self.data.oMf[shoulder].translation
            for elbow, shoulder in zip(self.elbow_ids, self.shoulder_ids)
        ])

    def set_reference(self, seed):
        """Explicitly choose the posture to preserve; normally set once on load."""
        q = self._configuration(seed)
        self._update(q)
        self.reference = q[self.q_indices].copy()
        self.reference_elbows = self._elbows().copy()

    @staticmethod
    def _validate_target(T):
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError("TCP target must be a finite 4x4 transform")
        R = T[:3, :3]
        if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(R), 1.0, atol=1e-6)):
            raise ValueError("TCP target must be a rigid SE(3) transform")
        return T

    def _task(self, targets):
        errors, jacobians, positions, angles = [], [], [], []
        for frame, Td in zip(self.tcp_ids, targets):
            T = self.data.oMf[frame].homogeneous
            # Week 5, p60: Vb = se3ToVec(MatrixLog6(TransInv(T) @ Td)).
            Vb = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ Td))
            errors.append(Vb)
            jacobians.append(self._jacobian(frame))
            positions.append(np.linalg.norm(T[:3, 3] - Td[:3, 3]))
            angles.append(np.linalg.norm(Vb[:3]))
        return np.vstack(jacobians), np.concatenate(errors), max(positions), max(angles)

    def _posture_gradient(self, q):
        th = q[self.q_indices]
        opt = self.options
        grad = opt.posture_weight * (th - self.reference)
        grad[self.elbow_indices] += opt.elbow_angle_weight * (
            th[self.elbow_indices] - self.reference[self.elbow_indices])
        for side, (elbow, shoulder) in enumerate(zip(self.elbow_ids, self.shoulder_ids)):
            delta = self._elbows()[side] - self.reference_elbows[side]
            Jp = (self._jacobian(elbow, pin.LOCAL_WORLD_ALIGNED)[3:]
                  - self._jacobian(shoulder, pin.LOCAL_WORLD_ALIGNED)[3:])
            grad += opt.elbow_position_weight * Jp.T @ delta / 0.3**2
        # Week 6, p50: H_grad = 2 * (th - mid) / span**2.
        mid, span = (self.lower + self.upper) / 2, self.upper - self.lower
        grad += opt.limit_weight * 2 * (th - mid) / span**2
        return grad

    def _bounded_trial(self, q, step, scale=1.0):
        # Scale the whole vector, preserving its primary/null-space direction.
        factor = max(1.0, float(np.max(np.abs(step) / self.step_bounds)))
        trial = q.copy()
        trial[self.q_indices] = np.clip(
            q[self.q_indices] + scale * step / factor, self.lower, self.upper)
        return trial

    def _joint_step(self, Jb, Vb, q, use_posture=True):
        """Re-solve with outward-moving arm joints at their limits held."""
        opt = self.options
        free = np.ones(len(self.names), dtype=bool)
        th = q[self.q_indices]
        H_grad = (self._posture_gradient(q)
                  if use_posture else np.zeros(len(self.names)))
        for _ in range(len(self.names) + 1):
            J = Jb[:, free]
            primary = np.zeros(len(self.names))
            secondary = np.zeros(len(self.names))
            primary[free] = dls_step(J, Vb, opt.damping)
            N = nullspace_projector(J)
            secondary[free] = -opt.nullspace_gain * (N @ H_grad[free])
            # The null-space property is first-order. Shrink its finite step
            # with the task error to prevent persistent second-order drift.
            secondary *= min(1.0, np.linalg.norm(primary) / (np.linalg.norm(secondary) + 1e-12))
            step = primary + secondary
            blocked = free & (((th <= self.lower + 1e-9) & (step < 0))
                              | ((th >= self.upper - 1e-9) & (step > 0)))
            if not np.any(blocked):
                return primary, step
            free[blocked] = False
        return np.zeros(len(self.names)), np.zeros(len(self.names))

    def _posture_cost(self, q):
        opt = self.options
        th = q[self.q_indices]
        delta = th - self.reference
        mid, span = (self.lower + self.upper) / 2, self.upper - self.lower
        return float(
            opt.posture_weight / 2 * (delta @ delta)
            + opt.elbow_angle_weight / 2 * np.sum(delta[self.elbow_indices]**2)
            + opt.elbow_position_weight / (2 * .3**2) * np.sum((self._elbows() - self.reference_elbows)**2)
            + opt.limit_weight * np.sum(((th - mid) / span)**2))

    def _refine_posture(self, q, targets):
        """Week 6, p49: null-space self-motion, then rewind nonlinear TCP drift.

        Keep only changes that lower the posture cost and still satisfy both
        TCP tolerances. This makes elbow preservation a secondary task.
        """
        opt = self.options
        steps = 0
        if opt.nullspace_gain == 0:
            return q, steps
        for _ in range(opt.posture_iterations):
            self._update(q)
            Jb, _, _, _ = self._task(targets)
            cost = self._posture_cost(q)
            free = np.ones(len(self.names), dtype=bool)
            th = q[self.q_indices]
            gradient = self._posture_gradient(q)
            for _ in range(len(self.names) + 1):
                step = np.zeros(len(self.names))
                J = Jb[:, free]
                step[free] = -opt.nullspace_gain * (nullspace_projector(J) @ gradient[free])
                blocked = free & (((th <= self.lower + 1e-9) & (step < 0))
                                  | ((th >= self.upper - 1e-9) & (step > 0)))
                if not np.any(blocked):
                    break
                free[blocked] = False
            if np.linalg.norm(step) < 1e-6:
                break
            accepted = False
            for scale in (1.0, 0.5, 0.25):
                trial = self._bounded_trial(q, step, scale)
                for _ in range(5):
                    self._update(trial)
                    J, V, position, angle = self._task(targets)
                    if position < opt.position_tolerance and angle < opt.rotation_tolerance:
                        break
                    correction, _ = self._joint_step(J, V, trial, use_posture=False)
                    trial = self._bounded_trial(trial, correction)
                self._update(trial)
                _, _, position, angle = self._task(targets)
                if (position < opt.position_tolerance and angle < opt.rotation_tolerance
                        and self._posture_cost(trial) < cost - 1e-12):
                    q, accepted = trial, True
                    steps += 1
                    break
            if not accepted:
                break
        return q, steps

    def solve(self, seed, left, right):
        targets = [self._validate_target(T) for T in (left, right)]
        q = self._configuration(seed)
        if self.reference is None:
            self.set_reference(seed)
        opt = self.options
        best_q, best_score = q.copy(), np.inf
        reason = "iteration_limit"
        for iteration in range(opt.max_iterations):
            self._update(q)
            Jb, Vb, position, angle = self._task(targets)
            score = float(Vb @ Vb)
            if score < best_score:
                best_q, best_score = q.copy(), score
            if position < opt.position_tolerance and angle < opt.rotation_tolerance:
                reason = "converged"
                break
            # Week 6: DLS primary - k * (N @ H_grad), for arm joints only.
            primary, step = self._joint_step(Jb, Vb, q)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                trial = self._bounded_trial(q, step, scale)
                self._update(trial)
                _, error, _, _ = self._task(targets)
                if error @ error < score - 1e-14:
                    q, accepted = trial, True
                    break
            if not accepted:
                # Joint clipping can obstruct the secondary step. Retry task only.
                for scale in (1.0, 0.5, 0.25, 0.125):
                    trial = self._bounded_trial(q, primary, scale)
                    self._update(trial)
                    _, error, _, _ = self._task(targets)
                    if error @ error < score - 1e-14:
                        q, accepted = trial, True
                        break
            if not accepted:
                reason = "stalled"
                break
        # Include the last accepted step, including at the iteration limit.
        self._update(q)
        _, error, _, _ = self._task(targets)
        if error @ error < best_score:
            best_q = q.copy()
        self._update(best_q)
        _, _, position, angle = self._task(targets)
        posture_steps = 0
        if iteration > 0 and position < opt.position_tolerance and angle < opt.rotation_tolerance:
            best_q, posture_steps = self._refine_posture(best_q, targets)
            self._update(best_q)
            _, _, position, angle = self._task(targets)
        self.last_diagnostics = {
            "iterations": iteration + 1,
            "posture_steps": posture_steps,
            "converged": bool(position < opt.position_tolerance and angle < opt.rotation_tolerance),
            "reason": reason,
            "position_error_m": float(position), "rotation_error_rad": float(angle),
            "elbow_drift_m": np.linalg.norm(self._elbows() - self.reference_elbows, axis=1).tolist(),
            "elbow_angle_drift_rad": (
                best_q[self.q_indices][self.elbow_indices] - self.reference[self.elbow_indices]).tolist(),
        }
        return dict(zip(self.names, best_q[self.q_indices].tolist())), float(position), float(angle)

    def initial_pose(self, seed, elbow_angle=-np.pi / 2):
        """Keep home TCP x/y and orientation; fix both J4 at -90°, let z settle.

        This is a posture generator, not a command publisher. Lift stays at
        its home value. The remaining 12 arm joints solve a 10-row task.
        """
        home_targets = self.forward(seed)
        q = self._configuration(seed)
        for index in self.elbow_indices:
            if not self.lower[index] <= elbow_angle <= self.upper[index]:
                raise ValueError("Initial elbow angle exceeds URDF limits")
            q[self.q_indices[index]] = elbow_angle
        free = [i for i in range(len(self.names)) if i not in self.elbow_indices]
        # A small symmetric shoulder seed avoids the straight-home singularity.
        for side in ("l", "r"):
            index = self.names.index(f"arm_{side}_joint1")
            q[self.q_indices[index]] = 0.5
        reason = "iteration_limit"
        for iteration in range(400):
            self._update(q)
            errors, jacobians = [], []
            for frame, Td in zip(self.tcp_ids, home_targets):
                T = self.data.oMf[frame].homogeneous
                omega = mr.so3ToVec(mr.MatrixLog3(Td[:3, :3] @ T[:3, :3].T))
                errors.extend([*omega, *(Td[:2, 3] - T[:2, 3])])
                jacobians.append(self._jacobian(frame, pin.LOCAL_WORLD_ALIGNED)[[0, 1, 2, 3, 4]][:, free])
            V = np.array(errors)
            if np.linalg.norm(V, np.inf) < 1e-7:
                reason = "converged"
                break
            step = np.zeros(len(self.names))
            step[free] = dls_step(np.vstack(jacobians), V, self.options.damping)
            q = self._bounded_trial(q, step)
        result = dict(seed)
        result.update(zip(self.names, q[self.q_indices].tolist()))
        reached = self.forward(result)
        if reason != "converged" or any(T[2, 3] <= home[2, 3] for T, home in zip(reached, home_targets)):
            raise RuntimeError("Could not find a raised 90-degree-elbow home pose")
        return result, {
            "iterations": iteration + 1, "elbow_angle_rad": float(elbow_angle),
            "home_tcp_m": [T[:3, 3].tolist() for T in home_targets],
            "raised_tcp_m": [T[:3, 3].tolist() for T in reached],
            "z_increase_m": [float(T[2, 3] - home[2, 3]) for T, home in zip(reached, home_targets)],
        }
