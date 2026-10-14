"""Numerical checks against independent least squares, PoE FK and finite differences."""

from dataclasses import replace
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pytest

from dls_ik import DLSIK, DLSOptions, dls_step, nullspace_projector
from dls_ik.initial_pose import read_home, write_home

URDF = Path(__file__).resolve().parents[3] / (
    "policy/lerobot/lerobot_engine/_vendor/dream/ai_worker/ffw_description/urdf/"
    "ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
)
# The normal tests run with /opt/cyclo_sdk bound; the engine is bound separately.
if not URDF.is_file():
    URDF = Path("/app/lerobot_engine/_vendor/dream/ai_worker/ffw_description/urdf/"
                "ffw_sg2_rev1_follower/ffw_sg2_follower.urdf")


@pytest.fixture
def solver():
    return DLSIK(URDF)


@pytest.fixture
def seed(solver):
    values = dict.fromkeys(solver.names, 0.0)
    values.update(arm_l_joint1=-0.3, arm_r_joint1=-0.3,
                  arm_l_joint2=0.5, arm_r_joint2=-0.5,
                  arm_l_joint4=-1.2, arm_r_joint4=-1.2, lift_joint=-0.25)
    return values


def test_dls_matches_stacked_least_squares_at_singularity():
    rng = np.random.default_rng(6)
    J = rng.normal(size=(12, 15))
    J[-1] = J[0] + J[1]
    V = rng.normal(size=12)
    lam = 0.01
    A = np.vstack([J, lam * np.eye(15)])
    expected = np.linalg.lstsq(A, np.r_[V, np.zeros(15)], rcond=None)[0]
    np.testing.assert_allclose(dls_step(J, V, lam), expected, atol=1e-10)


def test_exact_nullspace_does_not_leak_task_motion():
    J = np.random.default_rng(6).normal(size=(12, 15))
    N = nullspace_projector(J)
    np.testing.assert_allclose(J @ N, 0, atol=1e-13)
    np.testing.assert_allclose(N @ N, N, atol=1e-13)
    np.testing.assert_allclose(N.T, N, atol=1e-13)


def test_body_jacobian_matches_finite_difference(solver, seed):
    q = solver._configuration(seed)
    solver._update(q)
    frame = solver.tcp_ids[0]
    T = solver.data.oMf[frame].homogeneous.copy()
    Jb = solver._jacobian(frame)
    columns = []
    for index in solver.q_indices:
        trial = q.copy()
        trial[index] += 1e-7
        solver._update(trial)
        new = solver.data.oMf[frame].homogeneous
        columns.append(mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ new)) / 1e-7)
    np.testing.assert_allclose(Jb, np.array(columns).T, atol=2e-7)


def test_forward_and_jacobian_match_lecture_poe(solver, seed):
    home = dict.fromkeys(solver.names, 0.)
    M = solver.forward(home)[0]
    solver._update(solver._configuration(home))
    indices = [solver.lift_index, *(solver.names.index(f"arm_l_joint{i}") for i in range(1, 8))]
    Blist = solver._jacobian(solver.tcp_ids[0])[:, indices]
    theta = np.array([seed[solver.names[i]] for i in indices])
    expected = mr.FKinBody(M, Blist, theta)
    actual = solver.forward(seed)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    solver._update(solver._configuration(seed))
    np.testing.assert_allclose(solver._jacobian(solver.tcp_ids[0])[:, indices],
                               mr.JacobianBody(Blist, theta), atol=1e-12)


def test_identity_preserves_native_joints_and_reference(solver, seed):
    left, right = solver.forward(seed)
    result, position, angle = solver.solve(seed, left, right)
    np.testing.assert_allclose([result[name] for name in solver.names],
                               [seed[name] for name in solver.names], atol=1e-12)
    assert position < 1e-10 and angle < 1e-10
    reference = solver.reference.copy()
    left[0, 3] += 0.01
    result, position, angle = solver.solve(result, left, right)
    assert position < solver.options.position_tolerance
    np.testing.assert_array_equal(reference, solver.reference)


def test_bimanual_targets_with_shared_lift_and_rotation(solver, seed):
    left, right = solver.forward(seed)
    left[:3, 3] += [.02, -.01, .015]
    right[:3, 3] += [-.01, .01, .02]
    left[:3, :3] = left[:3, :3] @ mr.MatrixExp3(mr.VecToso3(np.array([.02, 0, 0])))
    result, position, angle = solver.solve(seed, left, right)
    assert position < 1e-4 and angle < 1e-3
    assert solver.lower[solver.lift_index] <= result["lift_joint"] <= 0


def test_raised_initial_pose_keeps_xy_orientation_and_90_degree_elbows(solver):
    home = dict.fromkeys(solver.names, 0.)
    initial, report = solver.initial_pose(home)
    for old, new in zip(solver.forward(home), solver.forward(initial)):
        np.testing.assert_allclose(old[:2, 3], new[:2, 3], atol=1e-7)
        np.testing.assert_allclose(old[:3, :3], new[:3, :3], atol=1e-7)
        assert new[2, 3] > old[2, 3] + .1
    assert initial["arm_l_joint4"] == initial["arm_r_joint4"] == -np.pi / 2
    assert initial["lift_joint"] == 0.
    assert report["iterations"] < 50
    # Regression: the upper-limit lift must be removed when it wants to rise.
    left, right = solver.forward(initial)
    left[0, 3] += .01
    right[2, 3] += .01
    _, position, angle = solver.solve(initial, left, right)
    assert position < 1e-4 and angle < 1e-3


def test_unreachable_returns_bounded_diagnostics(solver, seed):
    left, right = solver.forward(seed)
    left[0, 3] += 5.0
    result, position, angle = solver.solve(seed, left, right)
    values = np.array([result[name] for name in solver.names])
    assert np.isfinite([position, angle, *values]).all()
    assert np.all(values >= solver.lower) and np.all(values <= solver.upper)
    assert not solver.last_diagnostics["converged"]
    assert solver.last_diagnostics["iterations"] <= solver.options.max_iterations


def test_nullspace_reduces_elbow_motion_without_losing_tcp_targets(seed):
    results = []
    for gain in (0., .15):
        solver = DLSIK(URDF, options=DLSOptions(nullspace_gain=gain))
        anchors = solver.forward(seed)
        current, drifts = dict(seed), []
        for phase in np.linspace(0, 2 * np.pi, 60):
            targets = [T.copy() for T in anchors]
            targets[0][:3, 3] += [.04*np.sin(phase), .02*(1-np.cos(phase)), .02*np.sin(phase)]
            targets[1][:3, 3] += [.03*np.sin(phase), -.02*(1-np.cos(phase)), .01*np.sin(phase)]
            current, position, angle = solver.solve(current, *targets)
            assert position < 1e-4 and angle < 1e-3
            drifts.append(solver.last_diagnostics["elbow_drift_m"])
        results.append(np.mean(drifts, axis=0))
    assert np.all(results[1] < results[0] * .9)


@pytest.mark.parametrize("bad", [np.full((4, 4), np.nan), np.zeros((3, 3)), np.zeros((4, 4))])
def test_malformed_targets_are_rejected(solver, seed, bad):
    with pytest.raises(ValueError):
        solver.solve(seed, bad, np.eye(4))


def test_elbow_gradient_is_the_derivative_of_shoulder_relative_position(solver, seed):
    solver.set_reference(seed)
    moved = dict(seed, arm_l_joint1=-.2, arm_r_joint3=.15)
    q = solver._configuration(moved)
    # Isolate the position objective, then independently finite-difference it.
    solver.options = replace(solver.options, posture_weight=0, elbow_angle_weight=0, limit_weight=0)
    solver._update(q)
    analytical = solver._posture_gradient(q)
    def cost(test_q):
        solver._update(test_q)
        return solver.options.elbow_position_weight / (2 * .3**2) * np.sum(
            (solver._elbows() - solver.reference_elbows)**2)
    finite = []
    for index in solver.q_indices:
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite.append((cost(plus) - cost(minus)) / 2e-6)
    np.testing.assert_allclose(analytical, finite, atol=1e-8)


def test_initial_yaml_keeps_all_ros_sections_and_other_joints(tmp_path):
    template = Path(__file__).resolve().parents[1] / "config/zero_home.yaml"
    home = read_home(template)
    raised = dict(home, arm_l_joint4=-np.pi / 2, arm_r_joint4=-np.pi / 2)
    output = tmp_path / "raised.yaml"
    write_home(template, output, raised)
    actual = read_home(output)
    assert actual.keys() == home.keys()
    assert actual["head_joint1"] == home["head_joint1"]
    assert actual["gripper_l_joint1"] == home["gripper_l_joint1"]
    before = template.read_text().splitlines()
    after = output.read_text().splitlines()
    assert [s for s in before if not s.strip().startswith("home:")] == [
        s for s in after if not s.strip().startswith("home:")]


@pytest.mark.parametrize("kwargs", [{"damping": 0}, {"lift_weight": -1},
                                     {"nullspace_gain": np.nan}, {"max_iterations": 0}])
def test_invalid_options_fail_early(kwargs):
    with pytest.raises(ValueError):
        DLSOptions(**kwargs)
