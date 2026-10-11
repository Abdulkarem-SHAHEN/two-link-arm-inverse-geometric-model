# Tests of the kinematics module. Run from the repository root:  pytest

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinematics import (analytic_ik, forward_kinematics, is_reachable, jacobian,
                        joint_positions, newton_ik)

L1, L2 = 150.0, 100.0


def random_reachable_targets(n, seed=0, margin=5.0):
    rng = np.random.default_rng(seed)
    radius = rng.uniform(abs(L1 - L2) + margin, L1 + L2 - margin, n)
    angle = rng.uniform(-np.pi, np.pi, n)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])


def test_forward_kinematics_known_poses():
    assert np.allclose(forward_kinematics(np.array([0.0, 0.0]), L1, L2), [L1 + L2, 0.0])
    assert np.allclose(forward_kinematics(np.array([np.pi / 2, 0.0]), L1, L2), [0.0, L1 + L2])
    assert np.allclose(forward_kinematics(np.array([0.0, np.pi]), L1, L2), [L1 - L2, 0.0])
    elbow, end = joint_positions(np.array([0.0, np.pi / 2]), L1, L2)
    assert np.allclose(elbow, [L1, 0.0]) and np.allclose(end, [L1, L2])


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    eps = 1e-6
    for _ in range(50):
        v = rng.uniform(-np.pi, np.pi, 2)
        numeric = np.zeros((2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = eps
            numeric[:, j] = (forward_kinematics(v + step, L1, L2) - forward_kinematics(v - step, L1, L2)) / (2 * eps)
        assert np.allclose(jacobian(v, L1, L2), numeric, atol=1e-4)


def test_jacobian_determinant_is_l1_l2_sin_phi2():
    rng = np.random.default_rng(2)
    for _ in range(50):
        v = rng.uniform(-np.pi, np.pi, 2)
        assert np.isclose(np.linalg.det(jacobian(v, L1, L2)), L1 * L2 * np.sin(v[1]))


def test_analytic_ik_reaches_every_reachable_target():
    current = np.array([0.5, 0.5])
    for target in random_reachable_targets(500):
        result = analytic_ik(target, current, L1, L2)
        assert result.converged
        assert np.allclose(forward_kinematics(result.angles, L1, L2), target, atol=1e-8)
        current = result.angles


def test_analytic_ik_picks_the_solution_closest_to_the_current_pose():
    target = np.array([180.0, 60.0])
    up = analytic_ik(target, np.array([0.0, 1.0]), L1, L2).angles
    down = analytic_ik(target, np.array([0.0, -1.0]), L1, L2).angles
    assert up[1] > 0 and down[1] < 0                     # the two elbow configurations
    for angles in (up, down):
        assert np.allclose(forward_kinematics(angles, L1, L2), target, atol=1e-8)


def test_analytic_ik_out_of_reach_points_at_the_target():
    far = analytic_ik(np.array([0.0, 400.0]), np.array([0.5, 0.5]), L1, L2)
    assert not far.converged and np.allclose(far.angles, [np.pi / 2, 0.0])       # fully extended
    near = analytic_ik(np.array([10.0, 0.0]), np.array([0.5, 0.5]), L1, L2)
    assert not near.converged and np.allclose(near.angles, [0.0, np.pi])         # fully folded


def test_newton_ik_converges_from_a_nearby_pose():
    # Start each solve from the exact solution of a nearby target, as the app does between clicks
    rng = np.random.default_rng(3)
    for target in random_reachable_targets(200, seed=4, margin=20.0):
        start = analytic_ik(target + rng.uniform(-15, 15, 2), np.array([0.5, 0.5]), L1, L2).angles
        result = newton_ik(target, start, L1, L2, tol=1e-3, max_iter=20)
        assert result.converged
        assert np.linalg.norm(forward_kinematics(result.angles, L1, L2) - target) < 1e-3


def test_damped_newton_recovers_from_a_singular_start():
    # phi2 = 0 is singular (arm fully stretched): plain Newton cannot take a step from there
    target = np.array([1.2, 0.7])
    result = newton_ik(target, np.array([0.0, 0.0]), 1.0, 1.0, tol=1e-6, max_iter=100,
                       damping=0.5, singular_eps=1e-10, rng=np.random.default_rng(5))
    assert result.converged
    assert np.linalg.norm(forward_kinematics(result.angles, 1.0, 1.0) - target) < 1e-6


def test_newton_ik_reports_failure_for_an_unreachable_target():
    result = newton_ik(np.array([400.0, 0.0]), np.array([0.5, 0.5]), L1, L2, tol=1e-3, max_iter=20)
    assert not result.converged


@pytest.mark.parametrize("target, expected", [
    ((200.0, 0.0), True), ((250.0, 0.0), True), ((50.0, 0.0), True),
    ((260.0, 0.0), False), ((40.0, 0.0), False),
])
def test_is_reachable(target, expected):
    assert is_reachable(np.array(target), L1, L2) is expected
