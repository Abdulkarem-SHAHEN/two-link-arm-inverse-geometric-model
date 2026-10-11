# kinematics.py
# Geometric models of a planar 2R robot arm (2 links, 2 revolute joints), independent of
# any user interface.
#
#   forward_kinematics  direct geometric model:   joint angles -> end-effector position
#   newton_ik           inverse geometric model, solved numerically with the Jacobian
#   analytic_ik         inverse geometric model, closed-form solution
#
# "Inverse kinematics" is used in the English sense: positions, not velocities.
#
# Conventions
#   phi1 : angle of the first link, measured from the x-axis
#   phi2 : angle of the second link, measured from the first link
#   Angles are in radians. Lengths and positions share one unit (pixels or metres).

from typing import Callable, NamedTuple, Optional

import numpy as np


class IKResult(NamedTuple):
    angles: np.ndarray     # [phi1, phi2]
    iterations: int        # Newton iterations performed (0 for the analytic solver)
    converged: bool        # True if the end effector is within the tolerance of the target


# -----------------------------
# Forward kinematics
# -----------------------------
def joint_positions(angles: np.ndarray, l1: float, l2: float):
    """Positions of the elbow and of the end effector, relative to the base."""
    phi1, phi2 = angles
    elbow = np.array([l1 * np.cos(phi1), l1 * np.sin(phi1)])
    end = elbow + np.array([l2 * np.cos(phi1 + phi2), l2 * np.sin(phi1 + phi2)])
    return elbow, end


def forward_kinematics(angles: np.ndarray, l1: float, l2: float) -> np.ndarray:
    """Position of the end effector for the given joint angles."""
    phi1, phi2 = angles
    x = l1 * np.cos(phi1) + l2 * np.cos(phi1 + phi2)
    y = l1 * np.sin(phi1) + l2 * np.sin(phi1 + phi2)
    return np.array([x, y])


def jacobian(angles: np.ndarray, l1: float, l2: float) -> np.ndarray:
    """Matrix of the partial derivatives of the end-effector position with respect to the angles."""
    phi1, phi2 = angles
    dx_dphi1 = -l1 * np.sin(phi1) - l2 * np.sin(phi1 + phi2)
    dx_dphi2 = -l2 * np.sin(phi1 + phi2)
    dy_dphi1 = l1 * np.cos(phi1) + l2 * np.cos(phi1 + phi2)
    dy_dphi2 = l2 * np.cos(phi1 + phi2)
    return np.array([[dx_dphi1, dx_dphi2],
                     [dy_dphi1, dy_dphi2]])


def robot_func(target: np.ndarray, l1: float, l2: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Higher-order function: returns f(v) = forward_kinematics(v) - target.
    The joint angles that reach the target are the roots of f.
    """
    def f(v: np.ndarray) -> np.ndarray:
        return forward_kinematics(v, l1, l2) - target
    return f


def is_reachable(target: np.ndarray, l1: float, l2: float) -> bool:
    """A point is reachable when its distance to the base lies between |l1 - l2| and l1 + l2."""
    d = np.linalg.norm(target)
    return bool(abs(l1 - l2) <= d <= l1 + l2)


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Map angles to [-pi, pi). The pose is unchanged; Newton steps can add full turns."""
    return (np.asarray(angles, dtype=float) + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# Inverse kinematics: Newton's method
# -----------------------------
def newton_ik(target: np.ndarray, guess: np.ndarray, l1: float, l2: float,
              tol: float = 1e-3, max_iter: int = 20, damping: float = 1.0,
              singular_eps: Optional[float] = None,
              rng: Optional[np.random.Generator] = None) -> IKResult:
    """
    Solve forward_kinematics(v) = target with Newton's method, starting from `guess`.

        v_{n+1} = v_n - damping * J(v_n)^-1 * f(v_n)

    damping       1.0 is the plain Newton step; smaller values take shorter, more stable steps.
    singular_eps  None: stop if the linear system cannot be solved.
                  A number: when |det J| falls below it, the angles are nudged by a small random
                  amount to leave the singular configuration, and the iteration continues.
    """
    f = robot_func(np.asarray(target, dtype=float), l1, l2)
    rng = rng or np.random.default_rng()
    v = np.array(guess, dtype=float)
    iterations = 0

    for iterations in range(max_iter):
        fv = f(v)

        # Check for convergence
        if np.linalg.norm(fv) < tol:
            return IKResult(wrap_angles(v), iterations, True)

        J = jacobian(v, l1, l2)

        # Jacobian singular (determinant close to zero): perturb the angles and try again
        if singular_eps is not None and abs(np.linalg.det(J)) < singular_eps:
            v += rng.uniform(-0.1, 0.1, size=v.shape)
            continue

        # Solve J * delta = f(v) for delta
        try:
            delta = np.linalg.solve(J, fv)
        except np.linalg.LinAlgError:
            if singular_eps is None:
                break
            v += rng.uniform(-0.1, 0.1, size=v.shape)
            continue

        v = v - damping * delta
    else:
        iterations = max_iter

    return IKResult(wrap_angles(v), iterations, bool(np.linalg.norm(f(v)) < tol))


# -----------------------------
# Inverse kinematics: closed-form solution
# -----------------------------
def analytic_ik(target: np.ndarray, current_angles: np.ndarray, l1: float, l2: float) -> IKResult:
    """
    Exact solution from the law of cosines.

    A reachable point has two solutions (elbow on either side of the line base-target);
    the one closest to the current joint angles is returned, so the arm does not flip.
    A point outside the workspace is handled by pointing the arm at it, fully extended
    (too far) or fully folded (too close).
    """
    target = np.asarray(target, dtype=float)
    d = np.linalg.norm(target)

    if d > l1 + l2:
        # Fully extended toward target
        phi1 = np.arctan2(target[1], target[0])
        return IKResult(np.array([phi1, 0.0]), 0, False)

    if d < abs(l1 - l2):
        # Fully folded toward target (-pi is also valid; pi is chosen)
        phi1 = np.arctan2(target[1], target[0])
        return IKResult(np.array([phi1, np.pi]), 0, False)

    # Reachable point: compute both solutions and pick the closest
    cos_phi2 = (d**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_phi2 = np.clip(cos_phi2, -1, 1)  # guard against rounding errors
    phi2_positive = np.arccos(cos_phi2)

    best_angles = None
    best_norm_diff = float('inf')
    for phi2 in (phi2_positive, -phi2_positive):
        a = l1 + l2 * np.cos(phi2)
        b = l2 * np.sin(phi2)
        det = a**2 + b**2
        if det < 1e-12:
            # Target on the base itself (only possible when l1 == l2): any phi1 works, keep the current one
            phi1 = current_angles[0]
        else:
            cos_phi1 = (a * target[0] + b * target[1]) / det
            sin_phi1 = (-b * target[0] + a * target[1]) / det
            phi1 = np.arctan2(sin_phi1, cos_phi1)
        candidate = np.array([phi1, phi2])

        # Angle difference to the current pose, mapped to [-pi, pi]
        diff = candidate - current_angles
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        norm_diff = np.linalg.norm(diff)
        if norm_diff < best_norm_diff:
            best_norm_diff = norm_diff
            best_angles = candidate

    return IKResult(best_angles, 0, True)
