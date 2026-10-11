# compare_solvers.py
# Compares the three inverse kinematics solvers on the same random targets.
#
# Each solver follows a path of 1,000 random reachable targets, always starting from the
# pose it reached for the previous target (as the interactive apps do between two clicks).
#
# Run from the repository root:
#     python scripts/compare_solvers.py

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinematics import analytic_ik, forward_kinematics, newton_ik

L1, L2 = 150.0, 100.0
N = 1000
MARGIN = 5.0   # keep targets this far from the workspace boundaries


def random_targets(seed=0):
    rng = np.random.default_rng(seed)
    radius = rng.uniform(abs(L1 - L2) + MARGIN, L1 + L2 - MARGIN, N)
    angle = rng.uniform(-np.pi, np.pi, N)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])


SOLVERS = {
    "Analytic (closed form)": lambda t, a: analytic_ik(t, a, L1, L2),
    "Newton (20 iterations max)": lambda t, a: newton_ik(t, a, L1, L2, tol=1e-3, max_iter=20),
    "Damped Newton (0.5, 100 max)": lambda t, a: newton_ik(
        t, a, L1, L2, tol=1e-3, max_iter=100, damping=0.5, singular_eps=1e-10,
        rng=np.random.default_rng(0)),
}


def main():
    targets = random_targets()
    print(f"{N} random reachable targets, L1 = {L1:g}, L2 = {L2:g}, tolerance = 1e-3\n")
    print(f"{'Solver':<30}{'Reached':>10}{'Mean iterations':>18}{'Mean error':>14}{'Time per solve':>17}")
    for name, solve in SOLVERS.items():
        angles = np.array([0.5, 0.5])
        reached, iterations, errors = 0, [], []
        start = time.perf_counter()
        for target in targets:
            result = solve(target, angles)
            error = np.linalg.norm(forward_kinematics(result.angles, L1, L2) - target)
            if error < 1e-3:
                reached += 1
                iterations.append(result.iterations)
                errors.append(error)
            angles = result.angles
        elapsed = (time.perf_counter() - start) / N * 1e6
        print(f"{name:<30}{reached / N:>9.1%}{np.mean(iterations):>18.1f}{np.mean(errors):>14.1e}{elapsed:>14.0f} µs")


if __name__ == "__main__":
    main()
