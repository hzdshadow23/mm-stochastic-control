import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feynman_kac_checks as fk  # noqa: E402

BS_ARGS = (100.0, 100.0, 0.05, 0.2, 1.0)


def rng():
    return np.random.default_rng(2024)


def test_heat_quadratic():
    mc, exact = fk.heat_quadratic(rng())
    assert abs(mc - exact) / exact < 5e-3


def test_black_scholes_three_ways():
    closed = fk.bs_closed(*BS_ARGS)
    assert abs(closed - 10.4506) < 1e-4
    assert abs(fk.bs_monte_carlo(rng(), *BS_ARGS) - closed) < 0.03
    assert abs(fk.bs_crank_nicolson(*BS_ARGS) - closed) < 5e-3


def test_poisson_ode_system():
    ode, exact = fk.poisson_ode()
    assert abs(exact - 39.0) < 1e-12
    assert abs(ode - exact) < 1e-6
    assert abs(fk.poisson_mc(rng()) - exact) / exact < 5e-3


def test_optimal_half_spread():
    grid, exact = fk.optimal_half_spread()
    assert abs(grid - exact) < 1e-4
