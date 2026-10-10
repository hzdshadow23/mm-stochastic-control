import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import control_dpp_checks as cd  # noqa: E402

V_EXACT = -0.4 - 0.5 * np.log(5.0)      # Eq. (4.11) with c=4, sigma=1, tau=1, x=1


def rng():
    return np.random.default_rng(303)


def test_closed_form_numbers():
    assert abs(cd.v_feedback(0.0, 1.0) - V_EXACT) < 1e-14
    assert abs(V_EXACT - (-1.2047189562)) < 1e-9
    assert abs(cd.v_open_loop(0.0, 1.0) - (-2.4)) < 1e-14


def test_bellman_recursion_matches_closed_form():
    """Iterating (4.8) reproduces P_n = c/(1+c tau_n) and K_0 from (4.9)."""
    for N in (10, 100, 1000):
        Pn, Kn = cd.bellman_recursion(N)
        h = cd.T / N
        tn = np.arange(N + 1) * h
        assert np.max(np.abs(Pn - cd.P(tn))) < 1e-12
        K0 = 0.5 * cd.SIGMA**2 * h * sum(cd.C / (1 + cd.C * j * h) for j in range(N))
        assert abs(Kn[0] - K0) < 1e-12
        assert -0.5 * Pn[0] - Kn[0] <= V_EXACT          # v^h <= v, Eq. (4.5)


def test_first_order_convergence():
    """v^h - v ~ -(1/4) sigma^2 h (c - c/(1+c tau)) = -0.8 h."""
    N = 1000
    Pn, Kn = cd.bellman_recursion(N)
    err = (-0.5 * Pn[0] - Kn[0]) - V_EXACT
    assert abs(N * err - (-0.8)) / 0.8 < 0.02


def test_generic_grid_dpp():
    """Brute-force DPP (no quadratic ansatz) equals the exact discrete value (4.10)."""
    N = 10
    Pn, Kn = cd.bellman_recursion(N)
    assert abs(cd.grid_backward_induction(N) - (-0.5 * Pn[0] - Kn[0])) < 5e-4


def test_open_loop_optimisation():
    val, a = cd.open_loop_numeric()
    assert abs(val - (-2.4)) < 1e-6
    assert np.max(np.abs(a - (-0.8))) < 1e-4            # optimal schedule is constant


def test_monte_carlo_feedback_and_open_loop():
    mc, se = cd.mc_value(rng(), cd.optimal_policy)
    assert abs(mc - V_EXACT) < 4 * se + 5e-3              # 5e-3 allows Euler bias
    u = -cd.C * cd.X0 / (1 + cd.C * cd.T)
    mc_ol, se_ol = cd.mc_value(rng(), lambda s, x: u + 0 * x)
    assert abs(mc_ol - (-2.4)) < 4 * se_ol + 5e-3
    assert mc - mc_ol > 1.1                               # feedback beats open loop


def test_dpp_both_sides():
    """Cor. 2.2: optimal control keeps E[int f + v] = v; alpha = 0 loses value."""
    opt, zero, zero_exact = cd.dpp_check(rng())
    assert abs(zero_exact - (-1.3181471806)) < 1e-9
    assert abs(opt - V_EXACT) < 5e-3
    assert abs(zero - zero_exact) < 5e-3
    assert zero < opt - 0.1
