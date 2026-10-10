import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hjb_checks as hj  # noqa: E402

V_LQ = -0.4 - 0.5 * np.log(5.0)          # Eq. (4.5): c=4, sigma=1, tau=1, x=1
THETA_RN = 2 * 140.0 / (np.e * 1.5)       # Eq. (4.10): A=140, k=1.5, tau=1


def rng():
    return np.random.default_rng(404)


# ---------------- Example 4.1: LQ ----------------
def test_riccati_ode_matches_closed_form():
    P0, K0 = hj.riccati_ode()
    assert abs(P0 - 0.8) < 1e-9
    assert abs(K0 - 0.5 * np.log(5.0)) < 1e-9


def test_closed_form_solves_hjb_pointwise():
    """Plug (4.5) into (4.2) with finite differences in t and x at several points."""
    e = 1e-4
    for t in (0.0, 0.3, 0.7):
        for x in (-1.5, 0.2, 1.0):
            vt = (hj.v_closed(t + e, x) - hj.v_closed(t - e, x)) / (2 * e)
            vx = (hj.v_closed(t, x + e) - hj.v_closed(t, x - e)) / (2 * e)
            vxx = (hj.v_closed(t, x + e) - 2 * hj.v_closed(t, x) + hj.v_closed(t, x - e)) / e**2
            assert abs(vt + 0.5 * vxx + 0.5 * vx**2) < 1e-5


def test_finite_difference_hjb():
    """Monotone FD scheme with brute-force sup over a control grid (no ansatz)."""
    assert abs(hj.fd_hjb() - V_LQ) < 1e-3


def test_cole_hopf_feynman_kac():
    """v = sigma^2 ln E[exp(-c X_T^2 / (2 sigma^2))] (Remark 4.1) by Monte Carlo."""
    assert abs(hj.cole_hopf_mc(rng()) - V_LQ) < 3e-3


def test_verification_inequality_constant_gains():
    """Theorem 2.2(i): every admissible control does no better than w = v.
    Linear feedbacks with constant gain (exact rewards by moment ODE) all lose."""
    for kappa in (0.0, 0.5, 1.0, 2.0, 4.0):
        assert hj.constant_gain_value(kappa) < V_LQ
    assert abs(hj.constant_gain_value(0.0) - (-0.5 * 4 * 2)) < 1e-8   # -1/2 c (x^2+sigma^2 T)
    k_best, j_best = hj.best_constant_gain()
    assert V_LQ - j_best > 0.1            # time-varying gain P(t) is strictly better


# ---------------- Example 4.2: market maker ----------------
def test_sup_formula():
    """sup_delta Lam(delta)(delta + p) = A e^{-1} e^{kp} / k at delta = 1/k - p (4.9)."""
    p = np.array([-0.5, -0.1, 0.0, 0.2, 0.6, 1.0])
    val_c, d_c = hj.sup_closed(p)
    val_g, d_g = hj.sup_grid(p)
    assert np.max(np.abs(val_c - val_g) / val_c) < 1e-7
    assert np.max(np.abs(d_c - d_g)) < 2e-4
    interior = 1 / hj.K_MM - p > 0
    expect = hj.A_MM / (np.e * hj.K_MM) * np.exp(hj.K_MM * p)
    assert np.allclose(val_c[interior], expect[interior], rtol=1e-12)


def test_risk_neutral_market_maker():
    qs, th0, db, da, _ = hj.mm_ode(0.0)
    i0 = len(qs) // 2
    assert abs(THETA_RN - 68.6708) < 1e-4
    assert abs(th0[i0] - THETA_RN) < 1e-6
    assert abs(db[i0] - 1 / 1.5) < 1e-6 and abs(da[i0] - 1 / 1.5) < 1e-6
    k = 1 / hj.K_MM
    mc, se = hj.mm_monte_carlo(rng(), lambda t, q: (k + 0 * q, k + 0 * q))
    assert abs(mc - THETA_RN) < 4 * se


def test_inventory_penalty_ode_vs_grid_and_mc():
    """phi > 0: closed-form-sup ODE vs explicit grid-sup solver vs Monte Carlo of the
    resulting feedback; symmetric 1/k quotes do strictly worse (verification)."""
    phi = 0.05
    qs, th0, db, da, sol = hj.mm_ode(phi)
    i0 = len(qs) // 2
    _, th_grid = hj.mm_explicit_grid(phi)
    assert abs(th0[i0] - th_grid[i0]) < 1e-3
    assert np.all(np.diff(db[i0 - 10:i0 + 11]) > 0)        # bid widens as q grows
    assert np.all(np.diff(da[i0 - 10:i0 + 11]) < 0)        # ask tightens as q grows
    assert abs(db[i0] - da[i0]) < 1e-9                      # symmetric at q = 0
    mc_opt, se_opt = hj.mm_monte_carlo(rng(), hj.ode_quotes(sol), phi=phi)
    assert abs(mc_opt - th0[i0]) < 4 * se_opt + 0.02        # 0.02: time-step bias
    k = 1 / hj.K_MM
    mc_sym, se_sym = hj.mm_monte_carlo(rng(), lambda t, q: (k + 0 * q, k + 0 * q), phi=phi)
    assert mc_sym < th0[i0] - 1.0
