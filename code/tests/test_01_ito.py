import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ito_checks as ic  # noqa: E402


def rng():
    return np.random.default_rng(12345)


def test_quadratic_variation():
    mean, var, T, var_exact = ic.quadratic_variation(rng())
    assert abs(mean - T) < 0.01
    assert abs(var - var_exact) / var_exact < 0.15


def test_ito_integral_and_isometry():
    rms_err, second_moment, iso = ic.ito_integral_w_dw(rng())
    assert rms_err < 0.03                       # discretization error ~ sqrt(dt)
    assert abs(second_moment - iso) / iso < 0.1


def test_gbm_mean():
    mc, exact = ic.gbm_mean(rng())
    assert abs(mc - exact) / exact < 2e-3


def test_cara_certainty_equivalent():
    mc, exact = ic.cara_certainty_equivalent(rng())
    assert abs(exact - 298.2) < 1e-12
    assert abs(mc - exact) < 0.05


def test_poisson_mgf():
    mc, exact = ic.poisson_mgf(rng())
    assert abs(mc - exact) / exact < 5e-3
