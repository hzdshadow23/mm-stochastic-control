"""Numerical checks for note 01 (probability and Itô calculus).

Each function simulates a quantity whose closed form is derived in
notes/01_probability_and_ito.md and returns (simulated, closed_form).
"""
import numpy as np


def brownian_increments(rng, n_paths, n_steps, T):
    """Increments dW of shape (n_paths, n_steps) on a uniform grid of [0, T]."""
    dt = T / n_steps
    return rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))


def quadratic_variation(rng, n_paths=2000, n_steps=1000, T=2.0):
    """Eq. (2.1)-(2.3): Q_n = sum (dW)^2 -> T, Var(Q_n) = 2 T^2 / n."""
    dW = brownian_increments(rng, n_paths, n_steps, T)
    Q = (dW**2).sum(axis=1)
    return Q.mean(), Q.var(), T, 2 * T**2 / n_steps


def ito_integral_w_dw(rng, n_paths=2000, n_steps=2000, T=1.0):
    """Eq. (3.3): left-point sum of W dW vs 0.5 W_T^2 - 0.5 T, pathwise.

    Returns the RMS pathwise error, and the sample second moment of the
    integral vs the isometry value T^2 / 2 (Eq. 3.2).
    """
    dW = brownian_increments(rng, n_paths, n_steps, T)
    W = np.concatenate([np.zeros((n_paths, 1)), dW.cumsum(axis=1)], axis=1)
    ito_sum = (W[:, :-1] * dW).sum(axis=1)          # left endpoint
    closed = 0.5 * W[:, -1] ** 2 - 0.5 * T
    rms_err = np.sqrt(np.mean((ito_sum - closed) ** 2))
    return rms_err, np.mean(ito_sum**2), T**2 / 2


def gbm_mean(rng, S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_paths=400_000):
    """Eq. (5.4): E[S_T] = S0 exp(mu T) using the exact GBM solution."""
    WT = rng.normal(0.0, np.sqrt(T), n_paths)
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * WT)
    return ST.mean(), S0 * np.exp(mu * T)


def cara_certainty_equivalent(rng, s=100.0, sigma=2.0, T=1.0, gamma=0.1, q=3,
                              x=0.0, n_paths=2_000_000):
    """Eq. (5.5)-(5.6): CE = x + q s - 0.5 gamma q^2 sigma^2 T (= 298.2 here)."""
    ST = s + sigma * rng.normal(0.0, np.sqrt(T), n_paths)
    eu = np.mean(-np.exp(-gamma * (x + q * ST)))
    ce_mc = -np.log(-eu) / gamma                      # invert U(w) = -exp(-gamma w)
    return ce_mc, x + q * s - 0.5 * gamma * q**2 * sigma**2 * T


def poisson_mgf(rng, lam=3.0, t=2.0, theta=0.3, n_paths=1_000_000):
    """Eq. (6.5): E[exp(theta N_t)] = exp(lam t (e^theta - 1))."""
    N = rng.poisson(lam * t, n_paths)
    return np.mean(np.exp(theta * N)), np.exp(lam * t * (np.exp(theta) - 1))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("QV  (mean, var | T, 2T^2/n):", quadratic_variation(rng))
    print("WdW (rms err, E[I^2] | T^2/2):", ito_integral_w_dw(rng))
    print("GBM (MC | exact):", gbm_mean(rng))
    print("CARA CE (MC | exact):", cara_certainty_equivalent(rng))
    print("Poisson MGF (MC | exact):", poisson_mgf(rng))
