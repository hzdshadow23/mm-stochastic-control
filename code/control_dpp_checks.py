"""Numerical checks for note 03 (control problem and dynamic programming).

Running example (note 03, Section 4): steer a 1-d state to zero,
    dX = alpha ds + sigma dW,   reward  -1/2 alpha^2 per unit time,   terminal  -1/2 c X_T^2.

Closed forms checked here:
    open-loop value      v_OL = -1/2 c x^2 / (1 + c tau) - 1/2 c sigma^2 tau      (4.4)
    Bellman recursion    P_n = c / (1 + c tau_n),  K_n = 1/2 sigma^2 h sum P_m      (4.8)-(4.10)
    feedback value       v    = -1/2 P x^2 - 1/2 sigma^2 ln(1 + c tau)               (4.11)
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

C, SIGMA, T, X0 = 4.0, 1.0, 1.0, 1.0


# ---------- closed forms ----------
def P(t, c=C, T=T):
    return c / (1.0 + c * (T - t))


def v_feedback(t, x, c=C, sigma=SIGMA, T=T):
    """Eq. (4.11)."""
    tau = T - t
    return -0.5 * P(t, c, T) * x**2 - 0.5 * sigma**2 * np.log1p(c * tau)


def v_open_loop(t, x, c=C, sigma=SIGMA, T=T):
    """Eq. (4.4)."""
    tau = T - t
    return -0.5 * c * x**2 / (1.0 + c * tau) - 0.5 * c * sigma**2 * tau


# ---------- discrete-time Bellman recursion (Section 4.3) ----------
def bellman_recursion(N, c=C, sigma=SIGMA, T=T):
    """Backward recursion (4.8)-(4.9) for the quadratic ansatz v_n = -1/2 P_n x^2 - K_n.

    Returns arrays P_n, K_n for n = 0..N (time t_n = n h, h = T/N).
    """
    h = T / N
    Pn = np.empty(N + 1)
    Kn = np.empty(N + 1)
    Pn[N], Kn[N] = c, 0.0
    for n in range(N - 1, -1, -1):
        Pn[n] = Pn[n + 1] / (1.0 + Pn[n + 1] * h)
        Kn[n] = Kn[n + 1] + 0.5 * sigma**2 * h * Pn[n + 1]
    return Pn, Kn


def grid_backward_induction(N, x0=X0, c=C, sigma=SIGMA, T=T,
                            L=6.0, nx=241, na=801, a_max=20.0, n_gh=20):
    """Generic discrete DPP, no quadratic ansatz:
        v_n(x) = max_a { -1/2 a^2 h + E[ v_{n+1}(x + a h + sigma sqrt(h) xi) ] },  xi ~ N(0,1).

    x on a uniform grid, a on a uniform grid (brute-force max), the expectation by
    Gauss-Hermite quadrature, and v_{n+1} off the grid by a cubic spline (extrapolated).
    """
    h = T / N
    xs = np.linspace(-L, L, nx)
    As = np.linspace(-a_max, a_max, na)
    z, w = np.polynomial.hermite_e.hermegauss(n_gh)   # weight exp(-z^2/2)
    w = w / w.sum()
    v = -0.5 * c * xs**2
    for _ in range(N):
        spline = CubicSpline(xs, v, extrapolate=True)
        # mean next state for every (x, a): shape (nx, na)
        m = xs[:, None] + As[None, :] * h
        Ev = sum(wk * spline(m + sigma * np.sqrt(h) * zk) for zk, wk in zip(z, w))
        v = np.max(-0.5 * As[None, :] ** 2 * h + Ev, axis=1)
    return np.interp(x0, xs, v)


# ---------- open loop: numerical optimisation over deterministic controls ----------
def open_loop_numeric(N=20, x0=X0, c=C, sigma=SIGMA, T=T):
    """Maximise J over piecewise-constant deterministic controls a_0..a_{N-1}.

    For deterministic a, X_T ~ N(x0 + h sum a, sigma^2 T), so
    J(a) = -1/2 h sum a^2 - 1/2 c [ (x0 + h sum a)^2 + sigma^2 T ].
    """
    h = T / N

    def negJ(a):
        mT = x0 + h * a.sum()
        return 0.5 * h * (a**2).sum() + 0.5 * c * (mT**2 + sigma**2 * T)

    res = minimize(negJ, np.zeros(N), method="BFGS", options={"gtol": 1e-12})
    return -res.fun, res.x


# ---------- Monte Carlo ----------
def mc_value(rng, policy, t0=0.0, t1=T, x0=X0, terminal=None,
             c=C, sigma=SIGMA, n_paths=200_000, n_steps=400):
    """E[ int_{t0}^{t1} -1/2 alpha^2 ds + terminal(X_{t1}) ] for a feedback policy(s, x).

    Euler scheme on [t0, t1]; default terminal is the problem's g(x) = -1/2 c x^2.
    """
    if terminal is None:
        terminal = lambda x: -0.5 * c * x**2  # noqa: E731
    dt = (t1 - t0) / n_steps
    X = np.full(n_paths, x0)
    run = np.zeros(n_paths)
    for i in range(n_steps):
        s = t0 + i * dt
        a = policy(s, X)
        run -= 0.5 * a**2 * dt
        X = X + a * dt + sigma * np.sqrt(dt) * rng.normal(size=n_paths)
    vals = run + terminal(X)
    return vals.mean(), vals.std() / np.sqrt(n_paths)


def optimal_policy(s, x):
    return -P(s) * x


def dpp_check(rng, h=0.25):
    """Both sides of the DPP (2.1) at (0, X0) with horizon h:
    optimal feedback on [0,h] gives equality; alpha = 0 gives a strict '<'."""
    term = lambda x: v_feedback(h, x)  # noqa: E731
    opt, se_opt = mc_value(rng, optimal_policy, 0.0, h, terminal=term)
    zero, se_zero = mc_value(rng, lambda s, x: 0.0 * x, 0.0, h, terminal=term)
    zero_exact = -0.5 * P(h) * (X0**2 + SIGMA**2 * h) - 0.5 * SIGMA**2 * np.log1p(C * (T - h))
    return opt, zero, zero_exact


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("v (4.11) =", v_feedback(0, X0), "  v_OL (4.4) =", v_open_loop(0, X0))
    for N in (10, 100, 1000):
        Pn, Kn = bellman_recursion(N)
        print(f"N={N:5d}: P_0={Pn[0]:.6f}  v^h(0,1)={-0.5 * Pn[0] - Kn[0]:.6f}")
    print("grid DPP N=10:", grid_backward_induction(10))
    print("open-loop numeric:", open_loop_numeric()[0])
    print("MC feedback:", mc_value(rng, optimal_policy))
    u_ol = -C * X0 / (1 + C * T)
    print("MC open loop:", mc_value(rng, lambda s, x: u_ol + 0 * x))
    print("DPP check (opt, zero, zero exact):", dpp_check(rng))
