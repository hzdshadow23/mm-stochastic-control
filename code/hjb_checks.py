"""Numerical checks for note 04 (HJB equation and verification).

Example 4.1 (LQ steering, same problem as note 03 Sec. 4):
    HJB  d_t v + 1/2 sigma^2 v_xx + 1/2 v_x^2 = 0,  v(T,x) = -1/2 c x^2           (4.2)
    v = -1/2 P x^2 - K,  P' = P^2,  K' = -1/2 sigma^2 P                             (4.4)
    closed form  v = -1/2 c x^2/(1+c tau) - 1/2 sigma^2 ln(1+c tau)                 (4.5)
Example 4.2 (risk-neutral market maker, jump HJB):
    v = x + q s + theta(t, q)                                                        (4.7)
    d_t theta + sup_db L(db)[db + D+theta] + sup_da L(da)[da + D-theta] - phi q^2 = 0  (4.8)
    sup_delta L(delta)(delta + p) at delta = 1/k - p (clipped at 0)                  (4.9)
    phi = 0:  theta = 2 A tau / (e k),  delta* = 1/k                                 (4.10)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# ---- Example 4.1 parameters (as in note 03) ----
C, SIGMA, T, X0 = 4.0, 1.0, 1.0, 1.0
# ---- Example 4.2 parameters (Avellaneda-Stoikov 2008 values) ----
A_MM, K_MM, T_MM = 140.0, 1.5, 1.0


# ===================== Example 4.1: LQ =====================
def v_closed(t, x, c=C, sigma=SIGMA, T=T):
    tau = T - t
    return -0.5 * c * x**2 / (1 + c * tau) - 0.5 * sigma**2 * np.log1p(c * tau)


def riccati_ode(c=C, sigma=SIGMA, T=T):
    """Integrate (4.4) backward from P(T)=c, K(T)=0; return P(0), K(0)."""
    sol = solve_ivp(lambda t, y: [y[0] ** 2, -0.5 * sigma**2 * y[0]],
                    (T, 0.0), [c, 0.0], rtol=1e-12, atol=1e-12)
    return sol.y[0, -1], sol.y[1, -1]


def fd_hjb(x0=X0, c=C, sigma=SIGMA, T=T, L=4.0, nx=161, na=401, a_max=20.0, nt=1000):
    """Explicit monotone finite differences for the HJB (4.1), sup over a control grid.

        v^n = v^{n+1} + dt [ 1/2 sigma^2 D2 v + max_a ( a D0 v - 1/2 a^2 ) ],   D0 = central.

    Monotone (all neighbour weights >= 0) iff |a| <= sigma^2/dx and dt <= dx^2/sigma^2,
    so the control grid is capped at a_max <= sigma^2/dx. Boundary values by cubic
    extrapolation. No knowledge of the quadratic form of v is used.
    """
    xs = np.linspace(-L, L, nx)
    dx = xs[1] - xs[0]
    dt = T / nt
    assert a_max <= sigma**2 / dx + 1e-12 and dt <= dx**2 / sigma**2, "not monotone"
    As = np.linspace(-a_max, a_max, na)
    v = -0.5 * c * xs**2
    for _ in range(nt):
        D0 = (v[2:] - v[:-2]) / (2 * dx)
        D2 = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2
        ham = np.max(As[None, :] * D0[:, None] - 0.5 * As[None, :] ** 2, axis=1)
        v[1:-1] = v[1:-1] + dt * (0.5 * sigma**2 * D2 + ham)
        v[0] = 3 * v[1] - 3 * v[2] + v[3]
        v[-1] = 3 * v[-2] - 3 * v[-3] + v[-4]
    return np.interp(x0, xs, v)


def cole_hopf_mc(rng, x0=X0, c=C, sigma=SIGMA, T=T, n_paths=2_000_000):
    """Remark 4.1: v = sigma^2 ln w, w(t,x) = E[exp(-c (x + sigma W_tau)^2 / (2 sigma^2))]
    (Feynman-Kac for the heat equation, note 02). Monte Carlo of w, no HJB involved."""
    XT = x0 + sigma * np.sqrt(T) * rng.normal(size=n_paths)
    w = np.mean(np.exp(-c * XT**2 / (2 * sigma**2)))
    return sigma**2 * np.log(w)


def constant_gain_value(kappa, x0=X0, c=C, sigma=SIGMA, T=T):
    """Exact reward of the (suboptimal) feedback alpha = -kappa X, via the moment ODE
    m' = -2 kappa m + sigma^2,  m(0) = x0^2,  J = -1/2 kappa^2 int m - 1/2 c m(T)."""
    sol = solve_ivp(lambda t, y: [-2 * kappa * y[0] + sigma**2, kappa**2 * y[0]],
                    (0.0, T), [x0**2, 0.0], rtol=1e-11, atol=1e-12)
    m_T, int_k2m = sol.y[0, -1], sol.y[1, -1]
    return -0.5 * int_k2m - 0.5 * c * m_T


def best_constant_gain():
    res = minimize_scalar(lambda k: -constant_gain_value(k), bounds=(0.0, 10.0),
                          method="bounded", options={"xatol": 1e-8})
    return res.x, -res.fun


# ===================== Example 4.2: market maker =====================
def Lam(delta, A=A_MM, k=K_MM):
    return A * np.exp(-k * delta)


def sup_closed(p, A=A_MM, k=K_MM, d_lo=0.0):
    """sup_{delta >= d_lo} Lam(delta) (delta + p): interior optimum delta = 1/k - p (4.9),
    clipped at d_lo. Returns (value, argmax)."""
    d = np.maximum(1.0 / k - p, d_lo)
    return Lam(d, A, k) * (d + p), d


def sup_grid(p, A=A_MM, k=K_MM, d_lo=0.0, d_hi=6.0, n=60_001):
    d = np.linspace(d_lo, d_hi, n)
    vals = Lam(d, A, k)[None, :] * (d[None, :] + np.atleast_1d(p)[:, None])
    i = np.argmax(vals, axis=1)
    return vals[np.arange(len(i)), i], d[i]


def theta_closed(t, A=A_MM, k=K_MM, T=T_MM):
    return 2 * A * (T - t) / (np.e * k)


def _mm_rhs(theta, qs, phi, sup):
    """sup-terms of (4.8) for all q at once; no bid at q = Qmax, no ask at q = -Qmax."""
    up = np.empty_like(theta); dn = np.empty_like(theta)
    up[:-1] = theta[1:] - theta[:-1]          # theta(q+1) - theta(q)
    dn[1:] = theta[:-1] - theta[1:]           # theta(q-1) - theta(q)
    bid, db = sup(up); ask, da = sup(dn)
    bid, ask = np.array(bid, dtype=float), np.array(ask, dtype=float)
    bid[-1] = 0.0; ask[0] = 0.0
    return bid + ask - phi * qs**2, db, da


def mm_ode(phi, Qmax=60, A=A_MM, k=K_MM, T=T_MM, ell=0.0):
    """Solve (4.8) backward with the closed-form sup (4.9) by an adaptive ODE solver.
    Terminal theta(T,q) = -ell q^2. Returns grid q, theta(0,q), quotes at t=0."""
    qs = np.arange(-Qmax, Qmax + 1, dtype=float)
    sup = lambda p: sup_closed(p, A, k)  # noqa: E731
    sol = solve_ivp(lambda t, th: -_mm_rhs(th, qs, phi, sup)[0], (T, 0.0), -ell * qs**2,
                    rtol=1e-10, atol=1e-10, dense_output=True)
    th0 = sol.y[:, -1]
    _, db, da = _mm_rhs(th0, qs, phi, sup)
    return qs, th0, db, da, sol


def mm_explicit_grid(phi, Qmax=60, A=A_MM, k=K_MM, T=T_MM, nt=2000, n_delta=4001, ell=0.0):
    """Independent solver: explicit Euler in time, brute-force sup over a delta grid."""
    qs = np.arange(-Qmax, Qmax + 1, dtype=float)
    sup = lambda p: sup_grid(p, A, k, n=n_delta)  # noqa: E731
    th = -ell * qs**2
    dt = T / nt
    for _ in range(nt):
        th = th + dt * _mm_rhs(th, qs, phi, sup)[0]
    return qs, th


def ode_quotes(sol, k=K_MM, Qmax=60):
    """Optimal feedback (4.9) read off the ODE solution: delta_b(t,q), delta_a(t,q)."""
    def quote_fn(t, q):
        th = sol.sol(t)
        j = (q + Qmax).astype(int)
        up = th[np.minimum(j + 1, 2 * Qmax)] - th[j]
        dn = th[np.maximum(j - 1, 0)] - th[j]
        return np.maximum(1.0 / k - up, 0.0), np.maximum(1.0 / k - dn, 0.0)
    return quote_fn


def mm_monte_carlo(rng, quote_fn, phi=0.0, A=A_MM, k=K_MM, T=T_MM, sigma=2.0,
                   s0=100.0, n_paths=20_000, n_steps=1000, Qmax=60):
    """Simulate cash X, inventory q, mid S with fills as Bernoulli(Lam dt) per step.
    Returns mean and s.e. of  X_T + q_T S_T - phi int q^2 dt - (x0 + q0 s0)  with x0=q0=0."""
    dt = T / n_steps
    X = np.zeros(n_paths); q = np.zeros(n_paths); S = np.full(n_paths, s0)
    pen = np.zeros(n_paths)
    for i in range(n_steps):
        db, da = quote_fn(i * dt, q)
        bid = (rng.random(n_paths) < Lam(db, A, k) * dt) & (q < Qmax)
        ask = (rng.random(n_paths) < Lam(da, A, k) * dt) & (q > -Qmax)
        X += np.where(bid, -(S - db), 0.0) + np.where(ask, S + da, 0.0)
        q += bid.astype(float) - ask.astype(float)
        pen += phi * q**2 * dt
        S += sigma * np.sqrt(dt) * rng.normal(size=n_paths)
    w = X + q * S - pen
    return w.mean(), w.std() / np.sqrt(n_paths)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("LQ closed v(0,1):", v_closed(0, X0))
    P0, K0 = riccati_ode()
    print("Riccati ODE:", P0, K0, "->", -0.5 * P0 - K0)
    print("FD HJB:", fd_hjb())
    print("Cole-Hopf MC:", cole_hopf_mc(rng))
    print("best constant gain:", best_constant_gain())
    print("MM theta closed:", theta_closed(0.0))
    qs, th0, db, da, _ = mm_ode(0.0)
    i0 = len(qs) // 2
    print("MM ODE phi=0 theta(0,0):", th0[i0], "delta:", db[i0], da[i0])
    print("MM grid phi=0:", mm_explicit_grid(0.0)[1][i0])
    one_k = 1 / K_MM
    print("MM MC phi=0:", mm_monte_carlo(rng, lambda t, q: (one_k + 0 * q, one_k + 0 * q)))
    phi = 0.05
    qs, th0, db, da, sol = mm_ode(phi)
    print(f"phi={phi}: ODE theta(0,0)={th0[i0]:.5f}  grid={mm_explicit_grid(phi)[1][i0]:.5f}")
    print("  MC of optimal feedback:", mm_monte_carlo(rng, ode_quotes(sol), phi=phi))
    print("  MC of symmetric 1/k quotes:",
          mm_monte_carlo(rng, lambda t, q: (one_k + 0 * q, one_k + 0 * q), phi=phi))
    for j in range(i0 - 3, i0 + 4):
        print(f"  q={qs[j]:+.0f}  delta_b={db[j]:.4f}  delta_a={da[j]:.4f}  theta={th0[j]:.4f}")
