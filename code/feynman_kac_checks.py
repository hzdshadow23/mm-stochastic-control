"""Numerical checks for note 02 (Feynman–Kac).

Each check computes the same quantity from the PDE/ODE side and from the
expectation (Monte Carlo) side, plus a closed form where available.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.stats import norm


# ---------- Example 4.1: heat equation, g(x) = x^2 ----------
def heat_quadratic(rng, x=0.7, tau=2.0, n_paths=1_000_000):
    XT = x + rng.normal(0.0, np.sqrt(tau), n_paths)
    return np.mean(XT**2), x**2 + tau


# ---------- Example 4.2: Black–Scholes call ----------
def bs_closed(S, K, r, sigma, tau):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_monte_carlo(rng, S, K, r, sigma, tau, n_paths=2_000_000):
    Z = rng.normal(size=n_paths)
    ST = S * np.exp((r - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * Z)
    return np.exp(-r * tau) * np.mean(np.maximum(ST - K, 0.0))


def bs_crank_nicolson(S, K, r, sigma, tau, S_max_mult=4.0, nS=800, nt=800):
    """Solve Eq. (4.2) backward in time (forward in tau) with Crank–Nicolson.

    Grid S_j = j dS, j = 0..nS. Boundary: u(0) = 0, u(S_max) = S_max - K e^{-r tau}.
    """
    S_max = S_max_mult * K
    dS = S_max / nS
    dt = tau / nt
    Sg = np.arange(nS + 1) * dS
    u = np.maximum(Sg - K, 0.0)

    j = np.arange(1, nS)                       # interior nodes
    a = 0.5 * sigma**2 * j**2                  # coefficient of u_SS * dS^2
    c = 0.5 * r * j                            # coefficient of u_S * dS
    lower = a - c                              # multiplies u_{j-1}
    diag = -2 * a - r                          # multiplies u_j
    upper = a + c                              # multiplies u_{j+1}

    # A u = (lower, diag, upper) operator; CN: (I - dt/2 A) u^{n+1} = (I + dt/2 A) u^n
    ab = np.zeros((3, nS - 1))
    ab[0, 1:] = -0.5 * dt * upper[:-1]
    ab[1, :] = 1 - 0.5 * dt * diag
    ab[2, :-1] = -0.5 * dt * lower[1:]

    for n in range(nt):
        t_new = (n + 1) * dt
        rhs = u[1:-1] + 0.5 * dt * (lower * u[:-2] + diag * u[1:-1] + upper * u[2:])
        u_top_new = S_max - K * np.exp(-r * t_new)
        rhs[-1] += 0.5 * dt * upper[-1] * u_top_new   # implicit boundary term
        u[1:-1] = solve_banded((1, 1), ab, rhs)
        u[0], u[-1] = 0.0, u_top_new
    return np.interp(S, Sg, u)


# ---------- Example 5.2: Poisson counter, system of ODEs ----------
def poisson_ode(lam=2.0, tau=1.5, n=3, n_max=80):
    """Solve d/dtau u(n) = lam [u(n+1) - u(n)], u(0, n) = n^2 (Eq. 5.2 in tau).

    Truncated at n_max with u(n_max+1) approximated by the closed form
    (the truncation is far from the queried n, so its effect is negligible).
    """
    ns = np.arange(n_max + 1)

    def rhs(t, u):
        up = np.empty_like(u)
        up[:-1] = u[1:]
        up[-1] = (n_max + 1 + lam * t) ** 2 + lam * t
        return lam * (up - u)

    sol = solve_ivp(rhs, (0, tau), ns.astype(float) ** 2, rtol=1e-10, atol=1e-10)
    return sol.y[n, -1], (n + lam * tau) ** 2 + lam * tau


def poisson_mc(rng, lam=2.0, tau=1.5, n=3, n_paths=1_000_000):
    NT = n + rng.poisson(lam * tau, n_paths)
    return np.mean(NT.astype(float) ** 2)


# ---------- Example 5.3: risk-neutral optimal half-spread ----------
def optimal_half_spread(k=1.5, A=1.0):
    d = np.linspace(0.0, 5.0, 500_001)
    income_rate = 2 * A * np.exp(-k * d) * d
    return d[np.argmax(income_rate)], 1.0 / k


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Heat x^2 (MC | exact):", heat_quadratic(rng))
    args = (100.0, 100.0, 0.05, 0.2, 1.0)
    print("BS closed / MC / CN:", bs_closed(*args), bs_monte_carlo(rng, *args),
          bs_crank_nicolson(*args))
    print("Poisson ODE (ODE | exact):", poisson_ode(), " MC:", poisson_mc(rng))
    print("Half-spread (grid | 1/k):", optimal_half_spread())
