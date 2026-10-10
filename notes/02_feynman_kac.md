# 02 — The Feynman–Kac Formula

**Goal of this note.** Prove that a conditional expectation of a functional of an SDE solves a linear parabolic PDE, and conversely. This is the *linear, uncontrolled* version of everything that follows:

| This note | Later |
|---|---|
| $u(t,x)=\mathbb E[\,\cdot\mid X_t=x]$ | value function $v(t,x)=\sup_\alpha\mathbb E[\,\cdot\,]$ (note 03) |
| linear PDE $\partial_tu+\mathcal Lu=0$ | HJB $\partial_tv+\sup_\alpha\mathcal L^\alpha v=0$ (note 04) |
| proof via Itô + martingale | verification theorem (note 04) |
| linear: $u\leftrightarrow\mathbb E$ | nonlinear: $v\leftrightarrow$ BSDE (note 08) |

If this note is clear, the HJB derivation in note 04 is the same argument with one "sup" added.

**Prerequisites.** Note 01: Itô's lemma (5.1), the jump version (6.3), the generator (7.1), Dynkin's formula (7.2).

---

## Notation

| Symbol | Meaning | Type |
|---|---|---|
| $X_s$, $s\in[t,T]$ | state process, started at $X_t=x$ | real-valued process |
| $b(s,x)$, $\sigma(s,x)$ | drift and volatility of $X$ | functions, Lipschitz in $x$ |
| $\mathcal L$ | generator: $\mathcal L\varphi=b\,\partial_x\varphi+\tfrac12\sigma^2\partial_{xx}\varphi$ (+ jump term in §5) | operator |
| $g(x)$ | terminal payoff | function |
| $f(s,x)$ | running payoff rate (paid per unit time) | function |
| $r(s,x)\ge0$ | discount (killing) rate | function |
| $D_{t,s}=\exp\!\big(-\int_t^s r(\tau,X_\tau)\,d\tau\big)$ | discount factor from $t$ to $s$ | process |
| $u(t,x)$ | solution of the PDE / the expectation | function |
| $\tau=T-t$ | time to maturity | $\ge0$ |
| $\mathbb E_{t,x}[\cdot]$ | shorthand for $\mathbb E[\cdot\mid X_t=x]$ | — |

---

## 1. Statement

**Theorem 1.1 (Feynman–Kac).** Consider the SDE
$$
dX_s=b(s,X_s)\,ds+\sigma(s,X_s)\,dW_s,\qquad X_t=x, \tag{1.1}
$$
and the terminal-value problem
$$
\partial_tu+b\,\partial_xu+\tfrac12\sigma^2\partial_{xx}u-r\,u+f=0\quad\text{on }[0,T)\times\mathbb R,\qquad u(T,x)=g(x). \tag{1.2}
$$
Assume:
- **(A1)** $b,\sigma$ are Lipschitz in $x$ with linear growth, so (1.1) has a unique solution with finite moments of all orders;
- **(A2)** $r\ge0$ is bounded below, and $f,g$ have polynomial growth;
- **(A3)** $u\in C^{1,2}$ solves (1.2) and has polynomial growth in $x$.

Then
$$
\boxed{\ u(t,x)=\mathbb E_{t,x}\Big[D_{t,T}\,g(X_T)+\int_t^TD_{t,s}\,f(s,X_s)\,ds\Big].\ } \tag{1.3}
$$

**How to read it.** $u(t,x)$ is the present value, at time $t$ in state $x$, of:
- a terminal payoff $g$ received at $T$,
- plus a cash flow paid at rate $f$ over $[t,T]$,
- all discounted at rate $r$.

The PDE (1.2) is solved **backward** from the terminal condition at $T$, matching the fact that payoffs are known at maturity and values are needed today.

> **Intuition for a rates quant:** this is exactly why a bond or swap price satisfies a pricing PDE. With $X$ the short rate, $r(s,x)=x$, $g=1$, and $f=0$, (1.3) is the zero-coupon bond price $\mathbb E[e^{-\int r}]$ and (1.2) is the term-structure PDE you already use.

---

## 2. Proof: PDE ⟹ expectation

**Roadmap.** Build a process $M_s$ from $u$ that (i) equals $u(t,x)$ at $s=t$, (ii) equals the quantity inside the expectation in (1.3) at $s=T$, and (iii) has zero drift by the PDE. Then $M$ is a martingale and $\mathbb E[M_T]=M_t$ is exactly (1.3).

**Step 1: define the candidate martingale.** For $s\in[t,T]$, let
$$
M_s=D_{t,s}\,u(s,X_s)+\int_t^sD_{t,\tau}\,f(\tau,X_\tau)\,d\tau. \tag{2.1}
$$
Check the endpoints:
- At $s=t$: $D_{t,t}=1$ and the integral is empty, so $M_t=u(t,x)$.
- At $s=T$: $u(T,X_T)=g(X_T)$, so $M_T$ is the expression inside the expectation in (1.3).

**Step 2: differentiate the discount factor.** $D_{t,s}$ is an ordinary (pathwise differentiable) function of $s$, because its exponent is an ordinary $d\tau$-integral:
$$
dD_{t,s}=-r(s,X_s)\,D_{t,s}\,ds. \tag{2.2}
$$
It has no $dW$ part, so its covariation with anything is zero.

**Step 3: Itô on $u(s,X_s)$.** By note 01 eq. (5.1),
$$
du(s,X_s)=\big(\partial_su+\mathcal Lu\big)ds+\sigma\,\partial_xu\,dW_s. \tag{2.3}
$$

**Step 4: product rule.** By note 01 eq. (5.3), $d(D\,u)=D\,du+u\,dD+dD\,du$. The covariation $dD\,du$ is zero (Step 2). Using (2.2)–(2.3):
$$
d(D_{t,s}u)=D_{t,s}\big(\partial_su+\mathcal Lu-r\,u\big)ds+D_{t,s}\,\sigma\,\partial_xu\,dW_s.
$$

**Step 5: add the running payoff.** The integral term in (2.1) contributes $D_{t,s}f\,ds$. So
$$
dM_s=D_{t,s}\underbrace{\big(\partial_su+\mathcal Lu-r\,u+f\big)}_{=\,0\ \text{by (1.2)}}ds+D_{t,s}\,\sigma\,\partial_xu\,dW_s=D_{t,s}\,\sigma\,\partial_xu\,dW_s. \tag{2.4}
$$

**Step 6: the stochastic integral is a true martingale.** By note 01 §3, we need $\mathbb E\int_t^T D^2\sigma^2(\partial_xu)^2ds<\infty$. Under (A1)–(A3), $D\le e^{c(T-t)}$ (with $-c$ the lower bound of $r$; $c=0$ when $r\ge0$), $\sigma^2$ has quadratic growth, $\partial_xu$ is assumed to have polynomial growth (a standard part of (A3); it holds for all examples below), and $X$ has finite moments of all orders. So the integrand is integrable.

> **Remark:** this is the step where the growth assumptions are actually used. Without them, the integral is only a local martingale and (1.3) can fail (see Pitfall 2).

**Step 7: take expectations.** A martingale has constant expectation, so $\mathbb E_{t,x}[M_T]=M_t=u(t,x)$, which is (1.3). $\square$

**Recap.** The PDE is precisely the condition that the "discounted value plus accumulated cash flows" has no drift. Pricing = finding the function that makes this process a martingale.

---

## 3. Converse (heuristic): expectation ⟹ PDE

**Roadmap.** Now go the other way: *define* $u$ by the expectation (1.3) and derive the PDE. The argument below is formal (it assumes $u$ is smooth), but it is exactly the dynamic programming argument of note 03 with the "sup" removed, so it is worth seeing carefully.

For simplicity take $r=0$ and $f=0$, so $u(t,x)=\mathbb E_{t,x}[g(X_T)]$.

**Step 1: tower property + Markov property.** Fix a small $h>0$. Conditioning first on the information at $t+h$:
$$
u(t,x)=\mathbb E_{t,x}\Big[\mathbb E\big[g(X_T)\mid\mathcal F_{t+h}\big]\Big].
$$
Because $X$ solves an SDE, it is Markov: the future after $t+h$ depends on the past only through $X_{t+h}$. Hence $\mathbb E[g(X_T)\mid\mathcal F_{t+h}]=u(t+h,X_{t+h})$ and
$$
u(t,x)=\mathbb E_{t,x}\big[u(t+h,X_{t+h})\big]. \tag{3.1}
$$
In words: today's value is the expected value tomorrow. **This is the dynamic programming principle without control.**

**Step 2: expand with Dynkin.** Apply note 01 eq. (7.2) to $u$ on $[t,t+h]$:
$$
\mathbb E_{t,x}\big[u(t+h,X_{t+h})\big]=u(t,x)+\mathbb E_{t,x}\Big[\int_t^{t+h}(\partial_s+\mathcal L)u(s,X_s)\,ds\Big].
$$

**Step 3: combine and shrink $h$.** Substituting into (3.1), the $u(t,x)$ terms cancel:
$$
0=\mathbb E_{t,x}\Big[\int_t^{t+h}(\partial_s+\mathcal L)u(s,X_s)\,ds\Big].
$$
Divide by $h$ and let $h\to0$. By continuity of the integrand and of paths, the average over $[t,t+h]$ converges to its value at $(t,x)$:
$$
(\partial_t+\mathcal L)u(t,x)=0. \tag{3.2}
$$
Also $u(T,x)=\mathbb E_{T,x}[g(X_T)]=g(x)$. With $r,f$ restored, the same steps give (1.2).

**Recap.** Expectation ⟹ (DPP) ⟹ (Itô/Dynkin) ⟹ PDE. In note 04, (3.1) becomes $v(t,x)=\sup_\alpha\mathbb E[v(t+h,X_{t+h})]$ and (3.2) becomes $\sup_\alpha(\partial_t+\mathcal L^\alpha)v=0$.

> **What makes it rigorous:** the gap is assuming $u\in C^{1,2}$. For uniformly parabolic problems ($\sigma^2\ge\epsilon>0$) with smooth data, parabolic regularity theory provides it. In degenerate or controlled problems it can fail, which is why viscosity solutions (note 05) are needed.

---

## 4. Worked examples (diffusion)

### Example 4.1: heat equation with quadratic payoff

Take $b=0$, $\sigma=1$, $r=f=0$, $g(x)=x^2$. The PDE is $\partial_tu+\tfrac12\partial_{xx}u=0$, $u(T,x)=x^2$.

**Via expectation.** $X_T=x+(W_T-W_t)\sim\mathcal N(x,\tau)$ with $\tau=T-t$, so
$$
u(t,x)=\mathbb E[X_T^2]=x^2+\tau. \tag{4.1}
$$
**Check the PDE.** $\partial_tu=-1$ (since $\tau=T-t$), $\partial_{xx}u=2$, so $-1+\tfrac12\cdot2=0$ ✓. Terminal: $\tau=0$ gives $x^2$ ✓.

> **Intuition:** the extra $\tau$ is the variance accumulated before maturity. A convex payoff gains value from noise — the same $\tfrac12\sigma^2\partial_{xx}$ term as in Itô's lemma.

### Example 4.2: Black–Scholes call

Take $dS=rS\,dt+\sigma S\,dW$ (risk-neutral GBM), constant discount rate $r$, $f=0$, $g(S)=(S-K)^+$. The PDE (1.2) is the Black–Scholes equation
$$
\partial_tu+rS\,\partial_Su+\tfrac12\sigma^2S^2\partial_{SS}u-ru=0, \tag{4.2}
$$
and (1.3) gives $u=e^{-r\tau}\mathbb E[(S_T-K)^+]$. Evaluating the Gaussian integral yields the usual formula
$$
u=S\,\Phi(d_1)-Ke^{-r\tau}\Phi(d_2),\qquad d_{1,2}=\frac{\ln(S/K)+(r\pm\tfrac12\sigma^2)\tau}{\sigma\sqrt\tau}, \tag{4.3}
$$
where $\Phi$ is the standard normal CDF.

**Numbers.** $S=K=100$, $r=0.05$, $\sigma=0.2$, $\tau=1$: $d_1=0.35$, $d_2=0.15$, $u=10.4506$.
The test checks this three ways: closed form (4.3), Monte Carlo of (1.3), and a Crank–Nicolson finite-difference solve of (4.2).

### Example 4.3: CARA utility of inventory (from note 01, Ex. 5.4)

Take $dS=\sigma\,dW$, fixed $q$ and $x$, $g(S)=-e^{-\gamma(x+qS)}$, $r=f=0$. Feynman–Kac says $u(t,s)=\mathbb E[g(S_T)]$ solves $\partial_tu+\tfrac12\sigma^2\partial_{ss}u=0$.

**Verify the closed form from note 01 eq. (5.5).** Let
$$
u(t,s)=-\exp\big(-\gamma(x+qs)+\tfrac12\gamma^2q^2\sigma^2\tau\big).
$$
Then $\partial_tu=-\tfrac12\gamma^2q^2\sigma^2\,u$ (the exponent decreases as $t$ increases) and $\partial_{ss}u=\gamma^2q^2\,u$. So $\partial_tu+\tfrac12\sigma^2\partial_{ss}u=u\,(-\tfrac12\gamma^2q^2\sigma^2+\tfrac12\gamma^2q^2\sigma^2)=0$ ✓.

> **Why this matters:** in note 12, Avellaneda–Stoikov guesses the value function in exactly this exponential form. The guess works because CARA turns the PDE into a problem where $x$ and $s$ factor out.

---

## 5. Feynman–Kac with jumps

**Setup.** Add a Poisson jump as in note 01 eq. (6.2): $dX=b\,dt+\sigma\,dW+\eta\,dN$, where $N$ has intensity $\lambda$. The generator is now (note 01 eq. 7.1)
$$
\mathcal L\varphi(x)=b\,\partial_x\varphi+\tfrac12\sigma^2\partial_{xx}\varphi+\lambda\big[\varphi(x+\eta)-\varphi(x)\big]. \tag{5.1}
$$

**Theorem 5.1.** Under analogous assumptions (with $u$ also of polynomial growth so the jump term is integrable), Theorem 1.1 holds verbatim with this $\mathcal L$.

**Proof sketch.** Repeat §2 with note 01 eq. (6.3) instead of (5.1). The jump term $[u(X_{s-}+\eta)-u(X_{s-})]dN_s$ splits as $\lambda[\cdots]ds+[\cdots]d\tilde N_s$. The $ds$ part is the new term in $\mathcal L$ and is cancelled by the PDE; the $d\tilde N$ part is a martingale increment with zero mean. $\square$

**Note the structure.** The PDE is now a **nonlocal** equation: it couples $u(t,x)$ with $u(t,x+\eta)$. If $X$ lives on a lattice (like an integer inventory), the PDE becomes a **system of ODEs**, one per lattice point. This is exactly the structure of the GLFT market-making model (note 13).

### Example 5.2: Poisson counter — a system of ODEs

Take $X=N$ (pure counting process, $b=\sigma=0$, $\eta=1$), $g(n)=n^2$, $r=f=0$. The state $n\in\{0,1,2,\dots\}$ is integer, so (1.2) becomes, for each $n$,
$$
\frac{d}{dt}u(t,n)+\lambda\big[u(t,n+1)-u(t,n)\big]=0,\qquad u(T,n)=n^2. \tag{5.2}
$$

**Via expectation.** $N_T-N_t\sim\mathrm{Poisson}(\lambda\tau)$, with mean and variance $\lambda\tau$, so
$$
u(t,n)=\mathbb E[(n+N_T-N_t)^2]=(n+\lambda\tau)^2+\lambda\tau. \tag{5.3}
$$

**Check the ODE.** $\frac{d}{dt}u=-\lambda\big[2(n+\lambda\tau)+1\big]$ (chain rule with $d\tau/dt=-1$), and
$\lambda[u(n+1)-u(n)]=\lambda\big[(n+1+\lambda\tau)^2-(n+\lambda\tau)^2\big]=\lambda\big[2(n+\lambda\tau)+1\big]$. Sum $=0$ ✓.

**Numbers.** $\lambda=2$, $\tau=1.5$, $n=3$: $u=(3+3)^2+3=39$. The test solves the truncated ODE system (5.2) numerically and compares with 39 and with Monte Carlo.

### Example 5.3: expected spread income from passive quoting (market-making preview)

A market maker posts a bid and an ask, each at distance $\delta$ from mid. Assume fills on each side arrive as independent Poisson processes with intensity $\lambda(\delta)=Ae^{-k\delta}$, and each fill earns $\delta$ relative to mid. Ignoring inventory risk, the expected spread income over $[t,T]$ is (Feynman–Kac with $f=2\lambda(\delta)\delta$, $g=0$, $r=0$):
$$
u(t)=2\,Ae^{-k\delta}\,\delta\,\tau. \tag{5.4}
$$
Maximizing over $\delta$: $\frac{d}{d\delta}\big(\delta e^{-k\delta}\big)=e^{-k\delta}(1-k\delta)=0$ gives $\delta^*=1/k$.

**Numbers.** $k=1.5$ gives $\delta^*=0.667$. Compare with the Avellaneda–Stoikov half-spread term $\frac1\gamma\ln(1+\frac\gamma k)=0.645$ for $\gamma=0.1$ (from the earlier chat example); as $\gamma\to0$ it tends to $1/k$.

> **Intuition:** a risk-neutral market maker faces the monopolist's trade-off: quoting wider earns more per fill but gets fewer fills. The optimum $\delta^*=1/k$ depends only on how fast fills decay. Risk aversion ($\gamma>0$) then shrinks the spread term slightly and adds inventory skew — the content of notes 12–13.

---

## 6. Numerical check

Script: `code/feynman_kac_checks.py`; tests: `code/tests/test_02_feynman_kac.py`.

| Check | Methods compared | Expected |
|---|---|---|
| Heat, $g=x^2$ | MC vs (4.1) | $x^2+\tau$ |
| Black–Scholes | closed form (4.3) vs MC vs Crank–Nicolson PDE | $10.4506$ |
| Poisson ODE system | ODE solve of (5.2) vs (5.3) vs MC | $39$ |
| Optimal risk-neutral half-spread | grid search of (5.4) vs $1/k$ | $0.667$ |

The Crank–Nicolson solver is the PDE side of Feynman–Kac, and the Monte Carlo is the expectation side; their agreement is the theorem in action.

---

## 7. Pitfalls

1. **Time direction.** (1.2) is a *backward* equation: it has a terminal condition and is well-posed when solved from $T$ to $t$. Written in $\tau=T-t$, it is a standard forward heat-type equation. The *forward* equation for the density of $X$ is the Fokker–Planck equation (note 14), which is the adjoint of $\partial_t+\mathcal L$.
2. **Uniqueness needs growth conditions.** The heat equation has nonzero solutions with zero initial data that grow like $e^{cx^2}$ (Tychonoff's example). Feynman–Kac picks the unique solution of polynomial growth; this is why (A3) is needed in Step 6.
3. **$u$ is a function of the starting point.** $u(t,x)$ is not "the value along a path"; it is a deterministic function evaluated at the starting state. $u(s,X_s)$ along a path is what becomes a martingale.
4. **Sign of the potential.** $-ru$ in (1.2) corresponds to discounting $e^{-\int r}$. A source term $+f$ corresponds to cash received. Sign mistakes here propagate into every HJB later.

---

## 中文小结

- **定理**：PDE $\partial_tu+\mathcal Lu-ru+f=0$，$u(T)=g$ 的解等于条件期望 $\mathbb E_{t,x}[D\,g(X_T)+\int D\,f\,ds]$（贴现的终值收益 + 贴现的流动收益）。这就是利率衍生品定价 PDE 的来源。
- **证明（PDE ⟹ 期望）**：构造 $M_s=D\,u(s,X_s)+\int D\,f$，用 Itô 引理，PDE 恰好使漂移项为零，所以 $M$ 是鞅，取期望即得。增长条件保证随机积分是真鞅。
- **反方向（期望 ⟹ PDE）**：塔性质 + 马尔可夫性给出 $u(t,x)=\mathbb E[u(t+h,X_{t+h})]$（无控制的动态规划原理），再用 Dynkin 公式令 $h\to0$ 得到 PDE。第 04 篇 HJB 推导就是在此基础上加一个 sup。
- **例子**：热方程 $x^2+\tau$；Black–Scholes $10.4506$（闭式解、蒙特卡洛、Crank–Nicolson 三方一致）；CARA 指数形式满足 PDE（Avellaneda–Stoikov 猜解的基础）。
- **带跳情形**：生成元多出 $\lambda[u(x+\eta)-u(x)]$，方程变为非局部；状态为整数时变成 ODE 方程组——正是 GLFT 做市模型的结构。
- **做市预览**：风险中性下最优半价差 $\delta^*=1/k$（每笔收益与成交频率的权衡），$k=1.5$ 时为 0.667；风险厌恶会略微缩小价差并加入库存偏移。
