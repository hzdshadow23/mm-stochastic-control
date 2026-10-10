# 01 — Probability Foundations and Itô Calculus

**Goal of this note.** Build exactly the stochastic-calculus toolkit that every later note uses:
the Itô integral, Itô's lemma (for diffusions *and* Poisson jumps), and the **generator** of a process.
The generator is the single most important object here: in note 02 it turns expectations into PDEs, and in note 04 it becomes the Hamiltonian of the HJB equation.

**Prerequisites.** Measure-theoretic probability (σ-algebras, conditional expectation) at the level of a first graduate course. Nothing about stochastic processes is assumed.

---

## Notation

| Symbol | Meaning | Type |
|---|---|---|
| $(\Omega,\mathcal F,\mathbb P)$ | probability space: outcomes, events, probability measure | — |
| $\mathcal F_t$ | information available at time $t$ (filtration) | σ-algebra |
| $\mathbb E[\cdot]$, $\mathbb E[\cdot\mid\mathcal F_t]$ | expectation, conditional expectation given time-$t$ information | — |
| $W_t$ | standard Brownian motion | real-valued process |
| $\Delta t$, $\Delta W$ | small time step, Brownian increment over it | real |
| $H_t$ | integrand of a stochastic integral | adapted process |
| $X_t$ | an Itô process (solution of an SDE) | real-valued process |
| $b$, $\sigma$ | drift and volatility of $X$ | functions |
| $f(t,x)$ | smooth test function, $f\in C^{1,2}$ (once in $t$, twice in $x$) | function |
| $\partial_t f$, $\partial_x f$, $\partial_{xx} f$ | partial derivatives | functions |
| $N_t$ | Poisson process with intensity $\lambda$ | integer-valued process |
| $\lambda$ | jump intensity (expected jumps per unit time) | $>0$ |
| $\eta$ | jump size of $X$ when $N$ jumps | real |
| $\tilde N_t = N_t - \lambda t$ | compensated Poisson process | process |
| $\mathcal L$ | generator of $X$ | operator on functions |
| $[X]_t$ | quadratic variation of $X$ up to $t$ | process |

---

## 1. Setup & definitions

**Definition 1.1 (Filtration).** A family $\{\mathcal F_t\}_{t\ge 0}$ of σ-algebras with $\mathcal F_s\subseteq\mathcal F_t\subseteq\mathcal F$ for $s\le t$.
Read $\mathcal F_t$ as "everything that is known at time $t$." It only grows: information is never forgotten.

**Definition 1.2 (Adapted process).** A process $H_t$ is adapted if, for each $t$, $H_t$ is $\mathcal F_t$-measurable.
In words: the value of $H_t$ can be computed from what is known at time $t$. A trading strategy must be adapted — you cannot trade on tomorrow's price.

**Definition 1.3 (Martingale).** An adapted process $M_t$ with $\mathbb E|M_t|<\infty$ and
$$
\mathbb E[M_t\mid\mathcal F_s]=M_s\qquad(s\le t). \tag{1.1}
$$
A martingale is a "fair game": the best forecast of its future value is its current value. In particular $\mathbb E[M_t]=M_0$.

**Definition 1.4 (Brownian motion).** A process $W_t$ adapted to $\mathcal F_t$ such that
1. $W_0=0$ and $t\mapsto W_t$ is continuous;
2. for $s<t$, the increment $W_t-W_s$ is independent of $\mathcal F_s$;
3. $W_t-W_s\sim\mathcal N(0,\,t-s)$.

Two facts we will use constantly: $\mathbb E[\Delta W]=0$ and $\mathbb E[(\Delta W)^2]=\Delta t$, so $\Delta W$ is typically of size $\sqrt{\Delta t}$, which is **much larger** than $\Delta t$ when $\Delta t$ is small.

> **Intuition:** this $\sqrt{\Delta t}$ scaling is the whole reason stochastic calculus differs from ordinary calculus. A path that moves $\sqrt{\Delta t}$ in time $\Delta t$ has slope $1/\sqrt{\Delta t}\to\infty$: Brownian paths are nowhere differentiable, so $dW_t/dt$ does not exist and we cannot use the ordinary chain rule.

---

## 2. Quadratic variation: why $(dW)^2=dt$

**Roadmap.** We show that although each $(\Delta W)^2$ is random, their sum over a partition of $[0,t]$ is *deterministic* in the limit and equals $t$. This is the precise meaning of the slogan $(dW)^2=dt$.

Partition $[0,t]$ into $n$ equal steps $\Delta t=t/n$, with increments $\Delta W_i=W_{t_{i+1}}-W_{t_i}$. Define
$$
Q_n=\sum_{i=0}^{n-1}(\Delta W_i)^2. \tag{2.1}
$$

**Mean.** By Definition 1.4(3), $\mathbb E[(\Delta W_i)^2]=\Delta t$, so $\mathbb E[Q_n]=n\Delta t=t$.

**Variance.** The $\Delta W_i$ are independent, so the variances add. For $Z\sim\mathcal N(0,\Delta t)$, $\mathbb E[Z^4]=3\Delta t^2$, so $\mathrm{Var}(Z^2)=3\Delta t^2-\Delta t^2=2\Delta t^2$. Hence
$$
\mathrm{Var}(Q_n)=n\cdot 2\Delta t^2=\frac{2t^2}{n}\xrightarrow[n\to\infty]{}0. \tag{2.2}
$$

A random variable with fixed mean $t$ and variance tending to $0$ converges (in $L^2$) to the constant $t$. So
$$
[W]_t:=\lim_{n\to\infty}Q_n=t. \tag{2.3}
$$

**Recap.** The squared increments of $W$ add up to exactly $t$. For a smooth function $g$, the same sum $\sum(\Delta g)^2\approx\sum g'^2\Delta t^2\to 0$. This non-vanishing second-order term is what Itô's lemma accounts for.

**Multiplication table.** By the same argument (each term is a sum whose variance vanishes):

| $\times$ | $dt$ | $dW$ |
|---|---|---|
| $dt$ | $0$ | $0$ |
| $dW$ | $0$ | $dt$ |

For example, $\sum\Delta t\,\Delta W_i$ has mean $0$ and variance $n\Delta t^3\to0$, so $dt\,dW=0$.

---

## 3. The Itô integral

**Roadmap.** We want to define $\int_0^T H_t\,dW_t$ for adapted $H$. Since $W$ has no derivative, we cannot write $\int H\,\dot W\,dt$. Instead we use Riemann-type sums, with one essential choice: **evaluate $H$ at the left endpoint**.

**Definition 3.1 (Simple process).** $H$ is simple if $H_t=h_i$ for $t\in[t_i,t_{i+1})$, where each $h_i$ is $\mathcal F_{t_i}$-measurable and $\mathbb E[h_i^2]<\infty$. Define
$$
\int_0^T H_t\,dW_t:=\sum_i h_i\,(W_{t_{i+1}}-W_{t_i}). \tag{3.1}
$$

> **Intuition:** read $h_i$ as a position held over $[t_i,t_{i+1})$ and $\Delta W_i$ as the price change. Then (3.1) is trading P&L. The position must be chosen at the *start* of the interval (left endpoint), using only information available then. This is exactly the non-anticipation constraint of trading.

**Proposition 3.2 (Zero mean).** $\mathbb E\big[\int_0^T H\,dW\big]=0$.

*Proof.* For each term, condition on $\mathcal F_{t_i}$. Since $h_i$ is known at $t_i$ and $\Delta W_i$ is independent of $\mathcal F_{t_i}$ with mean $0$:
$$\mathbb E[h_i\Delta W_i]=\mathbb E\big[h_i\,\mathbb E[\Delta W_i\mid\mathcal F_{t_i}]\big]=\mathbb E[h_i\cdot 0]=0.\qquad\square$$

**Proposition 3.3 (Itô isometry).**
$$
\mathbb E\Big[\Big(\int_0^T H_t\,dW_t\Big)^2\Big]=\mathbb E\Big[\int_0^T H_t^2\,dt\Big]. \tag{3.2}
$$

*Proof.* Expand the square: $\big(\sum_i h_i\Delta W_i\big)^2=\sum_i h_i^2\Delta W_i^2+2\sum_{i<j}h_ih_j\Delta W_i\Delta W_j$.
- Diagonal terms: condition on $\mathcal F_{t_i}$: $\mathbb E[h_i^2\Delta W_i^2]=\mathbb E[h_i^2]\,\Delta t_i$.
- Cross terms ($i<j$): $h_ih_j\Delta W_i$ is known at $t_j$, and $\Delta W_j$ is independent of $\mathcal F_{t_j}$ with mean 0, so the expectation is $0$.

Summing gives $\sum_i\mathbb E[h_i^2]\Delta t_i=\mathbb E\int_0^T H_t^2\,dt$. $\square$

**Extension.** (3.2) says the map $H\mapsto\int H\,dW$ is an isometry from $L^2(\Omega\times[0,T])$ (adapted) into $L^2(\Omega)$. Simple processes are dense in the adapted $L^2$ processes, so the integral extends uniquely by continuity to every adapted $H$ with $\mathbb E\int_0^T H_t^2\,dt<\infty$. Propositions 3.2–3.3 continue to hold, and moreover $t\mapsto\int_0^t H\,dW$ is a martingale.

> **Caution:** if only $\int_0^T H_t^2dt<\infty$ almost surely (no expectation), the integral still exists but is only a *local* martingale and may have nonzero mean. This matters later: verification theorems in note 04 must check integrability before setting $\mathbb E\int H\,dW=0$.

### Worked example 3.4: $\int_0^T W_t\,dW_t$

**Roadmap.** Ordinary calculus would give $\tfrac12W_T^2$. We show the correct answer has an extra $-\tfrac12T$, which comes directly from quadratic variation.

Use the algebraic identity $a(b-a)=\tfrac12(b^2-a^2)-\tfrac12(b-a)^2$ with $a=W_{t_i}$, $b=W_{t_{i+1}}$:
$$
\sum_i W_{t_i}\Delta W_i=\tfrac12\sum_i\big(W_{t_{i+1}}^2-W_{t_i}^2\big)-\tfrac12\sum_i(\Delta W_i)^2.
$$
The first sum telescopes to $\tfrac12W_T^2$ (since $W_0=0$). The second is $\tfrac12Q_n\to\tfrac12T$ by (2.3). Therefore
$$
\int_0^T W_t\,dW_t=\tfrac12W_T^2-\tfrac12T. \tag{3.3}
$$

**Sanity checks.** Mean: $\tfrac12\mathbb E[W_T^2]-\tfrac12T=0$ ✓ (Prop. 3.2). Second moment: by (3.2), $\mathbb E\int_0^T W_t^2dt=\int_0^T t\,dt=T^2/2$; directly, $\mathrm{Var}(\tfrac12W_T^2)=\tfrac14\cdot2T^2=T^2/2$ ✓.

---

## 4. Itô processes and SDEs

**Definition 4.1 (Itô process).** A process of the form
$$
X_t=X_0+\int_0^t b_s\,ds+\int_0^t\sigma_s\,dW_s,\qquad\text{written}\quad dX_t=b_t\,dt+\sigma_t\,dW_t, \tag{4.1}
$$
with $b,\sigma$ adapted and integrable enough for both integrals to exist. The differential form is **only shorthand** for the integral form.

**Definition 4.2 (SDE).** When the coefficients depend on the current state, $b_t=b(t,X_t)$ and $\sigma_t=\sigma(t,X_t)$, equation (4.1) is a stochastic differential equation. If $b,\sigma$ are Lipschitz in $x$ (uniformly in $t$) with linear growth, a unique strong solution exists — the proof is Picard iteration, as for ODEs, with the Itô isometry replacing the ODE estimate.

**Quadratic variation of $X$.** From the multiplication table, $(dX)^2=\sigma^2(dW)^2=\sigma^2dt$, i.e. $[X]_t=\int_0^t\sigma_s^2ds$. The drift contributes nothing.

---

## 5. Itô's lemma (continuous case)

**Theorem 5.1 (Itô's lemma, 1-d).** Let $X$ satisfy (4.1) and $f\in C^{1,2}$. Then
$$
df(t,X_t)=\Big(\partial_tf+b_t\,\partial_xf+\tfrac12\sigma_t^2\,\partial_{xx}f\Big)dt+\sigma_t\,\partial_xf\,dW_t, \tag{5.1}
$$
with all derivatives evaluated at $(t,X_t)$.

**Derivation (the idea behind the proof).**
**Roadmap.** Taylor-expand $f$ to second order, then use the multiplication table to decide which terms survive.

Over a small step, with $\Delta X=b\,\Delta t+\sigma\,\Delta W$:
$$
\Delta f=\partial_tf\,\Delta t+\partial_xf\,\Delta X+\tfrac12\partial_{tt}f\,\Delta t^2+\partial_{tx}f\,\Delta t\,\Delta X+\tfrac12\partial_{xx}f\,(\Delta X)^2+\cdots
$$
Now classify each term by its order:
- $\Delta t^2$: order $\Delta t^2$ → vanishes when summed.
- $\Delta t\,\Delta X$: order $\Delta t^{3/2}$ → vanishes.
- $(\Delta X)^2=b^2\Delta t^2+2b\sigma\Delta t\Delta W+\sigma^2(\Delta W)^2$: only $\sigma^2(\Delta W)^2$ survives, and it sums to $\sigma^2\Delta t$ by Section 2.
- Third-order terms: order $\Delta t^{3/2}$ or smaller → vanish.

Keeping only the survivors:
$$
df=\partial_tf\,dt+\partial_xf\,(b\,dt+\sigma\,dW)+\tfrac12\partial_{xx}f\,\sigma^2dt,
$$
which is (5.1) after grouping. The rigorous proof makes "sums to" precise using (2.2) and a localization argument; the structure is exactly this.

**Recap.** Itô = ordinary chain rule **plus** the correction $\tfrac12\sigma^2\partial_{xx}f\,dt$. The correction is a drift: convexity of $f$ combined with noise produces a systematic trend (Jensen's inequality in differential form).

**Theorem 5.2 (Multi-dimensional).** Let $X\in\mathbb R^d$, $dX=b\,dt+\sigma\,dW$ with $W\in\mathbb R^m$ having independent components and $\sigma$ a $d\times m$ matrix. Then
$$
df=\Big(\partial_tf+b\cdot\nabla f+\tfrac12\operatorname{tr}\big(\sigma\sigma^\top\nabla^2f\big)\Big)dt+(\nabla f)^\top\sigma\,dW. \tag{5.2}
$$
Here $\nabla f$ is the gradient, $\nabla^2f$ the Hessian, and $\operatorname{tr}$ the trace. It follows from the same Taylor argument with $dW^i\,dW^j=\delta_{ij}\,dt$, so $dX^i\,dX^j=(\sigma\sigma^\top)_{ij}dt$.

**Product rule (special case).** For two Itô processes, take $f(x,y)=xy$ in (5.2):
$$
d(X_tY_t)=X_t\,dY_t+Y_t\,dX_t+dX_t\,dY_t. \tag{5.3}
$$
The last term (the *covariation*) is the new piece compared with ordinary calculus.

### Worked example 5.3: geometric Brownian motion

Let $dS=\mu S\,dt+\sigma S\,dW$ with constants $\mu,\sigma$. Apply (5.1) to $f(x)=\ln x$, using $\partial_xf=1/x$ and $\partial_{xx}f=-1/x^2$:
$$
d\ln S=\frac1S(\mu S\,dt+\sigma S\,dW)+\frac12\Big(-\frac1{S^2}\Big)\sigma^2S^2dt=\Big(\mu-\tfrac12\sigma^2\Big)dt+\sigma\,dW.
$$
The right side has constant coefficients, so integrate directly:
$$
S_T=S_0\exp\Big(\big(\mu-\tfrac12\sigma^2\big)T+\sigma W_T\Big). \tag{5.4}
$$
Check: $\mathbb E[e^{\sigma W_T}]=e^{\sigma^2T/2}$ (Gaussian MGF), so $\mathbb E[S_T]=S_0e^{\mu T}$ ✓.

### Worked example 5.4: CARA utility of a stock position (preview of market making)

This example is a direct preview of the Avellaneda–Stoikov reservation price (note 12).

**Setup.** Mid price $dS_t=\sigma\,dW_t$ (arithmetic BM, as in Avellaneda–Stoikov). You hold a fixed inventory of $q$ units and cash $x$, with no trading. Your terminal wealth is $x+qS_T$. Your utility is CARA, $U(w)=-e^{-\gamma w}$, with risk aversion $\gamma>0$.

**Compute the expected utility.** $S_T=s+\sigma W_T\sim\mathcal N(s,\sigma^2T)$, so $x+qS_T\sim\mathcal N(x+qs,\ q^2\sigma^2T)$. For $Z\sim\mathcal N(m,v)$, $\mathbb E[e^{-\gamma Z}]=e^{-\gamma m+\gamma^2v/2}$. Hence
$$
\mathbb E\big[-e^{-\gamma(x+qS_T)}\big]=-\exp\Big(-\gamma\big(x+qs-\tfrac12\gamma q^2\sigma^2T\big)\Big). \tag{5.5}
$$

**Certainty equivalent.** The term in the inner bracket is the sure wealth you would accept instead of the risky position:
$$
\mathrm{CE}=x+qs-\tfrac12\gamma q^2\sigma^2T. \tag{5.6}
$$
Inventory risk costs $\tfrac12\gamma q^2\sigma^2T$, which is **quadratic in $q$**.

**Reservation price.** How much is one more unit worth to you? The change in CE from $q$ to $q+1$ is
$$
\mathrm{CE}(q+1)-\mathrm{CE}(q)=s-\tfrac12\gamma\sigma^2T\big((q+1)^2-q^2\big)=s-\gamma\sigma^2T\big(q+\tfrac12\big).
$$
Up to the $\tfrac12$ (a buy/sell averaging detail treated in note 12), this is the Avellaneda–Stoikov reservation price $r=s-q\gamma\sigma^2T$.

**Numbers.** $s=100$, $\sigma=2$, $T=1$, $\gamma=0.1$, $q=3$, $x=0$:
- Inventory penalty: $\tfrac12\cdot0.1\cdot9\cdot4\cdot1=1.8$, so $\mathrm{CE}=300-1.8=298.2$.
- Value of a 4th unit: $100-0.1\cdot4\cdot1\cdot3.5=98.6$.

> **Intuition:** a long market maker values extra inventory *below* mid, because each extra unit adds more risk than the last (risk is quadratic in $q$). That is why a long market maker skews quotes downward.

We could also get (5.5) from Itô's lemma: applying (5.1) to $f(t,S)=-\exp(-\gamma(x+qS)+\tfrac12\gamma^2q^2\sigma^2(T-t))$ shows the $dt$-term is exactly zero, so $f(t,S_t)$ is a martingale. That "choose $f$ so the drift vanishes" step is the Feynman–Kac idea of note 02.

---

## 6. Poisson processes and Itô's lemma with jumps

Market-making fills are discrete events: they are modeled with Poisson processes, not Brownian motion. The HJB of note 12 has **jump terms**, which come from this section.

**Definition 6.1 (Poisson process).** A counting process $N_t$ with $N_0=0$, independent increments, and $N_t-N_s\sim\mathrm{Poisson}(\lambda(t-s))$. Equivalently, on a small interval:
$$
\mathbb P(\Delta N=1)=\lambda\Delta t+o(\Delta t),\quad\mathbb P(\Delta N=0)=1-\lambda\Delta t+o(\Delta t),\quad\mathbb P(\Delta N\ge2)=o(\Delta t). \tag{6.1}
$$
So $\mathbb E[\Delta N]=\lambda\Delta t$: the intensity $\lambda$ is the expected number of jumps per unit time.

**Compensated process.** $\tilde N_t:=N_t-\lambda t$ is a martingale: its increment has mean $\lambda\Delta t-\lambda\Delta t=0$ and is independent of the past. Writing $dN=\lambda\,dt+d\tilde N$ separates the predictable part (rate $\lambda$) from pure noise.

**Multiplication rule for jumps.** $\Delta N\in\{0,1\}$ on a small interval, so $(\Delta N)^2=\Delta N$, i.e. $(dN)^2=dN$. Unlike $(dW)^2=dt$, this is **not small**: a jump is an $O(1)$ change, so a Taylor expansion would need all orders. That is why the jump part of Itô's lemma is written as a full difference instead.

**Theorem 6.2 (Itô's lemma with jumps).** Let
$$
dX_t=b_t\,dt+\sigma_t\,dW_t+\eta\,dN_t, \tag{6.2}
$$
so $X$ diffuses and, at each jump time of $N$, jumps by $\eta$. For $f\in C^{1,2}$,
$$
df(t,X_t)=\Big(\partial_tf+b\,\partial_xf+\tfrac12\sigma^2\partial_{xx}f\Big)dt+\sigma\,\partial_xf\,dW_t+\big[f(t,X_{t-}+\eta)-f(t,X_{t-})\big]\,dN_t. \tag{6.3}
$$
Here $X_{t-}$ is the value just before a possible jump at $t$.

**Derivation.** Between jumps, $X$ is an Itô process and (5.1) applies. At a jump time, $X$ moves from $X_{t-}$ to $X_{t-}+\eta$, so $f$ changes by exactly $f(X_{t-}+\eta)-f(X_{t-})$, no Taylor expansion needed. Adding the two contributions gives (6.3).

**Expected change.** Substitute $dN=\lambda\,dt+d\tilde N$ into (6.3). The $dW$ and $d\tilde N$ terms are martingale increments (mean zero). What remains is the expected rate of change:
$$
\frac{\mathbb E[df\mid\mathcal F_t]}{dt}=\partial_tf+b\,\partial_xf+\tfrac12\sigma^2\partial_{xx}f+\lambda\big[f(t,x+\eta)-f(t,x)\big]. \tag{6.4}
$$

> **Intuition:** the jump term reads "rate of jumps × change in $f$ if a jump happens." In market making: rate of fills × change in value if filled. With fill rate $\lambda(\delta)=Ae^{-k\delta}$, this is precisely the term $\lambda(\delta^b)\,[u(q+1)-u(q)]$ in the Avellaneda–Stoikov HJB.

### Worked example 6.3: Poisson moment generating function

Let $\phi(t)=\mathbb E[e^{\theta N_t}]$. Apply (6.4) to $f(x)=e^{\theta x}$ with $b=\sigma=0$, $\eta=1$: the jump term is $\lambda(e^{\theta(x+1)}-e^{\theta x})=\lambda(e^\theta-1)e^{\theta x}$. Taking expectations,
$$
\phi'(t)=\lambda(e^\theta-1)\phi(t),\quad\phi(0)=1\ \Longrightarrow\ \phi(t)=\exp\big(\lambda t(e^\theta-1)\big). \tag{6.5}
$$
This matches the known Poisson MGF, confirming (6.4).

---

## 7. The generator and Dynkin's formula

**Definition 7.1 (Generator).** For the process (6.2) with $b=b(t,x)$, $\sigma=\sigma(t,x)$, define the operator acting on $f(x)$:
$$
\mathcal Lf(x)=b\,\partial_xf+\tfrac12\sigma^2\partial_{xx}f+\lambda\big[f(x+\eta)-f(x)\big]. \tag{7.1}
$$
It is the expected instantaneous rate of change of $f(X_t)$, given $X_t=x$:
$\mathcal Lf(x)=\lim_{h\to0}\frac{\mathbb E[f(X_{t+h})\mid X_t=x]-f(x)}{h}$.
For a pure diffusion, drop the jump term; in $d$ dimensions use (5.2).

**Theorem 7.2 (Dynkin's formula).** Integrate (6.3) from $t$ to $T$ and take expectations. If the stochastic-integral terms are true martingales (e.g. $\mathbb E\int\sigma^2(\partial_xf)^2ds<\infty$ and $f$ is bounded on jumps), they have zero mean, leaving
$$
\mathbb E\big[f(T,X_T)\mid X_t=x\big]=f(t,x)+\mathbb E\Big[\int_t^T(\partial_s+\mathcal L)f(s,X_s)\,ds\ \Big|\ X_t=x\Big]. \tag{7.2}
$$

**Recap and why it matters.** Dynkin's formula is the bridge from stochastic processes to PDEs:
- If $(\partial_t+\mathcal L)f=0$, then $f(t,x)=\mathbb E[f(T,X_T)\mid X_t=x]$. This is **Feynman–Kac** (note 02).
- If we also choose a control to make $(\partial_t+\mathcal L^\alpha)v$ as large as possible, we get the **HJB equation** (note 04).

---

## 8. Numerical check

Script: `code/ito_checks.py`; tests: `code/tests/test_01_ito.py` (run `python -m pytest code/tests -q`). Each test compares a closed form from this note with simulation:

| Check | Closed form | Equation |
|---|---|---|
| Quadratic variation | $Q_n\to T$, $\mathrm{Var}(Q_n)=2T^2/n$ | (2.2)–(2.3) |
| $\int W\,dW$ | left-point sum $\to\tfrac12W_T^2-\tfrac12T$ pathwise | (3.3) |
| Itô isometry | $\mathbb E[(\int_0^TW\,dW)^2]=T^2/2$ | (3.2) |
| GBM | $\mathbb E[S_T]=S_0e^{\mu T}$ | (5.4) |
| CARA certainty equivalent | $298.2$ for the numbers in Ex. 5.4 | (5.6) |
| Poisson MGF | $\exp(\lambda t(e^\theta-1))$ | (6.5) |

---

## 9. Pitfalls

1. **Left endpoint matters.** Using the midpoint in (3.1) gives the *Stratonovich* integral, under which the ordinary chain rule holds but the zero-mean property fails. Finance uses Itô because strategies are non-anticipating.
2. **Local vs. true martingale.** $\int H\,dW$ has zero mean only under integrability. Dropping it without checking is the most common gap in HJB verification arguments.
3. **$(dN)^2=dN$, not $0$.** Jumps are $O(1)$; never Taylor-expand across a jump. Use the full difference $f(x+\eta)-f(x)$.
4. **Drift correction direction.** $\tfrac12\sigma^2\partial_{xx}f$ is positive for convex $f$: $\mathbb E[S_T]$ grows at rate $\mu$, while the median of GBM grows at $\mu-\tfrac12\sigma^2$.

---

## 中文小结

- **基础概念**：信息流 $\mathcal F_t$ 代表"$t$ 时刻已知的信息"；适应过程不使用未来信息；鞅是"公平游戏"，$\mathbb E[M_t\mid\mathcal F_s]=M_s$。
- **二次变差**：布朗运动增量量级为 $\sqrt{\Delta t}$，平方和收敛到确定值 $t$，即 $(dW)^2=dt$；这是随机微积分区别于普通微积分的根本原因。
- **Itô 积分**：取左端点（对应交易策略不能预知未来），均值为零，满足 Itô 等距 $\mathbb E[(\int H\,dW)^2]=\mathbb E\int H^2dt$；例：$\int_0^TW\,dW=\tfrac12W_T^2-\tfrac12T$。
- **Itô 引理**：普通链式法则加修正项 $\tfrac12\sigma^2\partial_{xx}f\,dt$；带跳时增加 $[f(x+\eta)-f(x)]\,dN$，期望变化率为"跳跃强度 × 跳跃带来的变化"——这正是做市 HJB 中的成交项。
- **做市预览**：CARA 效用下持有 $q$ 单位的风险成本为 $\tfrac12\gamma q^2\sigma^2T$（二次），多头时额外一单位的价值低于中间价，因此报价下移。例：$q=3$ 时确定性等价为 298.2。
- **生成元与 Dynkin 公式**：$\mathcal L$ 是 $f(X_t)$ 的期望瞬时变化率；Dynkin 公式连接期望与 PDE，是 Feynman–Kac（第 02 篇）和 HJB（第 04 篇）的出发点。
