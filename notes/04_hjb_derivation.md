# 04 — The Hamilton–Jacobi–Bellman Equation and Verification

**Goal of this note.** Turn the dynamic programming principle of note 03 into a PDE, the **Hamilton–Jacobi–Bellman (HJB) equation**, and prove the **verification theorem**: a smooth solution of HJB with the right growth *is* the value function, and its pointwise maximizer *is* an optimal feedback control. Then extend both to controlled **jump** intensities, which is the form used for Avellaneda–Stoikov (note 12).

The logic is the same as note 02, with one $\sup$ added:

| Note 02 (no control) | This note |
|---|---|
| DPP without control (3.1) + Dynkin ⟹ $\partial_tu+\mathcal Lu=0$ (§3) | DPP (note 03, 2.1) + Dynkin ⟹ $\partial_tv+\sup_a\{\mathcal L^av+f\}=0$ (§3.1) |
| PDE solution ⟹ expectation, via Itô + martingale (§2) | HJB solution ⟹ value function, via Itô + supermartingale (§3.3) |

**Prerequisites.** Note 01: Itô (5.1)–(5.2), jumps (6.3)–(6.4), Dynkin (7.2), the local-martingale caution after (3.2). Note 02: Feynman–Kac (1.3), proof steps (2.1)–(2.4). Note 03: setup (1.1)–(1.7), (H1)–(H4), Lemma 1.2, DPP (2.1)–(2.3), Corollary 2.2, the example of §4.

---

## Notation

| Symbol | Meaning | Type |
|---|---|---|
| $X_s$, $\alpha_s$, $A$, $\mathcal A_t$ | state, control, control set, admissible controls (note 03 §1) | — |
| $b,\sigma,f,g$ | drift, volatility, running reward, terminal reward (note 03 (1.1), (1.6)) | functions |
| $J$, $v$ | reward of a control, value function (note 03 (1.6)–(1.7)) | functions |
| $\mathcal L^a$ | generator of the state when the control is frozen at $a\in A$ | operator |
| $p$, $M$ | placeholders for $\partial_xv$ (gradient) and $\partial_{xx}v$ (Hessian) | $\mathbb R^d$, $d\times d$ symmetric |
| $H(t,x,p,M)$ | Hamiltonian, eq. (1.2) | real |
| $w$ | candidate solution of the HJB equation | function |
| $\hat a(t,x)$ | maximizer in the Hamiltonian: candidate optimal feedback | function $\to A$ |
| $\tau_n$ | localizing stopping times in the verification proof | stopping times |
| $\mathcal K_n$, $C_n$ | compact set visited before $\tau_n$, bound of $w$ and its derivatives on it | set, constant |
| $G(s,y)$ | $\partial_sv+H(s,y,\partial_xv,\partial_{xx}v)$ in §3.1 | function |
| $\bar\lambda$, $\bar\eta$, $\bar\delta$ | bounds on intensities, jump sizes, quote distances | constants |
| $\theta(t,q)$ | non-trivial part of the market maker's value, eq. (4.7) | function |
| $N^i_s$, $\lambda_i$, $\eta_i$ | counting processes, their controlled intensities, jump sizes | — |
| $x$, $q$, $s$ (in §3.5, §4.2) | market maker's cash, inventory (integer), mid price | $\mathbb R$, $\mathbb Z$, $\mathbb R$ |
| $\delta^b$, $\delta^a$ | distances of bid and ask quotes from mid (the controls) | $\ge0$ |
| $\Lambda(\delta)=Ae^{-k\delta}$ | fill intensity of a quote at distance $\delta$ | $>0$ |
| $\phi$ | running inventory penalty | $\ge0$ |
| $c$, $\sigma$, $P(t)$, $K(t)$ | LQ example parameters and coefficients (note 03 §4) | — |
| $\tau=T-t$ | time to maturity | $\ge0$ |

---

## 1. Setup & definitions

We keep the setting of note 03: state equation (1.1) there, admissible controls Def. 1.1, reward (1.6), value (1.7), assumptions (H1)–(H4). Start in $d=1$, $m=1$ for readability; the $d$-dimensional formulas follow by note 01 eq. (5.2).

**Definition 1.1 (Controlled generator).** For a fixed control value $a\in A$ and $\varphi\in C^{1,2}$,
$$
\mathcal L^a\varphi(t,x)=b(t,x,a)\,\partial_x\varphi(t,x)+\tfrac12\sigma^2(t,x,a)\,\partial_{xx}\varphi(t,x). \tag{1.1}
$$
In $d$ dimensions: $\mathcal L^a\varphi=b\cdot\nabla\varphi+\tfrac12\operatorname{tr}(\sigma\sigma^\top\nabla^2\varphi)$.
This is note 01's generator (7.1) with the coefficients evaluated at control value $a$. By Itô's lemma (note 01 eq. 5.1) applied to (note 03 eq. 1.1), for **any** admissible control,
$$
d\varphi(s,X_s)=\big(\partial_s\varphi+\mathcal L^{\alpha_s}\varphi\big)(s,X_s)\,ds+\sigma(s,X_s,\alpha_s)\,\partial_x\varphi(s,X_s)\,dW_s.
$$
Itô's lemma applies because $X$ is an Itô process with adapted coefficients $b_s=b(s,X_s,\alpha_s)$, $\sigma_s=\sigma(s,X_s,\alpha_s)$; the control does not need to be a feedback.

**Definition 1.2 (Hamiltonian).** For $(t,x,p,M)\in[0,T]\times\mathbb R\times\mathbb R\times\mathbb R$,
$$
H(t,x,p,M)=\sup_{a\in A}\Big\{b(t,x,a)\,p+\tfrac12\sigma^2(t,x,a)\,M+f(t,x,a)\Big\}. \tag{1.2}
$$
In $d$ dimensions, $p\in\mathbb R^d$, $M$ symmetric $d\times d$, and $\tfrac12\sigma^2M$ becomes $\tfrac12\operatorname{tr}(\sigma\sigma^\top M)$.

Two remarks on (1.2):

- $H$ is a **supremum of functions affine in $(p,M)$**, hence **convex** in $(p,M)$. It is nondecreasing in $M$ (since $\sigma^2\ge0$): this is *degenerate ellipticity*, the structural property behind the comparison principle of note 05.
- $H$ may be $+\infty$ (e.g. $A=\mathbb R$, $b=a$, $f=0$: the controller can push the state at no cost). Then the problem is ill-posed. Under (H4) (compact $A$) and continuity, $H$ is finite and continuous.

The name comes from classical mechanics: with $f$ a negative kinetic energy $-\tfrac12a^2$ and $b=a$, $H(p)=\sup_a\{ap-\tfrac12a^2\}=\tfrac12p^2$ is the Legendre transform, as in Hamilton's equations.

---

## 2. Statement

### 2.1 The HJB equation

**HJB equation.** The value function is expected to satisfy
$$
\boxed{\ \partial_tv(t,x)+H\big(t,x,\partial_xv(t,x),\partial_{xx}v(t,x)\big)=0\ \ \text{on }[0,T)\times\mathbb R,\qquad v(T,x)=g(x).\ } \tag{2.1}
$$
Written out: $\partial_tv+\sup_{a\in A}\{\mathcal L^av+f(\cdot,a)\}=0$.

Compare note 02 eq. (1.2) with $r=0$: $\partial_tu+\mathcal Lu+f=0$. The HJB is that linear equation with a **pointwise** $\sup$ over the control value. It is a fully nonlinear (in general) backward parabolic equation; it is linear iff $A$ is a single point.

**Feedback from the Hamiltonian.** If $v$ is smooth and the $\sup$ in (2.1) is attained, define
$$
\hat a(t,x)\in\arg\max_{a\in A}\Big\{\mathcal L^av(t,x)+f(t,x,a)\Big\}. \tag{2.2}
$$
The infinite-dimensional optimization over processes (note 03 eq. 1.7) has become a **finite-dimensional optimization at each point $(t,x)$**. This is the main practical payoff of dynamic programming.

### 2.2 Verification theorem

**Theorem 2.1 (Verification).** Let $(t,x)\in[0,T)\times\mathbb R^d$. Assume (H1)–(H2) of note 03 and:

- **(V1) Regularity and growth.** $w\in C^{1,2}([0,T)\times\mathbb R^d)\cap C([0,T]\times\mathbb R^d)$ and, for some $C_w$, $p\ge1$: $|w(s,y)|\le C_w(1+|y|^p)$.
- **(V2) Supersolution.** For all $(s,y,a)\in[0,T)\times\mathbb R^d\times A$,
$$
\partial_sw(s,y)+\mathcal L^aw(s,y)+f(s,y,a)\le0,\qquad\text{and}\qquad w(T,y)\ge g(y). \tag{2.3}
$$
- **(V3) Integrability of admissible controls.** Every $\alpha\in\mathcal A_t$ satisfies, with the same $p$,
$$
\mathbb E\Big[\sup_{t\le s\le T}|X^{t,x,\alpha}_s|^p\Big]<\infty\qquad\text{and}\qquad\mathbb E\int_t^T\big|f(s,X^{t,x,\alpha}_s,\alpha_s)\big|\,ds<\infty. \tag{2.4}
$$
(If this is not automatic, it is made part of the definition of $\mathcal A_t$.)

Then:

**(i)** $w(t,x)\ge v(t,x)$.

**(ii)** If moreover there is a measurable $\hat a:[t,T)\times\mathbb R^d\to A$ such that

- **(V4) Equality along $\hat a$.** $\partial_sw+\mathcal L^{\hat a(s,y)}w+f(s,y,\hat a(s,y))=0$ for all $(s,y)$ and $w(T,\cdot)=g$; i.e. $w$ solves (2.1) and $\hat a$ attains the sup;
- **(V5) Admissible closed loop.** The closed-loop SDE (note 03 eq. 1.3) with $\hat a$ has a solution $\hat X$ started at $(t,x)$, and $\hat\alpha_s=\hat a(s,\hat X_s)$ belongs to $\mathcal A_t$,

then $w(t,x)=v(t,x)=J(t,x;\hat\alpha)$: $w$ is the value function and $\hat\alpha$ is optimal.

**How to read it.** (i) says: any smooth supersolution of the HJB is an **upper bound** on what any admissible strategy can earn. (ii) says: if the bound is attained by an admissible feedback, the bound is the value. No a-priori knowledge of $v$ (smoothness, DPP) is needed. The verification theorem uses only Itô's lemma.

### 2.3 Jump HJB

For the jump-controlled state of note 03 eq. (1.9), the generator acquires one term per counting process:
$$
\mathcal L^a\varphi(t,x)=b(t,x,a)\,\partial_x\varphi+\tfrac12\sigma^2(t,x,a)\,\partial_{xx}\varphi+\sum_{i=1}^I\lambda_i(t,x,a)\Big[\varphi\big(t,x+\eta_i(t,x,a)\big)-\varphi(t,x)\Big], \tag{2.5}
$$
and the HJB is (2.1) with this $\mathcal L^a$:
$$
\partial_tv+\sup_{a\in A}\Big\{\mathcal L^av+f(\cdot,a)\Big\}=0,\qquad v(T,\cdot)=g. \tag{2.6}
$$
This is the controlled version of note 01 eq. (6.4) and note 02 eq. (5.1): "rate of the event × change in value if the event happens", where now the **controller chooses the rate**. Theorem 2.1 holds for (2.6) under one extra assumption (J), see §3.4.

---

## 3. Derivation

### 3.1 Formal derivation of the HJB equation from the DPP

**Roadmap.** Two inequalities. (a) Using a *constant* control $a$ on a short interval $[t,t+h]$ and the half (2.3) of note 03's DPP gives "$\partial_tv+\mathcal L^av+f(a)\le0$ for every $a$". (b) Using an almost-optimal control and the half (2.2) gives "$\partial_tv+\sup_a\{\cdots\}\ge0$". Together: (2.1). Both steps are note 02 §3 with a $\sup$.

**Assumption (F) — heuristic.** $v\in C^{1,2}([0,T)\times\mathbb R)$ with $\partial_tv$, $\partial_xv$, $\partial_{xx}v$ bounded and continuous, (H1)–(H4) hold, and $H$ is finite and continuous. **This is an assumption about the unknown $v$; it is generally not verifiable in advance and can fail** (Pitfall 1). That is why this subsection is called formal, and why note 05 (viscosity solutions) exists.

**Step 1: Dynkin for a controlled process.** For any $\alpha\in\mathcal A_t$, integrate Itô's lemma (Def. 1.1) on $[t,t+h]$ and take expectations. The $dW$ integral has zero mean, because its integrand $\sigma(s,X_s,\alpha_s)\partial_xv$ has $\mathbb E\int|\cdot|^2<\infty$ (bounded $\partial_xv$, growth (H2), moment bound note 03 eq. 1.4; note 01 §3). So
$$
\mathbb E\big[v(t+h,X_{t+h})\big]=v(t,x)+\mathbb E\int_t^{t+h}\big(\partial_sv+\mathcal L^{\alpha_s}v\big)(s,X_s)\,ds. \tag{3.1}
$$

**Step 2 ("$\le0$"): constant controls.** Fix $a\in A$ and use $\alpha_s\equiv a$, which is admissible. Note 03's (2.3) says $\mathbb E[\int_t^{t+h}f(s,X_s,a)\,ds+v(t+h,X_{t+h})]\le v(t,x)$. Substituting (3.1), the $v(t,x)$ cancels:
$$
\mathbb E\int_t^{t+h}\Big[\big(\partial_sv+\mathcal L^av\big)(s,X_s)+f(s,X_s,a)\Big]ds\le0.
$$
Divide by $h$ and let $h\downarrow0$. The integrand is continuous in $(s,X_s)$, paths are continuous with $X_t=x$, and the integrand is bounded by $C(1+\sup_s|X_s|^2)$, which is integrable (note 03 eq. 1.4). By dominated convergence the average tends to its value at $(t,x)$:
$$
\partial_tv(t,x)+\mathcal L^av(t,x)+f(t,x,a)\le0\quad\text{for every }a\in A\quad\Longrightarrow\quad\partial_tv+H(t,x,\partial_xv,\partial_{xx}v)\le0. \tag{3.2}
$$

**Step 3 ("$\ge0$"): almost-optimal controls.** By note 03's (2.2), for each $h$ there is $\alpha^h\in\mathcal A_t$ with
$$
v(t,x)-h^2\le\mathbb E\Big[\int_t^{t+h}f(s,X^h_s,\alpha^h_s)\,ds+v(t+h,X^h_{t+h})\Big],
$$
where $X^h=X^{t,x,\alpha^h}$. Substituting (3.1) and then bounding the integrand pointwise by its supremum over control values (the definition (1.2) of $H$):
$$
-h^2\le\mathbb E\int_t^{t+h}\Big[\partial_sv+\mathcal L^{\alpha^h_s}v+f(\cdot,\alpha^h_s)\Big](s,X^h_s)\,ds\le\mathbb E\int_t^{t+h}G(s,X^h_s)\,ds, \tag{3.3}
$$
with $G(s,y):=\partial_sv(s,y)+H(s,y,\partial_xv(s,y),\partial_{xx}v(s,y))$, a continuous function.

Divide by $h$: $-h\le\frac1h\mathbb E\int_t^{t+h}G(s,X^h_s)\,ds$. The controls $\alpha^h$ change with $h$, so we need $X^h_s\to x$ **uniformly in the control**. Since $A$ is compact (H4), (H2) bounds $b,\sigma$ by $C(1+|X|)$ for every control, and the argument of note 03 Lemma 1.2 gives $\mathbb E\sup_{s\in[t,t+h]}|X^h_s-x|^2\le C(1+|x|^2)h$. Hence $G(s,X^h_s)\to G(t,x)$ in probability, uniformly over $s\in[t,t+h]$, and with dominated convergence
$$
0\le G(t,x)=\partial_tv(t,x)+H\big(t,x,\partial_xv,\partial_{xx}v\big). \tag{3.4}
$$

**Step 4: terminal condition.** $v(T,x)=g(x)$ by definition (note 03 §1.4).

Combining (3.2) and (3.4) gives (2.1). $\square$ (formal)

**Recap.** Step 2 is "every action is at best optimal" and gives $\le$. Step 3 is "optimal actions exist approximately" and gives $\ge$. The $\sup$ in $H$ appears in Step 3 when we bound $\mathcal L^{\alpha_s}v+f(\alpha_s)$ by its supremum over $a$. With a single control value, Steps 2 and 3 coincide and reproduce note 02 eq. (3.2).

> **Intuition:** $\partial_tv+\mathcal L^av+f(a)$ is the expected rate of change of "value so far + value to go" (note 03 eq. 2.4) under action $a$. HJB says this rate is $\le0$ for every action and $=0$ for the best one: the infinitesimal form of "supermartingale for every control, martingale for the optimal one" (note 03 Cor. 2.2).

### 3.2 Feedback controls from the Hamiltonian

Given a solution $w$ of (2.1), computing $\hat a$ in (2.2) is a static optimization at each $(t,x)$.

**Typical case (control in the drift, quadratic cost).** $b=a$, $\sigma$ independent of $a$, $f=-\tfrac12a^2+\ell(t,x)$, $A=\mathbb R$. The function $a\mapsto ap-\tfrac12a^2$ is concave with maximizer $a=p$, so
$$
\hat a(t,x)=\partial_xw(t,x),\qquad H=\tfrac12\sigma^2M+\tfrac12p^2+\ell.
$$
The optimal drift **points up the gradient of the value function**, with speed proportional to its steepness.

What can go wrong (and must be checked in (V5)): the $\arg\max$ may not be unique; it may not be measurable or continuous in $(t,x)$; and the closed-loop SDE may then fail to have a solution. In all examples below $\hat a$ is Lipschitz in $x$, so (V5) follows from note 01 Def. 4.2.

### 3.3 Proof of the verification theorem

**Roadmap.** Apply Itô to $w(s,X_s)$ for an arbitrary admissible control. (V2) says the drift of "$\int f\,ds+w(s,X_s)$" is $\le0$, so its expectation can only decrease: this gives (i). Under $\hat\alpha$ the drift is exactly $0$, which gives equality, i.e. (ii). The only technical work is that the $dW$ integral might be just a local martingale (note 01, caution after eq. 3.2): we **localize** with stopping times, and remove the localization with dominated convergence, which is where (V1) and (V3) are used.

**Step 0: localization.** Fix $\alpha\in\mathcal A_t$, write $X=X^{t,x,\alpha}$, and for integers $n>1/(T-t)$ set
$$
\tau_n=\inf\{s\ge t:|X_s-x|\ge n\}\wedge\big(T-\tfrac1n\big). \tag{3.5}
$$
On $[t,\tau_n]$, $(s,X_s)$ stays in the compact set $\mathcal K_n=[t,T-\tfrac1n]\times\{|y-x|\le n\}$, where $w,\partial_tw,\partial_xw,\partial_{xx}w$ are continuous (V1), hence bounded by some $C_n$. We stop before $T$ because (V1) only gives $w\in C^{1,2}$ on $[0,T)$; at $T$ the value function is often not smooth (e.g. $g$ has a kink).

**Step 1: Itô up to $\tau_n$.** By Itô's lemma (Def. 1.1) integrated from $t$ to $\tau_n$:
$$
w(\tau_n,X_{\tau_n})=w(t,x)+\int_t^{\tau_n}\big(\partial_sw+\mathcal L^{\alpha_s}w\big)(s,X_s)\,ds+\int_t^{\tau_n}\sigma(s,X_s,\alpha_s)\,\partial_xw(s,X_s)\,dW_s. \tag{3.6}
$$

**Step 2: the stopped stochastic integral has zero mean.** Write it as $\int_t^T\mathbf 1_{\{s\le\tau_n\}}\sigma\,\partial_xw\,dW_s$; the indicator is adapted, so the integrand is adapted. Its square integral is finite:
$$
\mathbb E\int_t^T\mathbf 1_{\{s\le\tau_n\}}|\sigma\,\partial_xw|^2ds\le C_n^2\,\mathbb E\int_t^T3K^2\big(1+|X_s|^2+|\alpha_s|^2\big)ds<\infty,
$$
using (H2), the moment bound note 03 eq. (1.4), and (1.2) there. So by note 01 §3 it is a true martingale with zero mean. (Without the localization we would need a growth bound on $\partial_xw$, which is not assumed.)

**Step 3: use the supersolution inequality.** Take expectations in (3.6). The $ds$ integrand is bounded on $\mathcal K_n$ by $C_n(1+|b|+\sigma^2)$, which is integrable for the same reasons, so the expectation splits. By (V2) applied with $a=\alpha_s$ (pointwise in $(s,\omega)$),
$$
\big(\partial_sw+\mathcal L^{\alpha_s}w\big)(s,X_s)\le-f(s,X_s,\alpha_s).
$$
Hence
$$
w(t,x)\ \ge\ \mathbb E\Big[\int_t^{\tau_n}f(s,X_s,\alpha_s)\,ds+w(\tau_n,X_{\tau_n})\Big]. \tag{3.7}
$$

**Step 4: remove the localization.** Let $n\to\infty$.

- $\tau_n\to T$ a.s.: the path $s\mapsto X_s$ is continuous on $[t,T]$, hence bounded, so for $n$ large the first term in (3.5) is never reached and $\tau_n=T-\tfrac1n$.
- Running term: $\int_t^{\tau_n}f\,ds\to\int_t^Tf\,ds$ a.s., and $|\int_t^{\tau_n}f\,ds|\le\int_t^T|f|\,ds$, integrable by (V3).
- Terminal term: $w(\tau_n,X_{\tau_n})\to w(T,X_T)$ a.s., by continuity of $w$ on $[0,T]\times\mathbb R^d$ (V1) and of the path; and $|w(\tau_n,X_{\tau_n})|\le C_w(1+\sup_s|X_s|^p)$, integrable by (V3).

By dominated convergence in (3.7), and then $w(T,\cdot)\ge g$ from (V2),
$$
w(t,x)\ge\mathbb E\Big[\int_t^Tf(s,X_s,\alpha_s)\,ds+w(T,X_T)\Big]\ge J(t,x;\alpha). \tag{3.8}
$$
Since $\alpha\in\mathcal A_t$ was arbitrary, $w(t,x)\ge\sup_\alpha J=v(t,x)$. This proves **(i)**.

**Step 5: equality for $\hat\alpha$.** Repeat Steps 0–4 with $\alpha=\hat\alpha$ (admissible by (V5)). By (V4), the inequality in Step 3 is an equality, and $w(T,\cdot)=g$, so every "$\ge$" above is "$=$": $w(t,x)=J(t,x;\hat\alpha)\le v(t,x)$. With (i), $w(t,x)=v(t,x)=J(t,x;\hat\alpha)$. This proves **(ii)**. $\square$

**Recap: where each assumption was used.**

| Assumption | Used in |
|---|---|
| (V1) $C^{1,2}$ on $[0,T)$ | Itô's lemma, Step 1; boundedness on $\mathcal K_n$, Steps 2–3 |
| (V1) continuity on $[0,T]$ | $w(\tau_n,X_{\tau_n})\to w(T,X_T)$, Step 4 |
| (V1) polynomial growth + (V3) moments | dominated convergence for the terminal term, Step 4 |
| (V3) $\mathbb E\int|f|<\infty$ | dominated convergence for the running term, Step 4 |
| (H2) + note 03 (1.2), (1.4) | true martingale property of the stopped integral, Step 2 |
| (V2) | the inequality, Step 3 and (3.8) |
| (V4), (V5) | equality and admissibility of the candidate optimum, Step 5 |

> **Remark (uniqueness).** If $w_1,w_2$ both satisfy (V1), (V4), (V5) (with (V3) for all controls), both equal $v$. So the HJB has **at most one** classical solution of polynomial growth with an admissible maximizer. Compare note 02 Pitfall 2 (Tychonoff): growth conditions select the right solution.

> **Remark (bounds without solving).** Part (i) needs only a *supersolution*. Any smooth $w$ with $\partial_tw+H\le0$, $w(T)\ge g$ is an upper bound on the value; any admissible control gives a lower bound $J$. This sandwich is often useful in practice.

### 3.4 Verification with jumps

**Roadmap.** Same proof, with Itô's lemma with jumps (note 01 eq. 6.3) in Step 1 and one more martingale in Step 2. We first state the Itô formula for counting processes with controlled intensity.

**Itô with controlled intensities.** Note 01's derivation of (6.3) was pathwise (between jumps use (5.1); at a jump, add the exact difference), so it holds for any counting process, whatever its intensity. For the state (note 03 eq. 1.9):
$$
d\varphi(s,X_s)=\big(\partial_s\varphi+\mathcal L^{\alpha_s}\varphi\big)ds+\sigma\,\partial_x\varphi\,dW_s+\sum_i\Big[\varphi(s,X_{s-}+\eta_i)-\varphi(s,X_{s-})\Big]\big(dN^i_s-\lambda_i\,ds\big), \tag{3.9}
$$
with $\mathcal L^a$ from (2.5) and $\eta_i,\lambda_i$ evaluated at $(s,X_{s-},\alpha_s)$. We wrote $dN^i=\lambda_i\,ds+(dN^i-\lambda_i\,ds)$; the $\lambda_i\,ds$ parts form the jump term of $\mathcal L^{\alpha_s}$.

**Fact 3.1 (compensated integrals).** If $Z_s$ is predictable and $\mathbb E\int_t^T|Z_s|\lambda_i(s)\,ds<\infty$, then $\int_t^\cdot Z_s(dN^i_s-\lambda_i\,ds)$ is a martingale with zero mean (Brémaud, *Point Processes and Queues*, Ch. II, T8). Here $Z_s=\varphi(s,X_{s-}+\eta_i)-\varphi(s,X_{s-})$ is predictable because it uses $X_{s-}$.

**Extra assumption (J).** The intensities and jump sizes are bounded: $0\le\lambda_i\le\bar\lambda$, $|\eta_i|\le\bar\eta$.

**Proof of Theorem 2.1 for (2.6).** Steps 0–5 go through with these changes:

- Step 1: use (3.9).
- Step 2: on $[t,\tau_n]$, $X_{s-}$ lies in the ball of radius $n$ around $x$ and $X_{s-}+\eta_i$ in the ball of radius $n+\bar\eta$, where $w$ is bounded. So $|Z_s|\le2C_n$, and $\mathbb E\int|Z|\lambda\le2C_n\bar\lambda T$. By Fact 3.1, the stopped compensated integrals have zero mean.
- Step 4: $\tau_n\to T$ still holds (a càdlàg path on $[t,T]$ is bounded), and $|X_{\tau_n}|\le|x|+n+\bar\eta$ is still controlled by $\sup_s|X_s|$, so dominated convergence applies as before. Left limits replace continuity: $w(\tau_n,X_{\tau_n})\to w(T,X_{T-})$, and $X_{T-}=X_T$ a.s. because a jump exactly at the fixed time $T$ has probability zero (bounded intensity). $\square$

### 3.5 The market-making jump HJB

We now write (2.6) for the market-making state. It is used in Example 4.2 and in note 12.

**State.** Cash $X_s$, inventory $q_s\in\mathbb Z$, mid price $S_s$ with $dS_s=\sigma\,dW_s$. The controls are the quote distances $\delta^b_s,\delta^a_s\in[0,\bar\delta]$: the bid is posted at $S_s-\delta^b_s$, the ask at $S_s+\delta^a_s$. Bid fills arrive at intensity $\Lambda(\delta^b_s)$ and ask fills at intensity $\Lambda(\delta^a_s)$, with $\Lambda(\delta)=Ae^{-k\delta}$. At a **bid fill** the market maker buys one unit at $S-\delta^b$: $(x,q)\to(x-s+\delta^b,\ q+1)$. At an **ask fill** it sells one unit at $S+\delta^a$: $(x,q)\to(x+s+\delta^a,\ q-1)$. So
$$
dX_s=(S_s+\delta^a_s)\,dN^a_s-(S_s-\delta^b_s)\,dN^b_s,\qquad dq_s=dN^b_s-dN^a_s,\qquad dS_s=\sigma\,dW_s. \tag{3.10}
$$
The intensities are bounded by $A$ and the jump sizes in $(x,q)$ are bounded on bounded $s$-sets, so (J) holds after the usual localization.

**HJB.** With a running reward $f=-\phi q^2$ (inventory penalty) and terminal reward $g(x,q,s)$, (2.5)–(2.6) give, for $v=v(t,x,q,s)$,
$$
\partial_tv+\tfrac12\sigma^2\partial_{ss}v-\phi q^2+\sup_{\delta^b}\Lambda(\delta^b)\Big[v(t,x-s+\delta^b,q+1,s)-v\Big]+\sup_{\delta^a}\Lambda(\delta^a)\Big[v(t,x+s+\delta^a,q-1,s)-v\Big]=0. \tag{3.11}
$$
The $\sup$ over $(\delta^b,\delta^a)$ **separates** into two one-dimensional problems, because each control affects only its own jump term. Avellaneda–Stoikov is (3.11) with $\phi=0$ and $g=-e^{-\gamma(x+qs)}$ (CARA utility of note 01 Ex. 5.4); note 12 solves it.

---

## 4. Worked examples

### Example 4.1: LQ steering — fully solved

**Roadmap.** The problem of note 03 §4. We solve the HJB with a quadratic ansatz, reduce it to a Riccati ODE, solve that, and check (V1)–(V5). This upgrades note 03's lower bound (4.11) to an equality and proves the feedback $-P(t)X$ optimal.

**The problem.** $dX=\alpha\,ds+\sigma\,dW$, $A=\mathbb R$, $f=-\tfrac12a^2$, $g=-\tfrac12cx^2$ (note 03 eqs. 4.1–4.2).

**HJB.** By (1.1), $\mathcal L^aw=a\,\partial_xw+\tfrac12\sigma^2\partial_{xx}w$, so (2.1) reads
$$
\partial_tw+\tfrac12\sigma^2\partial_{xx}w+\sup_{a\in\mathbb R}\Big\{a\,\partial_xw-\tfrac12a^2\Big\}=0,\qquad w(T,x)=-\tfrac12cx^2. \tag{4.1}
$$
By §3.2 the sup is attained at $\hat a=\partial_xw$ and equals $\tfrac12(\partial_xw)^2$:
$$
\partial_tw+\tfrac12\sigma^2\partial_{xx}w+\tfrac12(\partial_xw)^2=0. \tag{4.2}
$$

**Ansatz.** The terminal condition is quadratic and the dynamics are linear, so try
$$
w(t,x)=-\tfrac12P(t)\,x^2-K(t),\qquad P(T)=c,\ K(T)=0. \tag{4.3}
$$
Then $\partial_xw=-Px$, $\partial_{xx}w=-P$, $\partial_tw=-\tfrac12P'x^2-K'$. Substitute in (4.2):
$$
-\tfrac12P'x^2-K'-\tfrac12\sigma^2P+\tfrac12P^2x^2=0\quad\text{for all }x.
$$
Matching the $x^2$ and constant terms:
$$
P'=P^2,\qquad K'=-\tfrac12\sigma^2P. \tag{4.4}
$$
**Solve.** $(1/P)'=-P'/P^2=-1$ and $1/P(T)=1/c$, so $1/P(t)=1/c+(T-t)$, i.e. $P(t)=\dfrac{c}{1+c\tau}$. Then $K(t)=\tfrac12\sigma^2\int_t^TP(s)\,ds=\tfrac12\sigma^2\int_0^\tau\frac{c}{1+cu}\,du=\tfrac12\sigma^2\ln(1+c\tau)$. Hence
$$
\boxed{\ w(t,x)=-\frac{c\,x^2}{2(1+c\tau)}-\frac12\sigma^2\ln(1+c\tau),\qquad\hat a(t,x)=-P(t)\,x=-\frac{c\,x}{1+c\tau}.\ } \tag{4.5}
$$
This is the right side of note 03 eq. (4.11), and $P$ is exactly the discrete $P_n$ of note 03 eq. (4.9).

**Verification (Theorem 2.1).** Take $\mathcal A_t$ = progressively measurable real controls with $\mathbb E\int_t^T\alpha^2ds<\infty$ (note 03 Def. 1.1).

- (V1): $w$ is a quadratic polynomial in $x$ with smooth coefficients on $[0,T]$; $p=2$.
- (V2), (V4): by construction, $\partial_tw+\mathcal L^aw+f(a)=-\tfrac12(a-\partial_xw)^2\le0$ for every $a$ (complete the square), with equality iff $a=\hat a$; and $w(T)=g$.
- (V3): $X_s=x+\int_t^s\alpha+\sigma(W_s-W_t)$, so $\sup_s|X_s|^2\le3\big(x^2+T\int\alpha^2+\sigma^2\sup_s|W_s-W_t|^2\big)$, integrable by Doob; and $\mathbb E\int|f|=\tfrac12\mathbb E\int\alpha^2<\infty$.
- (V5): the closed loop $d\hat X=-P(s)\hat X\,ds+\sigma\,dW$ is a linear SDE (Lipschitz), so it has a unique Gaussian solution; $\hat\alpha=-P\hat X$ has $\mathbb E\int\hat\alpha^2<\infty$.

Therefore $v=w$ and $\hat\alpha_s=-P(s)X_s$ is optimal. Note 03's open-loop value (4.4) and discrete values (4.10) are strictly below $v$, as they must be by (i).

Note that this example violates (H4) ($A=\mathbb R$) and (H3) (quadratic rewards), which were needed for the DPP proof in note 03 and the formal derivation in §3.1. The verification theorem does not need them. This is a common pattern: derive HJB formally, then *prove* the answer by verification.

**Numbers.** $c=4$, $\sigma=1$, $T=1$:

| $(t,x)$ | $P(t)$ | $\hat a(t,x)$ | $v(t,x)$ |
|---|---|---|---|
| $(0,1)$ | $0.8$ | $-0.8$ | $-0.4-\tfrac12\ln5=-1.20472$ |
| $(0.5,1)$ | $4/3$ | $-1.333$ | $-0.6667-\tfrac12\ln3=-1.21597$ |
| $(0.9,1)$ | $2.857$ | $-2.857$ | $-1.4286-\tfrac12\ln1.4=-1.59681$ |

Comparison with suboptimal admissible controls (exact rewards from the moment ODE, see §5): doing nothing gives $-\tfrac12c(x^2+\sigma^2T)=-4$; the best **constant** gain $\alpha=-\kappa X$ is $\kappa\approx1.561$ with reward $-1.3443$; both below $v=-1.2047$. The time-varying gain $P(t)$ matters.

**Remark 4.1 (Cole–Hopf: the HJB is a heat equation in disguise).** Put $w=\sigma^2\ln\psi$. Then $\partial_xw=\sigma^2\psi_x/\psi$, $\partial_{xx}w=\sigma^2(\psi_{xx}/\psi-\psi_x^2/\psi^2)$, $\partial_tw=\sigma^2\psi_t/\psi$. Substituting into (4.2), the $\psi_x^2$ terms cancel:
$$
\frac{\sigma^2}{\psi}\Big(\partial_t\psi+\tfrac12\sigma^2\partial_{xx}\psi\Big)=0,\qquad\psi(T,x)=e^{-cx^2/(2\sigma^2)}. \tag{4.6}
$$
So $\psi$ solves the backward heat equation, and Feynman–Kac (note 02 eq. 1.3) gives $\psi(t,x)=\mathbb E\big[e^{-c(x+\sigma W_\tau)^2/(2\sigma^2)}\big]$. With the Gaussian identity $\mathbb E[e^{-\beta(m+sZ)^2}]=(1+2\beta s^2)^{-1/2}e^{-\beta m^2/(1+2\beta s^2)}$ for $Z\sim\mathcal N(0,1)$, $\beta=c/(2\sigma^2)$, $m=x$, $s^2=\sigma^2\tau$ (so $2\beta s^2=c\tau$):
$$
\psi=(1+c\tau)^{-1/2}\exp\Big(-\frac{cx^2}{2\sigma^2(1+c\tau)}\Big)\quad\Longrightarrow\quad\sigma^2\ln\psi=w\ \text{of (4.5)}.\ \checkmark
$$
This independent route (linear PDE + expectation) is one of the numerical checks.

> **Intuition:** the optimal controller pays $\tfrac12\sigma^2\ln(1+c\tau)$ for noise, against $\tfrac12c\sigma^2\tau$ for the open-loop one. The logarithm is the signature of continuously "re-hedging" noise as it arrives; the linear term is what happens when you let it accumulate.

### Example 4.2: risk-neutral market maker — jump HJB

**Roadmap.** Solve (3.11) with $\phi=0$ and terminal reward $g=x+qs$ (marked-to-mid wealth, no risk aversion). An ansatz removes $x$ and $s$, the jump terms become explicit, and each $\sup$ is a 1-d calculus problem. We recover note 02 Ex. 5.3's $\delta^*=1/k$, now as a proved optimal control. Then we add $\phi>0$ to see inventory skew appear.

**Ansatz.** Try
$$
w(t,x,q,s)=x+q\,s+\theta(t,q),\qquad\theta(T,q)=0. \tag{4.7}
$$
Then $\partial_{ss}w=0$, and the jump differences are
$$
w(t,x-s+\delta^b,q+1,s)-w=-s+\delta^b+s+\theta(t,q+1)-\theta(t,q)=\delta^b+\theta(t,q+1)-\theta(t,q),
$$
$$
w(t,x+s+\delta^a,q-1,s)-w=s+\delta^a-s+\theta(t,q-1)-\theta(t,q)=\delta^a+\theta(t,q-1)-\theta(t,q).
$$
The mid price cancels: a fill at $S\mp\delta$ is worth exactly $\delta$ relative to marking at mid. Substituting in (3.11), all dependence on $x,s$ drops out, and $\theta$ solves the **ODE system indexed by $q\in\mathbb Z$**
$$
\partial_t\theta(t,q)-\phi q^2+\sup_{\delta^b}\Lambda(\delta^b)\big[\delta^b+\theta(t,q+1)-\theta(t,q)\big]+\sup_{\delta^a}\Lambda(\delta^a)\big[\delta^a+\theta(t,q-1)-\theta(t,q)\big]=0. \tag{4.8}
$$
(Compare note 02 eq. (5.2): integer state ⟹ system of ODEs.)

**The pointwise sup.** For $p\in\mathbb R$, maximize $F(\delta)=Ae^{-k\delta}(\delta+p)$ over $\delta\ge0$. $F'(\delta)=Ae^{-k\delta}\big(1-k(\delta+p)\big)$, positive for $\delta<1/k-p$ and negative after, so
$$
\delta^*(p)=\Big(\frac1k-p\Big)\vee0,\qquad\sup_{\delta\ge0}F=\frac{A}{e\,k}\,e^{kp}\quad\text{if }p\le\frac1k. \tag{4.9}
$$
(At the interior optimum, $\delta^*+p=1/k$ and $e^{-k\delta^*}=e^{-1}e^{kp}$.) The optimal bid distance is $\delta^{b*}=\frac1k-\big[\theta(q+1)-\theta(q)\big]$: the quote moves by exactly the marginal value of one more unit of inventory.

**Case $\phi=0$.** Try $\theta=\theta(t)$, independent of $q$. Then both $p$'s are $0$, both quotes are $\delta^*=1/k$, both sups equal $A/(ek)$, and (4.8) becomes $\theta'+2A/(ek)=0$, $\theta(T)=0$:
$$
\boxed{\ v(t,x,q,s)=x+qs+\frac{2A}{e\,k}\,(T-t),\qquad\delta^{b*}=\delta^{a*}=\frac1k.\ } \tag{4.10}
$$

**Verification.** Control set $[0,\bar\delta]$ with $\bar\delta>1/k$ (so the optimum is interior); intensities $\le A$ and the jump in $(x,q)$ is bounded on bounded $s$-sets, so (J) holds after localization. (V1): $w$ is smooth and of linear growth ($p=1$). (V3): total fills on $[t,T]$ are dominated by a Poisson($2A\tau$) variable, which has all moments; $S$ is Gaussian; so $q$, $X$, $S$ have all moments, and $f=0$. (V2), (V4): by (4.9) with $p=0$, $\sup$ is attained at $1/k$. (V5): constant quotes are admissible. Hence (4.10) is the value function and constant quotes at $1/k$ are optimal.

**Numbers.** Avellaneda–Stoikov parameters $A=140$, $k=1.5$, $T=1$: $\delta^*=0.6667$ and $\frac{2A}{ek}T=\frac{280}{1.5\,e}=68.671$. That is $2\times140e^{-1}=103.0$ expected fills, each earning $0.6667$.

**Case $\phi>0$ (inventory aversion, numerical).** The constant-in-$q$ ansatz fails because of $-\phi q^2$, but (4.8)–(4.9) is still a closed ODE system, solved numerically in `code/hjb_checks.py` (truncated at $|q|\le60$). With $\phi=0.05$:

| $q$ | $-2$ | $-1$ | $0$ | $+1$ | $+2$ |
|---|---|---|---|---|---|
| $\delta^{b*}(0,q)$ | $0.6283$ | $0.6539$ | $0.6795$ | $0.7051$ | $0.7307$ |
| $\delta^{a*}(0,q)$ | $0.7307$ | $0.7051$ | $0.6795$ | $0.6539$ | $0.6283$ |

$\theta(0,0)=67.595$. A long market maker ($q>0$) **widens its bid and tightens its ask**, so it is more likely to sell than buy: the inventory skew of note 01 Ex. 5.4, now produced by the HJB. Keeping symmetric quotes at $1/k$ earns only $\approx66.08$ under the same penalty (Monte Carlo), consistent with (i) of Theorem 2.1. The skew per unit of inventory here is $\approx0.0256$; notes 12–13 derive its closed-form approximations.

> **Intuition:** in (4.9) the quote is "monopoly half-spread $1/k$ minus the marginal value of the inventory change the fill causes". If buying one more unit lowers your continuation value ($\theta(q+1)<\theta(q)$), you demand a bigger discount to buy.

---

## 5. Numerical check

Script: `code/hjb_checks.py`; tests: `code/tests/test_04_hjb.py` (run `python -m pytest code/tests -q`).

| Check | Methods compared | Expected | Tolerance |
|---|---|---|---|
| Riccati | `solve_ivp` on (4.4) vs closed form | $P(0)=0.8$, $K(0)=\tfrac12\ln5$ | $10^{-9}$ |
| HJB residual | finite differences of (4.5) plugged into (4.2) at 9 points | $0$ | $10^{-5}$ |
| HJB PDE solve | explicit monotone FD scheme for (4.1), brute-force $\sup$ over 401 control values, no ansatz | $-1.20472$ | $10^{-3}$ (observed $2.6\cdot10^{-4}$, second order) |
| Cole–Hopf | Monte Carlo of $\sigma^2\ln\mathbb E[e^{-cX_T^2/2\sigma^2}]$ (Remark 4.1) | $-1.20472$ | $3\cdot10^{-3}$ |
| Verification (i) | exact rewards of constant-gain feedbacks via moment ODE, and their optimum | all $<v$; best $-1.3443$ | strict |
| Sup formula | (4.9) vs brute-force grid over $\delta$ | — | $10^{-7}$ rel. |
| Risk-neutral MM | ODE system (4.8) vs (4.10) vs Monte Carlo of quotes at $1/k$ | $68.671$, $\delta^*=0.6667$ | $10^{-6}$; 4 s.e. |
| Inventory penalty | ODE with closed-form sup vs explicit solver with grid sup vs Monte Carlo of the resulting feedback; skew monotone; symmetric quotes worse | $\theta(0,0)=67.595$ | $10^{-3}$; 4 s.e. + 0.02 |

The FD scheme uses central differences for the drift; it is monotone because $|a|\le\sigma^2/\Delta x$ on the control grid and $\Delta t\le\Delta x^2/\sigma^2$. Monotonicity is the property that guarantees convergence to the right (viscosity) solution — the Barles–Souganidis theorem of note 05.

---

## 6. Pitfalls / intuition

1. **The HJB may have no classical solution.** Take $dX=\alpha\,ds$ (no noise), $A=[-1,1]$, $f=0$, $g(x)=-|x|$. Moving toward $0$ at full speed is optimal, so $v(t,x)=-(|x|-\tau)^+$. It is not differentiable on $|x|=\tau$. The HJB here is $\partial_tv+|\partial_xv|=0$, and $v$ satisfies it only almost everywhere.
2. **"Almost-everywhere solutions" are not unique.** For the same dynamics with $g=0$, the true value is $v\equiv0$, but $\tilde w(t,x)=(\tau-|x|)^+$ also satisfies $\partial_t\tilde w+|\partial_x\tilde w|=0$ a.e. (on $|x|<\tau$: $-1+1=0$) and $\tilde w(T)=0$. So the PDE in the a.e. sense does not determine $v$. Viscosity solutions (note 05) fix this.
3. **The Hamiltonian can be infinite.** With unbounded controls and no cost of control, $H=+\infty$ and the problem is degenerate (arbitrarily large rewards, or a singular/impulse control problem; note 14).
4. **Local martingales.** Without localization and integrability, the "$\mathbb E\int\cdots dW=0$" step can fail. Doubling-type strategies are admissible-looking controls that break it; (V3) and (1.2) of note 03 exclude them.
5. **Check the closed loop.** An $\arg\max$ that is discontinuous in $x$ (bang-bang controls, as in Pitfall 1) can make the closed-loop SDE ill-posed. (V5) is a real condition.
6. **Formal vs proved.** §3.1 is heuristic (it assumes $v$ smooth). Theorem 2.1 is a full proof, modulo the quoted Fact 3.1 for jumps. In practice: guess from §3.1, prove by Theorem 2.1, or fall back on viscosity theory + numerics when no smooth solution exists.
7. **Textbook vs practice.** Real market makers do not solve (3.11) online. They use the structure it reveals — quotes = base half-spread − marginal inventory value — with parameters ($A,k,\phi$) estimated and recalibrated from data (note 15).

---

## 中文小结

- **HJB 方程**：由 DPP 加 Itô/Dynkin 形式推导：用常数控制得到"对每个 $a$，$\partial_tv+\mathcal L^av+f\le0$"，用近似最优控制得到"$\ge0$"，合起来得 $\partial_tv+\sup_a\{\mathcal L^av+f\}=0$，$v(T)=g$。这是第 02 篇线性 PDE 加一个逐点 sup；推导假设 $v$ 光滑，属于启发式。
- **Hamiltonian**：$H=\sup_a\{bp+\tfrac12\sigma^2M+f\}$，关于 $(p,M)$ 凸且对 $M$ 单调（退化椭圆性）。最优反馈 $\hat a(t,x)$ 是逐点最大化的解，把对过程的无穷维优化变成每个点的有限维优化。
- **验证定理（完整证明）**：光滑、多项式增长的上解 $w$ 是任何容许策略收益的上界；若 $\hat a$ 达到 sup 且闭环 SDE 容许，则 $w=v$，$\hat a$ 最优。证明用停时局部化保证随机积分是真鞅，再用控制收敛去掉局部化；每个假设的用途都列在表中。
- **带跳 HJB**：生成元加上 $\sum\lambda_i(a)[v(x+\eta_i)-v(x)]$，控制者选择跳跃强度；做市问题中买卖两侧的 sup 可分离。Avellaneda–Stoikov 就是 $\phi=0$、CARA 终端效用的情形（第 12 篇）。
- **例 1（LQ）**：二次猜解化为 Riccati $P'=P^2$，$v=-\tfrac{cx^2}{2(1+c\tau)}-\tfrac12\sigma^2\ln(1+c\tau)$，反馈 $-P(t)x$；$c=4,\sigma=1,x=1$ 时 $v=-1.2047$。Cole–Hopf 变换把 HJB 化为热方程，可用 Feynman–Kac 独立验证。
- **例 2（风险中性做市）**：猜解 $x+qs+\theta$，最优报价 $1/k$，$v=x+qs+\tfrac{2A}{ek}\tau$；$A=140,k=1.5$ 时为 68.67。加入库存惩罚 $\phi q^2$ 后，多头时买价变宽、卖价变窄（库存偏移），数值解与蒙特卡洛一致。
- **陷阱**：HJB 可能没有经典解，几乎处处解不唯一（需要粘性解，第 05 篇）；Hamiltonian 可能为无穷；必须检查局部鞅与闭环 SDE 的适定性。