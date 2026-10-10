# 03 — The Control Problem and the Dynamic Programming Principle

**Goal of this note.** Set up a stochastic optimal control problem precisely: the controlled SDE, the admissible controls, the reward, and the value function. Then state and prove (with a careful sketch) the **dynamic programming principle (DPP)**. It is the controlled version of note 02 eq. (3.1):

| Note 02 (no control) | This note (control) |
|---|---|
| $u(t,x)=\mathbb E_{t,x}\big[u(t+h,X_{t+h})\big]$ — eq. (3.1) | $v(t,x)=\sup_\alpha\mathbb E_{t,x}\Big[\int_t^{t+h}f\,ds+v(t+h,X^\alpha_{t+h})\Big]$ — eq. (2.1) |
| $u(s,X_s)$ is a martingale | $\int_t^sf\,dr+v(s,X^\alpha_s)$ is a supermartingale for every $\alpha$, a martingale for the optimal $\alpha$ |

In note 04 the DPP plus Itô's lemma gives the HJB equation, exactly as note 02 §3 got the linear PDE from (3.1).

**Prerequisites.** Note 01: Itô isometry (3.2), SDEs (Def. 4.2), Itô's lemma (5.1)–(5.2), jumps (6.2)–(6.4). Note 02: Feynman–Kac (1.3), the uncontrolled DPP (3.1).

---

## Notation

| Symbol | Meaning | Type |
|---|---|---|
| $T$ | fixed horizon | $>0$ |
| $(\Omega,\mathcal F,\{\mathcal F_s\},\mathbb P)$ | filtered probability space carrying $W$ | — |
| $W_s$ | $m$-dimensional Brownian motion | $\mathbb R^m$-valued process |
| $A$ | control set: the values a control may take | closed subset of $\mathbb R^k$ |
| $a$ | a generic point of $A$ (a control *value*) | $\in A$ |
| $\alpha=(\alpha_s)_{s\in[t,T]}$ | a control *process* | $A$-valued process |
| $\mathcal A_t$ | admissible controls on $[t,T]$ (Def. 1.1) | set of processes |
| $X_s=X^{t,x,\alpha}_s$ | controlled state, $X_t=x$ | $\mathbb R^d$-valued process |
| $b(s,x,a)$, $\sigma(s,x,a)$ | controlled drift ($\mathbb R^d$) and volatility ($\mathbb R^{d\times m}$) | functions |
| $K$ | Lipschitz / growth constant in (H1)–(H3) | $>0$ |
| $f(s,x,a)$ | running reward rate | function, real |
| $g(x)$ | terminal reward | function, real |
| $J(t,x;\alpha)$ | expected reward of control $\alpha$ from $(t,x)$ | real |
| $v(t,x)$ | value function $=\sup_{\alpha}J(t,x;\alpha)$ | function |
| $h$ | DPP step, $0\le h\le T-t$ | time |
| $\theta$ | intermediate time ($=t+h$) | time |
| $S^\alpha_s$ | $\int_t^sf\,dr+v(s,X^\alpha_s)$ (Cor. 2.2) | process |
| $\hat a(s,x)$ | feedback (Markov) control rule | function $\to A$ |
| $c$ | terminal penalty in the example of §4 | $>0$ |
| $\tau=T-t$ | time to maturity | $\ge0$ |
| $P(t)$, $K(t)$ | coefficients of the value function in §4 | functions of $t$ |
| $N$, $h=\tau/N$, $t_n$ | number of decision dates, step, dates $t_n=t+nh$ in §4 | — |

Vectors are column vectors; $|\cdot|$ is the Euclidean norm (Frobenius norm for matrices).

---

## 1. Setup & definitions

### 1.1 The controlled SDE

**Roadmap.** A control problem has four ingredients: dynamics that depend on a control, a class of allowed controls, a reward, and the value (best achievable reward). We define them in this order.

Fix a starting time $t\in[0,T]$ and state $x\in\mathbb R^d$. Given a control process $\alpha$, the state solves
$$
dX_s=b(s,X_s,\alpha_s)\,ds+\sigma(s,X_s,\alpha_s)\,dW_s,\qquad s\in[t,T],\qquad X_t=x. \tag{1.1}
$$
For a fixed path of $\alpha$, (1.1) is an SDE as in note 01 Def. 4.2, but with *random* coefficients $\tilde b(s,x,\omega)=b(s,x,\alpha_s(\omega))$. The control enters only through these coefficients.

**Assumptions on the coefficients.** There is a constant $K$ such that for all $s\in[0,T]$, $x,y\in\mathbb R^d$, $a\in A$:

- **(H1) Lipschitz in $x$, uniformly in $a$:** $|b(s,x,a)-b(s,y,a)|+|\sigma(s,x,a)-\sigma(s,y,a)|\le K|x-y|$.
- **(H2) Linear growth:** $|b(s,x,a)|+|\sigma(s,x,a)|\le K(1+|x|+|a|)$; and $b,\sigma$ are continuous in $(s,x,a)$.

"Uniformly in $a$" is the point of (H1): the same constant works whatever control is applied. This is what makes the estimates below independent of the control.

### 1.2 Admissible controls

A control must be chosen using only past and present information, exactly like the integrand of the Itô integral (note 01 Def. 3.1).

**Definition 1.1 (Admissible control).** $\mathcal A_t$ is the set of processes $\alpha:[t,T]\times\Omega\to A$ that are

1. **progressively measurable:** for every $s$, the map $(r,\omega)\mapsto\alpha_r(\omega)$ on $[t,s]\times\Omega$ is $\mathcal B([t,s])\otimes\mathcal F_s$-measurable. (This is "adapted" plus joint measurability in time, which is what is needed for $\int\alpha_r\,dr$ and $\int\sigma(r,X_r,\alpha_r)\,dW_r$ to be adapted. Every adapted process with left- or right-continuous paths qualifies.)
2. **square integrable:**
$$
\mathbb E\int_t^T|\alpha_s|^2\,ds<\infty. \tag{1.2}
$$

Under (H1)–(H2) and (1.2), Picard iteration (note 01 Def. 4.2) works verbatim with the random coefficients, so (1.1) has a unique solution $X^{t,x,\alpha}$.

Two important subclasses:

- **Feedback (Markov) control:** $\alpha_s=\hat a(s,X_s)$ for a measurable function $\hat a:[t,T]\times\mathbb R^d\to A$. Then (1.1) becomes the *closed-loop* SDE
$$
dX_s=b\big(s,X_s,\hat a(s,X_s)\big)ds+\sigma\big(s,X_s,\hat a(s,X_s)\big)dW_s. \tag{1.3}
$$
It is admissible if (1.3) has a solution and (1.2) holds. This is **not automatic**: if $\hat a$ is discontinuous in $x$, the coefficients of (1.3) are not Lipschitz.
- **Open-loop deterministic control:** $\alpha_s=a(s)$, a deterministic function of time. It cannot react to the noise.

> **Intuition:** a feedback control is a *rule* ("if inventory is $q$, quote $\delta(q)$"); an open-loop control is a *schedule* fixed in advance. §4 computes how much the rule beats the schedule in a concrete example.

### 1.3 Two estimates on the controlled state

**Roadmap.** Every later argument needs (i) $X$ has finite second moments and (ii) $X$ depends continuously on its starting point, *with constants that do not depend on the control*. Both follow from Itô's isometry, Doob's inequality and Gronwall.

Two tools we use:

- **Doob's $L^2$ inequality.** For a square-integrable martingale $M$ on $[t,T]$, $\mathbb E\big[\sup_{t\le r\le s}|M_r|^2\big]\le4\,\mathbb E|M_s|^2$.
- **Gronwall's lemma.** If $\psi\ge0$ is bounded and $\psi(s)\le a+b\int_t^s\psi(r)\,dr$ on $[t,T]$, then $\psi(s)\le a\,e^{b(s-t)}$.

**Lemma 1.2 (moments and stability).** Assume (H1)–(H2) and let $\alpha\in\mathcal A_t$.

(a) There is $C$ depending only on $K,T$ with
$$
\mathbb E\Big[\sup_{t\le s\le T}|X^{t,x,\alpha}_s|^2\Big]\le C\Big(1+|x|^2+\mathbb E\int_t^T|\alpha_s|^2ds\Big). \tag{1.4}
$$
(b) For two starting points $x,y$ and the **same** $\alpha$,
$$
\mathbb E\Big[\sup_{t\le s\le T}\big|X^{t,x,\alpha}_s-X^{t,y,\alpha}_s\big|^2\Big]\le C_1|x-y|^2,\qquad C_1=3\,e^{3K^2(T+4)T}. \tag{1.5}
$$
$C_1$ depends only on $K$ and $T$, **not on $\alpha$**.

*Proof of (b).* Let $\Delta_s=X^{t,x,\alpha}_s-X^{t,y,\alpha}_s$ and $\psi(s)=\mathbb E\sup_{t\le r\le s}|\Delta_r|^2$. Subtracting the two integral equations,
$$
\Delta_s=(x-y)+\int_t^s\big[b(r,X^x_r,\alpha_r)-b(r,X^y_r,\alpha_r)\big]dr+\int_t^s\big[\sigma(r,X^x_r,\alpha_r)-\sigma(r,X^y_r,\alpha_r)\big]dW_r.
$$
Use $|p+q+w|^2\le3(|p|^2+|q|^2+|w|^2)$ and bound the three terms:

- the first is $|x-y|^2$;
- drift term: by Cauchy–Schwarz, $\big|\int_t^s\cdots dr\big|^2\le(s-t)\int_t^s|\cdots|^2dr$, and by (H1) $|\cdots|\le K|\Delta_r|$. Its $\mathbb E\sup$ is at most $TK^2\int_t^s\psi(r)\,dr$;
- noise term: Doob's inequality, then Itô's isometry (note 01 eq. 3.2), then (H1): $\mathbb E\sup|\int\cdots dW|^2\le4\,\mathbb E\int_t^s|\cdots|^2dr\le4K^2\int_t^s\psi(r)\,dr$.

Hence $\psi(s)\le3|x-y|^2+3K^2(T+4)\int_t^s\psi(r)\,dr$, and Gronwall gives (1.5). The control appeared only inside $b(\cdot,\cdot,\alpha_r)$ and $\sigma(\cdot,\cdot,\alpha_r)$, where (H1) removed it.

*Proof of (a)* is identical with $\Delta$ replaced by $X$, using (H2) instead of (H1); the $|\alpha|^2$ term from (H2) produces $\mathbb E\int|\alpha|^2$. (Strictly, Gronwall needs $\psi$ bounded; one first stops $X$ when $|X|$ exceeds $n$ and lets $n\to\infty$.) $\square$

**Recap.** Moments are finite for every admissible control, and the flow $x\mapsto X^{t,x,\alpha}$ is Lipschitz in mean square, uniformly over controls.

### 1.4 Reward and value function

**Assumption (H3).** $f:[0,T]\times\mathbb R^d\times A\to\mathbb R$ and $g:\mathbb R^d\to\mathbb R$ are continuous, Lipschitz in $x$ with constant $K$ uniformly in $(s,a)$, and $|f(s,x,a)|+|g(x)|\le K(1+|x|)$.

**Definition 1.3 (Reward and value).** For $\alpha\in\mathcal A_t$,
$$
J(t,x;\alpha)=\mathbb E\Big[\int_t^Tf\big(s,X^{t,x,\alpha}_s,\alpha_s\big)\,ds+g\big(X^{t,x,\alpha}_T\big)\Big], \tag{1.6}
$$
$$
v(t,x)=\sup_{\alpha\in\mathcal A_t}J(t,x;\alpha). \tag{1.7}
$$
By (H3) and (1.4), $J$ is finite. We assume $v$ is finite (true e.g. if $f,g$ are bounded above).

Comments on the definition:

- $v$ is a **deterministic function** of the starting point, like $u$ in note 02 (Pitfall 3 there). It is defined by an optimization over *processes*, i.e. over an infinite-dimensional set.
- At $t=T$ there is nothing to choose: $v(T,x)=g(x)$.
- **Minimization** (costs) is the same problem with $f,g$ replaced by $-f,-g$. **Discounting** at rate $r$ is included by replacing $f$ with $e^{-r(s-t)}f$ and $g$ with $e^{-r(T-t)}g$, or by adding the discount factor as an extra state as in note 02 eq. (2.2).
- A control $\alpha^*$ is **optimal** if $J(t,x;\alpha^*)=v(t,x)$. It need not exist; $\varepsilon$-optimal controls ($J\ge v-\varepsilon$) always exist by the definition of $\sup$.

### 1.5 The freezing lemma

One probabilistic tool is used repeatedly below and in note 04.

**Lemma 1.4 (Freezing lemma).** Let $\mathcal G\subseteq\mathcal F$ be a σ-algebra, $Y$ a $\mathcal G$-measurable random variable, $\xi$ a random variable independent of $\mathcal G$, and $\Phi$ measurable with $\mathbb E|\Phi(Y,\xi)|<\infty$. Then
$$
\mathbb E\big[\Phi(Y,\xi)\mid\mathcal G\big]=\varphi(Y),\qquad\varphi(y):=\mathbb E\big[\Phi(y,\xi)\big]. \tag{1.8}
$$
In words: condition on what you know by **freezing** it at its realized value, then average over the independent noise. (Proof: true for $\Phi(y,z)=1_B(y)1_D(z)$ by independence, then by a monotone class argument.) Note 02's Markov-property step behind eq. (3.1) is an instance of this lemma.

### 1.6 Extension: controlled jumps

For market making (notes 12–13), the control changes the **intensity** of Poisson-type fills rather than the drift. The state equation becomes
$$
dX_s=b(s,X_s,\alpha_s)\,ds+\sigma(s,X_s,\alpha_s)\,dW_s+\sum_{i=1}^I\eta_i(s,X_{s-},\alpha_s)\,dN^i_s, \tag{1.9}
$$
where $N^i$ are counting processes whose **intensity** is $\lambda_i(s,X_{s-},\alpha_s)$: by definition, $N^i_s-\int_t^s\lambda_i(r,X_{r-},\alpha_r)\,dr$ is a martingale. This generalizes note 01 §6, where $\lambda$ was a constant. For this to make sense the control must be **predictable** (roughly: determined by information strictly before $s$, e.g. left-continuous and adapted), so a quote cannot be changed "at the instant of the fill" to exploit it.

Constructing such $N^i$ (by thinning a Poisson random measure, or by a change of measure) is standard but not done here. With bounded intensities and bounded jump sizes $\eta_i$, Lemma 1.2 and §2–§3 go through with the obvious changes (the compensated jump integrals $\int\cdots(dN^i-\lambda_i\,ds)$ are handled like the Itô integral, with Itô's isometry replaced by $\mathbb E\big(\int H\,(dN-\lambda\,ds)\big)^2=\mathbb E\int H^2\lambda\,ds$ for predictable $H$); note 04 §3.4 writes out the jump HJB.

---

## 2. Statement

**Theorem 2.1 (Dynamic programming principle).** Assume (H1)–(H3) and

- **(H4)** $A$ is compact.

Then for every $(t,x)\in[0,T]\times\mathbb R^d$ and every $h\in[0,T-t]$,
$$
\boxed{\ v(t,x)=\sup_{\alpha\in\mathcal A_t}\mathbb E\Big[\int_t^{t+h}f\big(s,X^{t,x,\alpha}_s,\alpha_s\big)\,ds+v\big(t+h,X^{t,x,\alpha}_{t+h}\big)\Big].\ } \tag{2.1}
$$
Equivalently, the following two inequalities hold:
$$
v(t,x)\le\sup_{\alpha\in\mathcal A_t}\mathbb E\Big[\int_t^{t+h}f\,ds+v(t+h,X^\alpha_{t+h})\Big], \tag{2.2}
$$
$$
\mathbb E\Big[\int_t^{t+h}f\,ds+v(t+h,X^\alpha_{t+h})\Big]\le v(t,x)\qquad\text{for every }\alpha\in\mathcal A_t. \tag{2.3}
$$

**How to read it.** Optimizing over $[t,T]$ in one go is the same as: optimize over $[t,t+h]$, *assuming you will behave optimally afterwards*, so that the future is worth $v(t+h,\cdot)$. The infinite-dimensional problem is split into a short-horizon problem plus a known "continuation value". This is **Bellman's principle**: the tail of an optimal policy is optimal for the tail problem.

**How note 02's DPP gains a sup.** If $A=\{a_0\}$ is a single point, then $\mathcal A_t$ contains only the constant control, $J=v$ is the Feynman–Kac expectation of note 02 eq. (1.3) (with $r=0$), the $\sup$ in (2.1) is over one element, and (2.1) is note 02 eq. (3.1) (with the running reward $f$ added). With more than one control, (2.1) keeps the same tower-property skeleton, and the $\sup$ appears because the controller re-optimizes at time $t+h$ knowing $X_{t+h}$.

**Corollary 2.2 (martingale optimality principle).** For $\alpha\in\mathcal A_t$, define
$$
S^\alpha_s=\int_t^sf\big(r,X^\alpha_r,\alpha_r\big)\,dr+v\big(s,X^\alpha_s\big),\qquad s\in[t,T]. \tag{2.4}
$$
Then (i) $S^\alpha$ is a **supermartingale** for every $\alpha$; (ii) $\alpha$ is optimal **iff** $S^\alpha$ is a **martingale**.

*Proof.* (i) The conditional version of (2.3), started at $(s,X^\alpha_s)$ instead of $(t,x)$ (same proof as §3.4, conditioned on $\mathcal F_s$), gives $\mathbb E[S^\alpha_{s'}\mid\mathcal F_s]\le S^\alpha_s$ for $s\le s'$.

(ii) Note $S^\alpha_t=v(t,x)$ and, since $v(T,\cdot)=g$, $\mathbb E[S^\alpha_T]=J(t,x;\alpha)$.
If $S^\alpha$ is a martingale, $J(t,x;\alpha)=\mathbb E S^\alpha_T=S^\alpha_t=v(t,x)$, so $\alpha$ is optimal.
Conversely, if $\alpha$ is optimal then $\mathbb E S^\alpha_T=S^\alpha_t$. Fix $s$ and let $D=S^\alpha_s-\mathbb E[S^\alpha_T\mid\mathcal F_s]$. By (i), $D\ge0$. By (i) again, $\mathbb E S^\alpha_s\le S^\alpha_t=\mathbb E S^\alpha_T$, so $\mathbb E D\le 0$. A nonnegative variable with nonpositive mean is $0$ a.s., so $S^\alpha_s=\mathbb E[S^\alpha_T\mid\mathcal F_s]$: a martingale. $\square$

> **Intuition:** "value so far + value to go" can only decrease in expectation under a bad decision, and stays flat under optimal decisions. Every unit of expected decrease is value lost to a suboptimal action. Compare note 02: there, with no choices, $D\,u+\int D f$ was always a martingale.

---

## 3. Derivation (proof sketch of Theorem 2.1)

**Roadmap.** (2.2) is the "easy" half: condition on $\mathcal F_{t+h}$ and note that whatever the control does after $t+h$ is at best optimal. (2.3) is the "hard" half: we must *build* a control that follows $\alpha$ until $t+h$ and then behaves almost optimally from wherever it lands. The difficulty is choosing the continuation measurably in the landing point; we avoid it with a countable partition and the continuity of $v$ (Lemma 3.1). One measure-theoretic fact (Fact 3.2) is quoted, not proved.

Throughout, $\theta=t+h$.

### 3.1 Continuity of the value function

**Lemma 3.1.** Under (H1)–(H3), with $C_2=K(T+1)\sqrt{C_1}$,
$$
|J(t,x;\alpha)-J(t,y;\alpha)|\le C_2|x-y|\quad\text{for all }\alpha\in\mathcal A_t,\qquad|v(t,x)-v(t,y)|\le C_2|x-y|. \tag{3.1}
$$

*Proof.* With $\Delta=X^{t,x,\alpha}-X^{t,y,\alpha}$ and the Lipschitz bounds of (H3),
$$
|J(t,x;\alpha)-J(t,y;\alpha)|\le\mathbb E\Big[\int_t^TK|\Delta_s|\,ds+K|\Delta_T|\Big]\le K(T+1)\,\mathbb E\Big[\sup_s|\Delta_s|\Big]\le K(T+1)\sqrt{C_1}\,|x-y|,
$$
using $\mathbb E Z\le(\mathbb EZ^2)^{1/2}$ (Cauchy–Schwarz) and (1.5).

For $v$: for each $\alpha$, $J(t,x;\alpha)\le J(t,y;\alpha)+C_2|x-y|\le v(t,y)+C_2|x-y|$. Taking $\sup_\alpha$ on the left, $v(t,x)\le v(t,y)+C_2|x-y|$. Swap $x$ and $y$. $\square$

The key is that $C_2$ is uniform in $\alpha$; a supremum of uniformly Lipschitz functions is Lipschitz.

### 3.2 The conditioning fact

**Fact 3.2 (conditioning on $\mathcal F_\theta$).** Let $\alpha\in\mathcal A_t$, $X=X^{t,x,\alpha}$.

(a) Almost surely,
$$
\mathbb E\Big[\int_\theta^Tf(s,X_s,\alpha_s)\,ds+g(X_T)\ \Big|\ \mathcal F_\theta\Big]\le v(\theta,X_\theta). \tag{3.2}
$$
(b) The value $v(\theta,y)$ is unchanged if, in (1.7) at time $\theta$, we restrict to controls that are functionals of the **future increments** $(W_s-W_\theta)_{s\ge\theta}$ only (hence independent of $\mathcal F_\theta$).

*Why it is true.* After $\theta$, $X$ solves (1.1) started from $X_\theta$, driven by the increments $W_s-W_\theta$, which are independent of $\mathcal F_\theta$ (note 01 Def. 1.4(2)). The control after $\theta$ may depend on the past, but conditionally on $\mathcal F_\theta$ the past is **frozen** (Lemma 1.4): for a fixed past $\omega$, $s\mapsto\alpha_s$ is a control driven by the future increments, started at the frozen state $X_\theta(\omega)$. Its conditional reward is therefore some $J(\theta,X_\theta(\omega);\alpha^\omega)\le v(\theta,X_\theta(\omega))$.
Making "for a fixed past" rigorous needs regular conditional probabilities on the canonical Wiener space; see Pham, Lemma 3.3.1 and Thm 3.3.1, or Fleming–Soner Ch. IV. **This step is quoted, not proved here.**

### 3.3 Proof of (2.2): "$v\le$ sup"

Fix $\alpha\in\mathcal A_t$. Split the reward at $\theta$ and apply the tower property:
$$
J(t,x;\alpha)=\mathbb E\Big[\int_t^\theta f\,ds+\mathbb E\Big[\int_\theta^Tf\,ds+g(X_T)\ \Big|\ \mathcal F_\theta\Big]\Big]\le\mathbb E\Big[\int_t^\theta f\,ds+v(\theta,X_\theta)\Big],
$$
where the inequality is (3.2). The right side is at most the $\sup$ over $\alpha$ in (2.2). Taking $\sup_\alpha$ on the left gives (2.2). $\square$

### 3.4 Proof of (2.3): "$v\ge$ every candidate"

**Roadmap.** Fix any $\alpha\in\mathcal A_t$ and $\varepsilon,\delta>0$. We paste: "use $\alpha$ until $\theta$, then switch to an $\varepsilon$-optimal control for the cell of a fine partition in which $X_\theta$ landed." This pasted control is admissible, so its reward is at most $v(t,x)$; and its reward is within $\varepsilon+2C_2\delta$ of the left side of (2.3).

**Step 1: discretize the landing point.** Partition $\mathbb R^d$ into countably many disjoint Borel sets $B_1,B_2,\dots$ of diameter $<\delta$ (e.g. half-open cubes), and pick $x_i\in B_i$.

**Step 2: choose $\varepsilon$-optimal continuations.** For each $i$, by Fact 3.2(b) choose $\alpha^i\in\mathcal A_\theta$, a functional of the increments $(W_s-W_\theta)_{s\ge\theta}$, with
$$
J(\theta,x_i;\alpha^i)\ge v(\theta,x_i)-\varepsilon.
$$
This is a *countable* family of choices, so no measurable-selection theorem is needed.

**Step 3: paste.** Define
$$
\hat\alpha_s=\begin{cases}\alpha_s,&t\le s<\theta,\\[2pt]\sum_i\mathbf 1\{X^{t,x,\alpha}_\theta\in B_i\}\,\alpha^i_s,&\theta\le s\le T.\end{cases} \tag{3.3}
$$
*Admissibility.* The event $\{X_\theta\in B_i\}$ is in $\mathcal F_\theta$, and each $\alpha^i$ is progressively measurable on $[\theta,T]$, so $\hat\alpha$ is progressively measurable. It takes values in $A$, which is compact by (H4), so (1.2) holds. Hence $\hat\alpha\in\mathcal A_t$. (Without (H4) one needs an extra truncation; this is where compactness is used.)

**Step 4: compute the conditional reward of $\hat\alpha$.** On $[t,\theta]$, $\hat\alpha=\alpha$, so $X^{\hat\alpha}=X^\alpha$ there. On the event $\{X_\theta\in B_i\}$, after $\theta$ the state solves (1.1) from $X_\theta$ with control $\alpha^i$. Here $X_\theta$ is $\mathcal F_\theta$-measurable and $\alpha^i$ together with the driving increments is independent of $\mathcal F_\theta$. By the freezing lemma (1.8) and pathwise uniqueness for (1.1),
$$
\mathbb E\Big[\int_\theta^Tf\,ds+g(X^{\hat\alpha}_T)\ \Big|\ \mathcal F_\theta\Big]=J(\theta,X_\theta;\alpha^i)\qquad\text{on }\{X_\theta\in B_i\}.
$$

**Step 5: compare with $v$ using continuity.** On $\{X_\theta\in B_i\}$ we have $|X_\theta-x_i|<\delta$, so by (3.1) twice,
$$
J(\theta,X_\theta;\alpha^i)\ge J(\theta,x_i;\alpha^i)-C_2\delta\ge v(\theta,x_i)-\varepsilon-C_2\delta\ge v(\theta,X_\theta)-\varepsilon-2C_2\delta.
$$
This holds on every cell, hence a.s.

**Step 6: conclude.** Write $i(y)$ for the index of the cell containing $y$. Since $\hat\alpha$ is admissible, by the tower property, Step 4 and Step 5,
$$
v(t,x)\ge J(t,x;\hat\alpha)=\mathbb E\Big[\int_t^\theta f(s,X^\alpha_s,\alpha_s)\,ds+J\big(\theta,X^\alpha_\theta;\alpha^{i(X^\alpha_\theta)}\big)\Big]\ge\mathbb E\Big[\int_t^\theta f\,ds+v(\theta,X^\alpha_\theta)\Big]-\varepsilon-2C_2\delta.
$$
$\varepsilon,\delta$ are arbitrary, so (2.3) holds. $\square$

**Recap.** The "$\le$" half is just conditioning. The "$\ge$" half needs (i) the ability to paste controls at $\theta$ (admissibility is stable under pasting), and (ii) continuity of $v$ in $x$ (Lemma 3.1), which lets a countable family of near-optimal controls cover all landing points. Without continuity, the DPP still holds in a weaker form (Bouchard–Touzi 2011), but that refinement is not needed here.

> **Remark (stopping times).** The same proof works with $\theta$ replaced by any stopping time with values in $[t,T]$. We only use deterministic $\theta=t+h$ in note 04.

---

## 4. Worked example: steering a noisy state to zero

**Roadmap.** A 1-d problem where the controller pays for effort and is penalized for distance from $0$ at $T$. We compute (i) the best **open-loop** value by elementary calculus, (ii) the best value over controls that decide at $N$ dates, by running the DPP backward in discrete time — this gives an exact formula for every $N$, and (iii) the limit $N\to\infty$. The gap between (i) and (iii) is the value of using feedback.

This problem violates (H3)–(H4) (quadratic rewards, $A=\mathbb R$). Everything in this section is proved directly by elementary arguments, so Theorem 2.1 is not used.

### 4.1 The problem

State and reward ($d=m=k=1$, $A=\mathbb R$, $c>0$, $\sigma>0$):
$$
dX_s=\alpha_s\,ds+\sigma\,dW_s,\qquad X_t=x, \tag{4.1}
$$
$$
J(t,x;\alpha)=\mathbb E\Big[-\int_t^T\tfrac12\alpha_s^2\,ds-\tfrac12c\,X_T^2\Big]. \tag{4.2}
$$
Reading: $X$ is, e.g., an inventory you must bring to $0$ by $T$; $\alpha$ is the trading rate with a quadratic cost; $\sigma\,dW$ is unhedgeable order flow. (Almgren–Chriss, note 11, is a relative.)

### 4.2 Best open-loop control

For deterministic $\alpha=a(\cdot)$, let $I=\int_t^Ta(s)\,ds$. Then from (4.1)
$$
X_T=x+I+\sigma(W_T-W_t)\sim\mathcal N\big(x+I,\ \sigma^2\tau\big), \tag{4.3}
$$
so $\mathbb E X_T^2=(x+I)^2+\sigma^2\tau$ and $J=-\tfrac12\int a^2-\tfrac12c\big[(x+I)^2+\sigma^2\tau\big]$.

*Step 1: fix $I$, minimize effort.* By Cauchy–Schwarz, $I^2=\big(\int_t^T1\cdot a\big)^2\le\tau\int_t^Ta^2$, with equality iff $a$ is constant. So the best schedule with total $I$ is constant, $a\equiv I/\tau$, and $J=-\tfrac{I^2}{2\tau}-\tfrac12c(x+I)^2-\tfrac12c\sigma^2\tau$.

*Step 2: optimize $I$.* Setting the derivative to zero, $-I/\tau-c(x+I)=0$, so $I^*=-\dfrac{c\tau x}{1+c\tau}$ and $x+I^*=\dfrac{x}{1+c\tau}$. Then
$\dfrac{(I^*)^2}{\tau}+c(x+I^*)^2=\dfrac{c^2\tau x^2+cx^2}{(1+c\tau)^2}=\dfrac{cx^2}{1+c\tau}$. Hence
$$
v_{\rm OL}(t,x)=-\frac{c\,x^2}{2(1+c\tau)}-\frac12c\sigma^2\tau. \tag{4.4}
$$
The noise term $-\tfrac12c\sigma^2\tau$ is the full penalty on the accumulated variance: an open-loop schedule cannot react to it at all.

### 4.3 Decisions at $N$ dates: the DPP in discrete time

Let $h=\tau/N$, $t_n=t+nh$. Restrict to controls that are constant on each $[t_n,t_{n+1})$, with value $a_n$ chosen using information up to $t_n$ ($a_n$ is $\mathcal F_{t_n}$-measurable, $\mathbb E a_n^2<\infty$). These are admissible controls of the continuous problem, so the best reward among them, $v^h(t,x)$, satisfies
$$
v^h(t,x)\le v(t,x). \tag{4.5}
$$
Writing $X_n=X_{t_n}$ and $\xi_n=(W_{t_{n+1}}-W_{t_n})/\sqrt h\sim\mathcal N(0,1)$ (independent of $\mathcal F_{t_n}$), (4.1) gives exactly $X_{n+1}=X_n+a_nh+\sigma\sqrt h\,\xi_n$, and the running reward over a step is exactly $-\tfrac12a_n^2h$.

**Backward recursion.** Define $V_N(y)=-\tfrac12cy^2$ and, for $n=N-1,\dots,0$,
$$
V_n(y)=\sup_{a\in\mathbb R}\Big\{-\tfrac12a^2h+\mathbb E\big[V_{n+1}\big(y+ah+\sigma\sqrt h\,\xi\big)\big]\Big\},\qquad\xi\sim\mathcal N(0,1). \tag{4.6}
$$
This is (2.1) with $h$ one step and the $\sup$ over a single number $a$, because the control is constant over the step.

**Solve the recursion.** Guess $V_{n+1}(y)=-\tfrac12P_{n+1}y^2-K_{n+1}$ (true for $n+1=N$ with $P_N=c$, $K_N=0$). Using $\mathbb E(m+\sigma\sqrt h\xi)^2=m^2+\sigma^2h$,
$$
V_n(y)=\sup_a\Big\{-\tfrac12a^2h-\tfrac12P_{n+1}(y+ah)^2\Big\}-\tfrac12\sigma^2hP_{n+1}-K_{n+1}.
$$
The bracket is a concave quadratic in $a$. Its derivative is $-ah-P_{n+1}h(y+ah)$, which vanishes at
$$
a_n^*=-\frac{P_{n+1}}{1+P_{n+1}h}\,y. \tag{4.7}
$$
Then $y+a_n^*h=\dfrac{y}{1+P_{n+1}h}$ and the bracket equals $-\tfrac12\dfrac{hP_{n+1}^2y^2+P_{n+1}y^2}{(1+P_{n+1}h)^2}=-\tfrac12\dfrac{P_{n+1}}{1+P_{n+1}h}y^2$. So the guess propagates with
$$
P_n=\frac{P_{n+1}}{1+P_{n+1}h},\qquad K_n=K_{n+1}+\tfrac12\sigma^2h\,P_{n+1}. \tag{4.8}
$$
**Closed form.** The first relation is $1/P_n=1/P_{n+1}+h$, so $1/P_n=1/c+(N-n)h$:
$$
P_n=\frac{c}{1+c(T-t_n)},\qquad a_n^*=-P_n\,X_n,\qquad K_0=\tfrac12\sigma^2h\sum_{j=0}^{N-1}\frac{c}{1+c\,jh}. \tag{4.9}
$$
(For $a_n^*$: $\frac{P_{n+1}}{1+P_{n+1}h}=P_n$ by (4.8). For $K_0$: $K_0=\tfrac12\sigma^2h\sum_{n=1}^NP_n$ and $T-t_n=(N-n)h$ runs over $jh$, $j=0,\dots,N-1$.)

**Why the recursion gives the optimum (discrete verification).** Let $(a_n)$ be any admissible discrete control and $X_n$ its state. By the freezing lemma (1.8) with $\mathcal G=\mathcal F_{t_n}$, $Y=(X_n,a_n)$, and $\xi=\xi_n$:
$$
\mathbb E\big[-\tfrac12a_n^2h+V_{n+1}(X_{n+1})\mid\mathcal F_{t_n}\big]=\Big\{-\tfrac12a^2h+\mathbb E V_{n+1}(y+ah+\sigma\sqrt h\xi)\Big\}\Big|_{y=X_n,\,a=a_n}\le V_n(X_n),
$$
with equality iff $a_n=a_n^*(X_n)$. Take expectations and sum over $n=0,\dots,N-1$; the $V_n$ telescope:
$$
J(t,x;(a_n))=\mathbb E\Big[\sum_n-\tfrac12a_n^2h+V_N(X_N)\Big]\le V_0(x),
$$
with equality for $a_n=-P_nX_n$. (Integrability: $\mathbb E a_n^2<\infty$ implies $\mathbb E X_n^2<\infty$, so all terms are finite.) Hence $v^h=V_0$:
$$
v^h(t,x)=-\tfrac12P_0x^2-K_0. \tag{4.10}
$$
This telescoping argument is the discrete prototype of the verification theorem of note 04, and the inequality with equality iff optimal is Corollary 2.2 in discrete time.

### 4.4 The continuous-time limit

As $N\to\infty$, $P_0=c/(1+c\tau)$ does not change, and $K_0$ is a left Riemann sum of a decreasing function:
$$
K_0\ \longrightarrow\ \tfrac12\sigma^2\int_0^\tau\frac{c}{1+cu}\,du=\tfrac12\sigma^2\ln(1+c\tau),
$$
with error $\approx\tfrac14\sigma^2h\,\big(c-\tfrac{c}{1+c\tau}\big)=O(h)$ (trapezoid correction). By (4.5), $v\ge\lim_Nv^h$, i.e.
$$
v(t,x)\ \ge\ -\frac{c\,x^2}{2(1+c\tau)}-\frac12\sigma^2\ln(1+c\tau). \tag{4.11}
$$
In note 04 (Example 4.1) the HJB equation plus the verification theorem shows that (4.11) holds with **equality**, with optimal feedback $\alpha^*_s=-P(s)X_s$, $P(s)=c/(1+c(T-s))$. Here we have only the lower bound, because a continuous-time control could in principle do better than every piecewise-constant one.

**Value of feedback.** Comparing (4.4) and (4.11), the $x$-dependent parts coincide; the difference is in the noise term:
$$
v-v_{\rm OL}\ \ge\ \tfrac12\sigma^2\big[c\tau-\ln(1+c\tau)\big]>0\quad(c\tau>0). \tag{4.12}
$$
(The inequality $\ln(1+z)<z$ for $z>0$.) For $\sigma=0$ the gap vanishes: without noise there is nothing to react to.

### 4.5 Numbers

$c=4$, $\sigma=1$, $T=1$, $t=0$, $x=1$ (so $\tau=1$, $P_0=0.8$):

| Quantity | Formula | Value |
|---|---|---|
| open-loop value | (4.4): $-0.4-2$ | $-2.4000$ |
| $N=10$ dates | (4.10) | $-1.29102$ |
| $N=100$ dates | (4.10) | $-1.21278$ |
| $N=1000$ dates | (4.10) | $-1.20552$ |
| limit | (4.11): $-0.4-\tfrac12\ln5$ | $-1.20472$ |
| first action | $a^*_0=-P_0x$ | $-0.8$ |

The errors $0.0863,\ 0.0081,\ 0.0008$ decrease like $h$, and the predicted leading error $\tfrac14h(4-0.8)=0.8h$ matches. Feedback cuts the expected cost roughly in half ($2.40\to1.20$): the open-loop controller pays $\tfrac12c\sigma^2\tau=2$ for terminal variance, the feedback controller only $\tfrac12\sigma^2\ln5=0.80$, because it keeps correcting the noise as it arrives.

**DPP sanity check (Corollary 2.2).** Take $h=0.25$ and use the limit formula (4.11) as $v$ (justified in note 04). Then $P(0.25)=4/(1+3)=1$.
- Under $\alpha\equiv0$ on $[0,0.25]$: $X_{0.25}=1+W_{0.25}$, no running cost, and
$\mathbb E\,v(0.25,X_{0.25})=-\tfrac12\cdot1\cdot(1+0.25)-\tfrac12\ln4=-1.31815<-1.20472=v(0,1)$: a strict loss of $0.113$, as (2.3) requires.
- Under the optimal feedback on $[0,0.25]$, the same quantity equals $v(0,1)$ (martingale). The Monte Carlo test checks both.

> **Intuition:** the gain $P(t)=c/(1+c(T-t))$ increases toward $c$ as $t\to T$: early on, a deviation can be corrected gradually and cheaply (and more noise is still to come, so effort now is partly wasted); near $T$ the controller pushes hard. The same urgency appears in Almgren–Chriss and in inventory skew near the close.

---

## 5. Numerical check

Script: `code/control_dpp_checks.py`; tests: `code/tests/test_03_control_dpp.py` (run `python -m pytest code/tests -q`).

| Check | Methods compared | Expected | Tolerance |
|---|---|---|---|
| Bellman recursion | iterate (4.8) vs closed form (4.9)–(4.10) | $P_0=0.8$, $v^h$ for $N=10,100,1000$ | $10^{-12}$ |
| $O(h)$ convergence | $N\,(v^h-v)$ at $N=1000$ vs predicted leading term | $-0.8$ | $2\%$ |
| Generic DPP on a grid | brute-force $\max_a$ + Gauss–Hermite + spline, **no quadratic ansatz**, vs (4.10), $N=10$ | $-1.29102$ | $5\cdot10^{-4}$ |
| Open-loop | BFGS over 20 deterministic control values vs (4.4); optimizer is constant | $-2.4$ | $10^{-6}$ |
| Feedback value | Monte Carlo of $\alpha=-P(s)X$ vs (4.11) | $-1.20472$ | 4 s.e. + Euler bias |
| Open-loop MC | Monte Carlo of the constant schedule vs (4.4) | $-2.4$ | 4 s.e. |
| DPP / Cor. 2.2 | MC of both sides of (2.1) at $h=0.25$ | $-1.20472$ (optimal), $-1.31815$ ($\alpha=0$) | 4 s.e. + bias |

---

## 6. Pitfalls / intuition

1. **$v$ is defined by an optimization over processes**, not numbers. The DPP is what reduces it to a sequence of finite-dimensional choices; in note 04 it reduces to a pointwise $\sup_a$ in the Hamiltonian.
2. **The "$\ge$" half of the DPP is the hard one.** It needs pasting of controls and continuity (or measurable selection). Proofs that "just condition on $\mathcal F_{t+h}$" prove only (2.2).
3. **Admissibility is part of the problem.** Changing $\mathcal A_t$ (e.g. allowing unbounded controls without (1.2)) can change $v$. Feedback rules must produce a well-posed closed-loop SDE (1.3).
4. **Open-loop ≠ feedback under noise.** The best deterministic schedule is generally strictly worse (4.12). Without noise ($\sigma=0$) they coincide.
5. **Discrete decision dates give lower bounds.** $v^h\le v$ always (4.5); in this example the gap is $O(h)$. This is the practical meaning of "re-quote more often".
6. **Heuristic vs proved.** Fact 3.2 (conditioning with a frozen past) is quoted from Pham. The discrete-time results of §4.3 are fully proved; the equality in (4.11) is proved in note 04.

---

## 中文小结

- **控制问题四要素**：受控 SDE $dX=b(s,X,\alpha)ds+\sigma(s,X,\alpha)dW$；容许控制（循序可测、平方可积，只能用当前及过去信息）；收益 $J=\mathbb E[\int f\,ds+g(X_T)]$；值函数 $v=\sup_\alpha J$，是起点 $(t,x)$ 的确定性函数，$v(T,x)=g(x)$。
- **两个估计**：在关于 $x$ 一致 Lipschitz 的条件下，$X$ 的二阶矩有限，且对初值的依赖是 Lipschitz 的，常数与控制无关；由此 $v$ 关于 $x$ Lipschitz 连续。
- **动态规划原理**：$v(t,x)=\sup_\alpha\mathbb E[\int_t^{t+h}f\,ds+v(t+h,X_{t+h})]$。它是第 02 篇 (3.1) 加上一个 sup：只有一个控制时退化为无控制 DPP。"$\le$"方向只需条件期望；"$\ge$"方向需要在 $t+h$ 处拼接 $\varepsilon$-最优控制，靠可数划分和 $v$ 的连续性避免可测选择问题。条件化时"冻结过去"的严格化引自 Pham。
- **鞅最优性原理**：$\int f+v(s,X_s)$ 对任意控制是上鞅，当且仅当控制最优时是鞅。
- **例子（把带噪状态推向 0）**：离散时间 DPP 可精确求解，$P_n=c/(1+c\tau_n)$，最优反馈 $a=-P_nX_n$；连续极限 $v\ge-\tfrac{cx^2}{2(1+c\tau)}-\tfrac12\sigma^2\ln(1+c\tau)$（第 04 篇证明取等号）。开环最优值为 $-\tfrac{cx^2}{2(1+c\tau)}-\tfrac12c\sigma^2\tau$。$c=4,\sigma=1,\tau=1,x=1$ 时：反馈 $-1.205$，开环 $-2.4$，反馈把成本几乎减半；离散决策误差为 $O(h)$。
- **数值验证**：Bellman 递推、无假设网格 DPP、开环数值优化、蒙特卡洛（含 DPP 两侧检验）全部与闭式解一致。