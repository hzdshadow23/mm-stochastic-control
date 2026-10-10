# CLAUDE.md

## Purpose

A self-study repo to **understand stochastic optimal control (HJB, BSDE) and its application to market making**, from first principles to the formulas used in practice. The goal is understanding every derivation step, not building a production system.

## Reader

- PhD in applied math (numerical PDEs, NN-based PDE solvers). Strong in PDE theory, analysis, numerics.
- Rates quant (pricing, curves, risk). Comfortable with Itô calculus at a working level; new to control theory and market microstructure.
- Therefore: do **not** re-explain basic PDE/analysis; **do** spell out every probabilistic and control-theoretic step.

## Writing rules (non-negotiable)

1. **Define before use.** Every symbol, term, and assumption is defined before (or at) its first appearance in each note. No undefined notation, no "it is well known that".
2. **No skipped steps.** Derivations go line by line. When a step uses a lemma (Itô, DPP, Girsanov, martingale representation, …), name it and state exactly what is applied.
3. **State assumptions explicitly** (regularity, integrability, boundedness) and say where each is used. Flag heuristic steps as heuristic (e.g. formal HJB derivation assuming $v \in C^{1,2}$).
4. **Math in LaTeX** (`$…$`, `$$…$$`). No plain-text formulas.
5. **Concrete numbers.** Every model gets a worked numerical example; every closed form is checked against a numerical solution in `code/`.
6. **Bilingual.** English main body; each note ends with a concise `## 中文小结`.
7. **Very detailed and readable.** Notes are the opposite of terse: explain *why* each step is taken, not just *what* it is. Readability rules:
   - Short paragraphs; one idea per paragraph.
   - Before each derivation, a 1–3 sentence roadmap of where it is going; after it, a 1–2 sentence recap of what was shown.
   - Number important equations and refer back to them by number.
   - Use a "Notation" table at the top of each note listing every symbol, its meaning, and its type/units.
   - Add intuition boxes (`> **Intuition:** …`) after key results, and a small numeric sanity check where possible.
   - Prefer the 1-d case first, then generalize to $d$ dimensions.
   - Still no filler or motivational prose: detail means more explanation, not more words.
8. **Honesty.** If a result is approximate (e.g. Avellaneda–Stoikov small-$q$ expansion), say so and quantify the error. Distinguish textbook models from what electronic market makers actually do.

## Repo layout

```
notes/        # one Markdown file per topic, numbered in dependency order
code/         # Python numerical verification (numpy/scipy; torch only for Deep BSDE)
  tests/      # pytest checks: closed forms vs. numerics
references.md # papers/books with the exact sections used
```

Each note: `## Setup & definitions` → `## Statement` → `## Derivation` → `## Worked example` → `## Numerical check` (link to `code/`) → `## Pitfalls / intuition` → `## 中文小结`.

## Roadmap (dependency order)

**Part I — Foundations**
1. `01_probability_and_ito.md` — filtration, adapted processes, Brownian motion, Itô integral, Itô's lemma (incl. jump/Poisson version).
2. `02_feynman_kac.md` — linear PDE ↔ conditional expectation.

**Part II — Stochastic control**
3. `03_control_problem_and_dpp.md` — controlled SDE, value function, dynamic programming principle.
4. `04_hjb_derivation.md` — formal HJB derivation, Hamiltonian, verification theorem.
5. `05_viscosity_solutions.md` — why needed, definition, comparison principle, Barles–Souganidis.
6. `06_merton_and_lq.md` — Merton portfolio + linear-quadratic → Riccati; first fully solved examples.

**Part III — BSDE**
7. `07_bsde_basics.md` — definition, existence/uniqueness (Lipschitz driver), martingale representation.
8. `08_nonlinear_feynman_kac.md` — FBSDE ↔ semilinear PDE, $Y=v$, $Z=\sigma^\top\nabla v$.
9. `09_pontryagin.md` — stochastic maximum principle, adjoint BSDE.
10. `10_deep_bsde.md` — Han–Jentzen–E scheme, implementation, convergence intuition.

**Part IV — Market making & execution**
11. `11_almgren_chriss.md` — optimal liquidation, Riccati, $\sinh$ solution.
12. `12_avellaneda_stoikov.md` — model, jump HJB, CARA ansatz, reservation price, spread; small-$q$ approximation and its error.
13. `13_glft.md` — inventory cap, ODE system, stationary quotes.
14. `14_extensions.md` — alpha signals / adverse selection (Cartea–Jaimungal), impulse control & QVI, mean-field games.
15. `15_practice_vs_theory.md` — what real electronic MMs do; interview-style derivations.

## Code conventions

- Python ≥ 3.11, `numpy`, `scipy`, `matplotlib`; `torch` only in Deep BSDE.
- Each closed form has a test comparing it to an independent numeric method (finite differences, ODE solver, or Monte Carlo) with an explicit tolerance.
- Seed all randomness. Keep scripts small and readable; clarity over performance.

## Workflow

- Work one note at a time, in roadmap order; a note may only use concepts from earlier notes (or must define them).
- After writing a derivation, re-check each line; run the matching test before marking the note done.
- Commit per note.

## Budget & review checkpoints (cloud sessions)

- The user has **$100 of credit** for cloud sessions on this project.
- **Stop and ask for review** when either (a) the roadmap is complete, or (b) about **$50** has been spent (≈ $50 remaining). Summarize what was done, what is next, and any open questions.
- Claude **cannot see the credit balance** directly. So:
  - Treat each Part (I–IV) as a natural checkpoint; at the end of each Part, report progress and ask the user to check the remaining balance before continuing.
  - If the user reports the balance at any point, record it in `PROGRESS.md` and use it to decide whether to continue.
  - Prefer finishing a note cleanly over starting a new one when close to a checkpoint.
- Keep `PROGRESS.md` updated after every note: notes done, tests passing, next step, last reported balance.

## Core references

- Pham, *Continuous-time Stochastic Control and Optimization with Financial Applications* (HJB, viscosity, BSDE).
- Cartea, Jaimungal, Penalva, *Algorithmic and High-Frequency Trading* (market making, execution).
- Avellaneda & Stoikov (2008); Guéant, Lehalle, Fernandez-Tapia (2013); Almgren & Chriss (2000); Han, Jentzen, E (2018).
