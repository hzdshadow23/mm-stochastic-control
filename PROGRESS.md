# Progress

| # | Note | Status | Tests |
|---|------|--------|-------|
| 01 | Probability & Itô calculus (incl. jumps, generator, Dynkin) | ✅ done | 5/5 pass |
| 02 | Feynman–Kac (diffusion + jumps, MM spread preview) | ✅ done | 4/4 pass |
| 03 | Control problem & DPP (estimates, DPP proof sketch, open-loop vs feedback LQ example) | ✅ done | 7/7 pass |
| 04 | HJB derivation, verification theorem (full proof), jump HJB, LQ + market-maker examples | ✅ done | 8/8 pass |
| 05 | Viscosity solutions | ⏳ next | — |
| 06 | Merton & LQ/Riccati | not started | — |
| 07–10 | Part III — BSDE | not started | — |
| 11–15 | Part IV — market making & execution | not started | — |

Run all tests: `python -m pytest code/tests -q`
Build PDFs of the notes: `./build_pdf.sh` (output in `pdf/`).

**Checkpoint:** notes 03–04 done (end of this run's budget) — waiting for user review and balance check.
**Next step:** note 05 (viscosity solutions), then 06 (Merton, LQ) to finish Part II.
**All tests:** 24/24 pass.
**Last reported credit balance:** $100 (start; not yet updated).
**Model:** Claude Opus 5.5 for all sessions (user decision, 2026-10-10).
**Current run:** Part II on Claude Opus 5.5; stop after ≈ $20 spend (default: after notes 03–04).
