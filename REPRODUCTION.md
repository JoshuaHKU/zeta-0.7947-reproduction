# Reproduction checklist — preprint-0.8/paper.tex

DOI: 10.5281/zenodo.21975236
Public repository / 公开仓库:
https://github.com/JoshuaHKU/zeta-0.7947-reproduction

Self-contained reproduction and certification package for the paper
"More than 0.7962 of the zeros of the Riemann zeta function are simple
and on the critical line" (authors: Hongyi Yang, Shihua Yang;
mathematical development by Claude (Anthropic), see the paper's
Acknowledgements). The paper is built on the publicly posted
two-thirds preprint [C26] (Claude, Anthropic, August 2026,
https://www-cdn.anthropic.com/564f962e60643842f5fcb4a17c9dbc8f608f1c37.pdf),
whose framework it cites without repetition (paper §2); all other
references are published literature. Everything referenced below lives inside
this directory (pipeline/, scripts/, certification/, lean/); the
program's complete forward-only research archive (round logs, memos,
ladders, correction history) is preserved separately and is not
required to reproduce any quoted constant.

Environment: python 3.10, numpy 2.2.6, scipy 1.15.3, sympy (pip);
pdflatex (TeX Live 2022). Lean: elan toolchain
`leanprover/lean4:v4.33.0`, core only (no mathlib). One CPU for the
short set (≈ 15 min); the midpoint ladders are long gates (hours).

## 0. Build the paper

    pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex
    # → paper.pdf, 33 pages, no errors, no undefined references

## 1. Exact dual certificate — the consumption step (paper §s:lp)

    cd certification
    python3 certify_lp.py            # exact Fractions; < 1 s

Recorded 2026-08-17 (ALL constants identified: C5 = 1/36,
{4,2} = −23/420, {6} = −1/126; gpu/COMPUTATION_REPORT.md §4/4bis,
paper §5.5/D15-D16):
- P(x) at atoms (5323/10000, 6561/5000, 10293/5000): perfect
  square, P(0)=1, y₅<0, y₆>0 — UNBOUNDED support, no LP solver
- M5 = 101/18; M6 = 12809/1260 — EXACT rationals end to end
  (the {3,3} allowance retired, register D17)
- w₀ = 829278553005924403328783/8140995278473611944088783
  (asserted exactly against the paper and lean/RhGate/Certificate.lean)
- **1−2w₀ ≥ 0.7962, 1−w₀ ≥ 0.8981 — exact rational**
- 13/18 certificate re-check: E[Q] = 5/36 exact

## 2. Identity and face gate suite (paper §§5–10 faces)

    cd pipeline && python3 run_all.py         # ~60 s at T=9600

Recorded 2026-08-16 — ALL GATES PASS:
- local closure: 0 pointwise violations (b=4,5,6 × p=5,7,11 × s=1,2)
- Parseval exact; SW RMS 1.599e-03; dispersion 4.163e-03 / 0.73%
- arcs: 0 spacing violations; size tightness 24.8–31.3×
- Bell(5)=52 (chords 10/5), Bell(6)=203 (chords 30/15, matchings
  5/6/3/1); five-fold 0.06–1.29%, six-fold 0.01–0.73%
- consumption LP grid face: **0.79472/0.89736** at dx=0.002 and
  0.001 — consistent with the exact certificate 0.79471/0.89735
  (grid discretization ≥ certificate)

Height-trend table (paper §10): `python3 run_all.py T` for
T ∈ {2400 … 153600}.

## 3. Convolution calculus gates (paper §s:conv)

    cd certification
    python3 cyclic_cumulant.py check    # F-CYC / F-TRACE; ~1 min

Recorded: b=2 gate exact (∫O₂C₂ = 1/3 closed form, dev 0.00e+00);
∫O₃C₃ = 0 (m₃ = 2); partition-cyclic formula vs direct Ursell engine:
b=3 dev 4.4e-16, b=4 dev 3.1e-15, b=5 dev 3.4e-14 — ALL PASS.

## 4. Grouped-midpoint ladders — the connected constants (long gates)

    python3 midpoint_ladder.py 4 DV [i0 i1]   # b=4 calibration vs Φ₄ exact
    python3 midpoint_ladder.py 5 DV [i0 i1]   # C₅ ladder
    python3 midpoint_ladder.py 6 DV [i0 i1]   # {6} ladder  (checkpointed)

Recorded ladders (midpoint protocol, paper §s:conv; retired endpoint
values, register D10):
- b=4 calibration: matches Φ₄ = −1/60 and t_adj = 7/60 exactly at the
  ladder's own band (the audit that forced the anchor correction, D7)
- C₅: 0.02784/0.02781/0.02780 at dv=0.05/0.04/0.032 → **0.0278(1)**
- {4,2}: → **−0.0552(8)**;  {6}: → **−0.0078(8)**
- WARNING (D10): endpoint grids sign-flip at b≥6 coarse steps
  (b=6 at dv=0.4: +0.036 vs true −0.010); midpoint only.

## 5. Exact pairing layer, N-item and write-out suites

    cd certification
    python3 exact_t222.py anchors    # ~1 s:  t_adj = 7/60, t_opp = 1/30
    python3 exact_t222.py T0|T1|T2|T3  # 3/70, 1/90, 1/180, 1/70 (≤ 2 h)
    python3 certificate_family.py    # N1 payoff table (paper A5), exact
      # S0 at the corrected bands -> 0.79472/0.89736 (re-optimized atoms)
    python3 n2_c0_certification.py   # c₀ = γ−2 (N2 partial), ~3 min
    python3 writeouts_verification.py  # W1–W3 faces, ~1 min

    cd ../scripts
    python3 certificate_verification.py  # §3 certificate laws, exact
    python3 moment_lp_reopt.py         # threshold 0.0721
    python3 mu3_ledger_gates.py      # μ³ ledger gates
    python3 cumulant_engine.py   # C₂ exact; ∫O₃C₃ = 0
    python3 m4_band_split.py        # m₄ assembly 3.25103 (long)
    python3 gue_bands.py    # GUE bands (paper Eq. bands)
    python3 m5m6_bands.py # m₅/m₆ sine-model bands
    python3 r_class_measurement.py T     # direct R measurement (T = 600 … 38400)
    python3 landau_gonek.py      # Landau–Gonek calibration
    python3 finite_height_demo.py # finite-height illustration
    python3 m1_suite.py validate         # §4 constants, ~67 s
    python3 sawtooth_cert.py          # C_saw = 2.100
    python3 quadrature_cert.py          # tail ratio 0.500; c₀ stable

Data: ck_zeros.npy (first 1500 zero ordinates, for the calibration
faces). Helper modules babs_mean/t2_swaps/g1_ledger/mains_envelope/tail_bound are included
for the sawtooth chain (sawtooth_cert) — the dependency closure of scripts/
is complete (verified by AST import audit). The framework of paper §2 (matrix, counting, tail, first two
moments) is that of [C26], cited; its finite-height faces are the
calibration scripts above.

All recorded outputs match the values quoted in the paper (§10, §11,
Appendix A); the sawtooth, quadrature and §4 constants reproduce the
audit record digit-for-digit.

## 6. Lean — compiled identity + certificate layers (paper §11(f))

    cd lean/RhGate
    lake build                       # toolchain: leanprover/lean4:v4.33.0

COMPILED and kernel-checked (maintainers' machine, 2026-08-16;
final clean build 20:31, artifacts in .lake/):
- Anchors.lean — anchor-free assemblies 13/4, 67/12, 39/4 (∀ anchors),
  consumption arithmetic 13/18, 31/36 (`grind`; register D11)
- LocalClosure.lean — local law b=4, p=5: 125 lock classes, kernel
  `decide` on ℕ; **axiom-free**
- LocalClosure2.lean — local law b=4, p=7 (343) and b=5, p=5 (625),
  kernel `decide`, namespace LC2; **axiom-free**
- Certificate.lean — denominator-cleared dual-certificate identity,
  perfect-square nonnegativity (register D12: core lemma
  `Lean.Grind.OrderedRing.sq_nonneg`; `grind` treats (…)² as an
  opaque atom), normalization, BOTH corner evaluations, the
  corner-selection sign y₅+6y₆ < 0, headlines 1−2w₀ ≥ 7947/10000
  and 1−w₀ ≥ 8973/10000, monic expansion (A6), {2,2,2} = 131/420,
  instantiated anchor identities

4 modules, 20 theorems, 0 sorry. Axioms: propext / Classical.choice /
Quot.sound only; no sorryAx, no ofReduceBool. Roots cover every .lean
file in the directory. What Lean certifies is the rational-arithmetic
layer alone; the moment pinning, the Chebyshev–Markov measure step
and the analytic chain are graded in paper §11.

## 7. Constant → script map (constants quoted in the paper)

| constant | value | script |
|---|---|---|
| dual certificate & headline (EXACT) | 0.7962/0.8981 | certification/certify_lp.py + lean/RhGate/Certificate.lean |
| t_adj, t_opp, T₀–T₃, {2,2,2} (EXACT) | 7/60, 1/30, 3/70, 1/90, 1/180, 1/70, 131/420 | certification/exact_t222.py |
| C₅, {4,2}, {6} (ALL IDENTIFIED) | 1/36, −23/420, −1/126 | gpu/COMPUTATION_REPORT.md §4/4bis |
| F-CYC / F-TRACE gates | exact / ≤ 3.4e-14 | certification/cyclic_cumulant.py |
| identity faces, LP grid edge | ALL PASS; 0.79472/0.89736 | pipeline/run_all.py |
| certificate laws, κ(c), threshold | exact; 0.0721 | scripts/certificate_verification.py, moment_lp_reopt.py |
| §4 constants, C_saw, c₀ | −0.00545…, 2.100, γ−2 | scripts/m1_suite.py, sawtooth_cert.py, quadrature_cert.py |
| anchor-free 13/4, 67/12, 39/4; 13/18, 31/36 | exact, COMPILED | lean/RhGate/Anchors.lean |
| local law instances | 125/343/625 classes, COMPILED | lean/RhGate/LocalClosure.lean, LocalClosure2.lean |

## 8. Claim-grade summary (do not skip)

Machine-exact / exact-rational: certificate + κ(c) laws, local closure
(kernel-compiled), Parseval/product identities, Bell ledgers, pairing
layer, exact dual certificate (no LP solver, no support bound;
kernel-compiled), anchor-free assemblies (kernel-compiled).
Certified-candidate analytic layer: μ₃ ledger, truncation, Lemma D
spectral proof + transports, W1–W3, the §4 chain. Numerical with
stated bands, PROVED RATIONAL (§s:conv): C₅, {4,2}, {6} —
certification = identifying three fractions (N1; exact integrator
shipped, cluster scale). Open: N1 fractions, N2 remainder, external
review. Registry: 194 pre-registered checks, 160 passed, 34 fired &
converted (all forward-recorded; 31st–34th = D7, D10, D11, D12).
The paper's claims are advanced at certified-candidate grade — not
as records — per its §11 and the program's verification gate.
