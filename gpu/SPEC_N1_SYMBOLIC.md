# N1 symbolic verification — exact-integrator specification
# N1 符号验证 —— 精确积分器规格（交付 EPYC/CPU 集群）

Goal / 目标: prove, in exact rational (Fraction) arithmetic, the
three identified constants (paper §5.5, register D15–D16):

    C₅ = 1/36        {4,2} = −23/420        {6} = −1/126

upgrading item N1 from *identification* to *proof*.  This is a
CPU/Fraction task (no GPU): the method is the paper's
Proposition p:t222 algorithm (exact iterated integration with
full breakpoint enumeration and the dual-scheme identical-rational
self-check), extended from 3D to 4D/5D.
从识别升级为证明。方法 = 论文命题 p:t222 的精确迭代积分
（断点全枚举 + 双方案同一有理数自检），由 3D 推广至 4D/5D。

## 1. Objects / 对象（与数值引擎逐字同构）

Each target is a signed sum of terms
∫ ov(walk_A) · ov(walk_B) [· min(|v|,1)] dv over [−2,2]^d:

| target | d | terms | structure |
|---|---|---|---|
| {4,2} | 4 | 3 placements × 26 C₄-terms = 78 | ov₆(placement walk) · ov₄(C₄-term walk) · min(\|v\|,1) |
| C₅    | 4 | 150 (P,σ) terms | ov₅(v-walk) · ov(merged walk) |
| {6}   | 5 | 1082 (P,σ) terms | ov₆(v-walk) · ov(merged walk) |

Walk definitions identical to `midpoint_ladder_gpu.py`
(`compile_terms`) and `spectator_42_reference.py`.  All kink
hyperplanes are ±1-combinations of variables at levels {0,±1}
(the family ℱ of paper A1) — unit leading coefficients throughout,
so the p:t222 machinery applies unchanged.

## 2. Algorithm / 算法（p:t222 的直接推广）

Per term, integrate variables innermost-out.  At each level:
(a) enumerate breakpoints of the current piecewise-polynomial
integrand: direct kinks (the ±1-forms), collisions of two inner
breakpoints, breakpoint–boundary collisions — all remain
small-integer linear forms in the outer variables because leading
coefficients are units; (b) on each kink-free piece integrate
exactly by closed Newton–Cotes at rational nodes (degree per
variable ≤ 2 per overlap factor ⟹ total degree ≤ 4–6; NC order
7/9 suffices); (c) SELF-CHECK: integrate each piece whole and in
two halves — identical rationals or abort (a missed kink forces a
discrepancy).  Sum terms with signs; compare with the target
fraction.  Any mismatch fires F-SYM-N1 (pre-registered).

## 3. Cost & parallelization / 成本与并行

3D reference (T₀…T₃ suite): ≤ 2 h single-core.  Per extra level
×(breakpoints ≈ 10–20).  Estimates (single EPYC core / 256 cores,
term-parallel — terms are independent, embarrassingly parallel):

| target | est. core-hours | 256-core wall |
|---|---|---|
| {4,2} (78 terms, 4D) | ~40–80 | < 1 h |
| C₅ (150 terms, 4D)   | ~80–160 | < 1 h |
| {6} (1082 terms, 5D) | ~10⁴–10⁵ | 2–20 days |

Recommendation: run {4,2} and C₅ first (verdicts within an hour);
{6} as a background batch with per-term checkpointing (each term's
exact rational logged; sum audited at the end).  A C++/GMP port of
the integrator (mirroring the mladder_fast.cu engineering) would
cut {6} to hours — worthwhile if the Python run projects long.

## 4. Deliverable protocol / 交付协议

Per term: (term-id, exact rational value, #pieces, self-check OK).
Final: Σ per target as a single fraction; assert equality with
1/36, −23/420, −1/126.  Partial sums are meaningful: {4,2}'s three
placement classes should individually equal −2/315, −1/1260,
−1/252 (per the componentwise identification), giving three
intermediate gates.

## 5. Calibration gates before production /投产前校准门

G1: the 4D integrator on the pure 4-cycle object must return
    Φ₄ = −1/60 exactly (known theorem, 26 terms — cheap).
G2: re-derive t_adj = 7/60 and {2,2,2}=131/420 with the same code
    path (3D regression vs exact_t222.py).
G3: dual-scheme identical-rational check active on every piece.

Status: **COMPLETED**.  The direct piecewise generalization
(exact_connected.py) closed {4,2} but was infeasible at b=6
(multiplicative piece growth); the polytope lift
(exact_polytope.py: each term one rational polytope volume,
exact facet recursion, ~10^9x reduction at b=6) evaluated all
1310 terms as exact fractions (gpu/exact/), summing to 1/36,
-23/420, -1/126 exactly.  All gates G1-G3 passed; F-SYM-N1 did
not fire; N1 upgraded identification → proof (paper §5.5,
register D18).  状态：已完成——多面体提升路线求出全部 1310 项
精确分数，三常数符号证明成立，N1 由识别升为证明。
