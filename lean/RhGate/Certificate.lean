/- Certificate.lean — core Lean only (no mathlib, no `ring`, no
   `decide` on Rat; `grind` for ring/linear goals, and the core
   ordered-ring lemma `Lean.Grind.OrderedRing.sq_nonneg` for the
   perfect-square step, since `grind` alone treats `(...)^2` as
   an opaque atom (paper register D12).

   The exact rational dual certificate of the consumption step
   (paper §9, Lemma "Exact dual certificate", and Appendix A6).
   消费步的精确有理对偶证书 (论文 §9 对偶证书引理与附录 A6).

   Connected-constant inputs (paper §5.5): C5 = 278/10^4 with band
   1/10^4; {4,2} = -552/10^4; {6} = -78/10^4; independent band
   18/10^4.  Then:

     band-worst corner  M5 = 168331/30000,  M6 = 42701/4200;
     certificate atoms  a = 27/50, b = 263/200, c = 103/50;
     P(x) = [(x-a)(x-b)(x-c)]^2 / (abc)^2,  P >= 0,  P(0) = 1;
     w0 := sum_k y_k m_k + y5 M5 + y6 M6
         = 1153107070889/11233957316589  (~ 0.102645),
     giving 1 - 2 w0 >= 7947/10000, 1 - w0 >= 8973/10000.

   To keep every statement inside the ring fragment that `grind`
   handles, the polynomial identities are stated multiplied
   through by (abc)^2 = 534950348409/(25·10^10) — equivalent to
   the P-form since (abc)^2 > 0.  All statements are
   proxy-verified in exact Fraction arithmetic
   (certification/certify_lp.py). 全部陈述已由精确分数运算代理核验. -/

namespace RhGate

/-- Denominator-cleared certificate identity: (10^10/534950348409)
    times the squared cubic equals the y-polynomial; equivalently
    [(x-a)(x-b)(x-c)]^2 = (abc)^2 * sum_k y_k x^k.  A pure ring
    identity over Rat. -/
theorem cert_identity (x : Rat) :
    ((x - 27/50) * (x - 263/200) * (x - 103/50))^2
    = (534950348409/250000000000)
      * (1 - (4531400/731403)*x
         + (7996839235000/534950348409)*x^2
         - (118538500000/6604325289)*x^3
         + (6097506250000/534950348409)*x^4
         - (72500000000/19812975867)*x^5
         + (250000000000/534950348409)*x^6) := by
  grind

/-- Nonnegativity of the numerator (a plain square); with
    (abc)^2 > 0 this gives P >= 0 on all of Rat.

    NB: `grind` alone does NOT close this — its ring/linarith core
    treats `(...)^2` as an opaque atom and never learns it is a
    square.  Core Lean does ship the ordered-ring square lemma
    `Lean.Grind.OrderedRing.sq_nonneg`, and `Init.Grind.Ordered.Rat`
    supplies the `Rat` instance, so this stays mathlib-free. -/
theorem cert_nonneg (x : Rat) :
    0 ≤ ((x - 27/50) * (x - 263/200) * (x - 103/50))^2 :=
  Lean.Grind.OrderedRing.sq_nonneg

/-- Normalization: at x = 0 the squared cubic equals (abc)^2,
    i.e. P(0) = 1. -/
theorem cert_norm :
    ((0 - 27/50) * (0 - 263/200) * (0 - 103/50) : Rat)^2
    = 534950348409/250000000000 := by
  grind

/-- The corner evaluation: sum of y_k against the pinned moments
    (1, 1, 4/3, 2, 13/4) and the band-worst corner (M5, M6) equals
    the certified origin mass w0. -/
theorem w0_corner :
    (1 : Rat) - (4531400/731403) * 1
      + (7996839235000/534950348409) * (4/3)
      - (118538500000/6604325289) * 2
      + (6097506250000/534950348409) * (13/4)
      - (72500000000/19812975867) * (168331/30000)
      + (250000000000/534950348409) * (42701/4200)
    = 1153107070889/11233957316589 := by
  grind

/-- Headline: simple-zeros constant, 1 - 2 w0 >= 0.7947. -/
theorem headline_simple :
    (7947/10000 : Rat) ≤ 1 - 2 * (1153107070889/11233957316589) := by
  grind

/-- Headline: distinct-zeros constant, 1 - w0 >= 0.8973. -/
theorem headline_distinct :
    (8973/10000 : Rat) ≤ 1 - 1153107070889/11233957316589 := by
  grind

/- ── The corner selection, inside Lean ─────────────────────────
   The band family is affine in the single parameter δ (the same
   δ enters M5 once and M6 six times), so
   dw0/dδ = y5 + 6·y6; its sign decides which corner binds.     -/

/-- dw0/dδ = y5 + 6 y6 < 0: w0 is DECREASING in δ, so the binding
    (band-worst) corner is δ = -1/10000, i.e. w0_corner above. -/
theorem corner_monotone :
    (-(72500000000/19812975867) + 6 * (250000000000/534950348409)
      : Rat) < 0 := by
  grind

/-- The other corner (δ = +1/10000: M5 = 168337/30000,
    M6 = 1067651/105000) evaluates strictly lower — the affine
    family's two extreme values, bracketing every band point. -/
theorem w0_corner_upper :
    (1 : Rat) - (4531400/731403) * 1
      + (7996839235000/534950348409) * (4/3)
      - (118538500000/6604325289) * 2
      + (6097506250000/534950348409) * (13/4)
      - (72500000000/19812975867) * (168337/30000)
      + (250000000000/534950348409) * (1067651/105000)
    = 1151185570889/11233957316589 := by
  grind

theorem corner_order :
    (1151185570889/11233957316589 : Rat)
      ≤ 1153107070889/11233957316589 := by
  grind

/- ── Expansion, pairing arithmetic, anchor instances ────────── -/

/-- The certificate cubic's square, expanded monic (paper App. A6):
    ring identity, numerator side. -/
theorem cert_expand_monic (x : Rat) :
    ((x - 27/50) * (x - 263/200) * (x - 103/50))^2 =
      x^6 - (783/100)*x^5 + (975601/40000)*x^4
      - (19203237/500000)*x^3 + (1599367847/50000000)*x^2
      - (16571397771/1250000000)*x + 534950348409/250000000000 := by
  grind

/-- The exact {2,2,2} value (certified by exact_t222.py):
    5·(3/70) + 6·(1/90) + 3·(1/180) + 1/70 = 131/420. -/
theorem t222_exact :
    5*(3/70 : Rat) + 6*(1/90) + 3*(1/180) + 1/70 = 131/420 := by
  grind

/-- Anchor-free m₅ layer INSTANTIATED at the exact anchors
    t_adj = 7/60, t_opp = 1/30 (Anchors.lean proves the ∀-anchor
    form; this pins the certified values of exact_t222.py). -/
theorem anchor_free_m5 :
    10*(7/60 : Rat) + 5*(1/4 - 2*(7/60) - 1/30) + 5*(1/30) = 5/4 := by
  grind

theorem anchor_free_m6 :
    30*(7/60 : Rat) + 15*(1/30) + 15*(1/4 - 2*(7/60) - 1/30) = 15/4 := by
  grind

end RhGate
