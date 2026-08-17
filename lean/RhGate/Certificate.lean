/- Certificate.lean — core Lean only (no mathlib, no `ring`, no
   `decide` on Rat; `grind` for ring/linear goals, and the core
   ordered-ring lemma `Lean.Grind.OrderedRing.sq_nonneg` for the
   perfect-square step, since `grind` alone treats `(...)^2` as
   an opaque atom (paper register D12).

   The exact rational dual certificate of the consumption step
   (paper §9, Lemma "Exact dual certificate", and Appendix A6)
   at the IDENTIFIED constants (paper §5.5 / register D15):
   all three constants identified and the {3,3} allowance
   retired (register D15-D17): C5 = 1/36, {4,2} = -23/420,
   {6} = -1/126.  消费步的精确有理对偶证书 (全精确版):

     M5 = 67/12 + 1/36 = 101/18                      (exact);
     M6 = 39/4 + 131/420 + 1/6 - 23/420 - 1/126
        = 12809/1260                                 (exact);
     atoms a = 5323/10000, b = 6561/5000, c = 10293/5000;
     P(x) = [(x-a)(x-b)(x-c)]^2/(abc)^2, P >= 0, P(0) = 1;
     w0 = 829278553005924403328783/8140995278473611944088783,
     1 - 2 w0 >= 7962/10000,  1 - w0 >= 8981/10000.

   To keep every statement inside the ring fragment that `grind`
   handles, the polynomial identities are stated multiplied
   through by (abc)^2 — equivalent to the P-form since
   (abc)^2 > 0.  All statements are proxy-verified in exact
   Fraction arithmetic (certification/certify_lp.py).
   全部陈述已由精确分数运算代理核验.                              -/

namespace RhGate

/-- Denominator-cleared certificate identity:
    [(x-a)(x-b)(x-c)]^2 = (abc)^2 * sum_k y_k x^k.  A pure ring
    identity over Rat. -/
theorem cert_identity (x : Rat) :
    ((x - 5323/10000) * (x - 6561/5000) * (x - 10293/5000))^2
    = (129222147277358919747441/62500000000000000000000)
      * (1
         - (20200560698400422913/1562500000000000000
            * 62500000000000000000000/129222147277358919747441)*x
         + (39293368568783721/1250000000000000
            * 62500000000000000000000/129222147277358919747441)*x^2
         - (4746141355593/125000000000
            * 62500000000000000000000/129222147277358919747441)*x^3
         + (2422533313/100000000
            * 62500000000000000000000/129222147277358919747441)*x^4
         - (39031/5000
            * 62500000000000000000000/129222147277358919747441)*x^5
         + (62500000000000000000000/129222147277358919747441)*x^6)
    := by
  grind

/-- Nonnegativity of the numerator (a plain square). -/
theorem cert_nonneg (x : Rat) :
    0 ≤ ((x - 5323/10000) * (x - 6561/5000)
         * (x - 10293/5000))^2 :=
  Lean.Grind.OrderedRing.sq_nonneg

/-- Normalization: at x = 0 the squared cubic equals (abc)^2,
    i.e. P(0) = 1. -/
theorem cert_norm :
    ((0 - 5323/10000) * (0 - 6561/5000)
     * (0 - 10293/5000) : Rat)^2
    = 129222147277358919747441/62500000000000000000000 := by
  grind

/-- M5 is exact: 67/12 + 1/36 = 101/18 (identified C5 = 1/36). -/
theorem M5_exact : (67/12 : Rat) + 1/36 = 101/18 := by grind

/-- M6 fully exact:
    39/4 + 131/420 + 6·(1/36) − 23/420 − 1/126 = 12809/1260. -/
theorem M6_exact :
    (39/4 : Rat) + 131/420 + 6*(1/36) - 23/420 - 1/126
      = 12809/1260 := by
  grind

/-- Positivity of y6 (the upper-moment conversion direction). -/
theorem y6_pos :
    (0 : Rat) < 62500000000000000000000/129222147277358919747441
    := by
  grind

/-- The corner evaluation: sum of y_k against the pinned moments
    (1, 1, 4/3, 2, 13/4) and (M5, M6) equals the certified w0. -/
theorem w0_corner :
    (1 : Rat)
      - (20200560698400422913/1562500000000000000
         * 62500000000000000000000/129222147277358919747441) * 1
      + (39293368568783721/1250000000000000
         * 62500000000000000000000/129222147277358919747441) * (4/3)
      - (4746141355593/125000000000
         * 62500000000000000000000/129222147277358919747441) * 2
      + (2422533313/100000000
         * 62500000000000000000000/129222147277358919747441) * (13/4)
      - (39031/5000
         * 62500000000000000000000/129222147277358919747441)
        * (101/18)
      + (62500000000000000000000/129222147277358919747441)
        * (12809/1260)
    = 829278553005924403328783/8140995278473611944088783 := by
  grind

/-- Headline: simple-zeros constant, 1 - 2 w0 >= 0.7962. -/
theorem headline_simple :
    (7962/10000 : Rat)
      ≤ 1 - 2 * (829278553005924403328783/8140995278473611944088783)
    := by
  grind

/-- Headline: distinct-zeros constant, 1 - w0 >= 0.8981. -/
theorem headline_distinct :
    (8981/10000 : Rat)
      ≤ 1 - 829278553005924403328783/8140995278473611944088783 := by
  grind

/- ── Expansion, pairing arithmetic, anchor instances ────────── -/

/-- The certificate cubic's square, expanded monic (paper App. A6):
    ring identity, numerator side. -/
theorem cert_expand_monic (x : Rat) :
    ((x - 5323/10000) * (x - 6561/5000) * (x - 10293/5000))^2 =
      x^6 - (39031/5000)*x^5 + (2422533313/100000000)*x^4
      - (4746141355593/125000000000)*x^3
      + (39293368568783721/1250000000000000)*x^2
      - (20200560698400422913/1562500000000000000)*x
      + 129222147277358919747441/62500000000000000000000 := by
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
