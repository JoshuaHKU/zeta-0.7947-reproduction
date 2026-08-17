/- Certificate.lean — core Lean only (no mathlib, no `ring`, no
   `decide` on Rat; `grind` for ring/linear goals, and the core
   ordered-ring lemma `Lean.Grind.OrderedRing.sq_nonneg` for the
   perfect-square step, since `grind` alone treats `(...)^2` as
   an opaque atom (paper register D12).

   The exact rational dual certificate of the consumption step
   (paper §9, Lemma "Exact dual certificate", and Appendix A6)
   at the IDENTIFIED constants (paper §5.5 / register D15):
   C5 = 1/36 and {6} = -1/126 identified at machine precision,
   {4,2} banded.  消费步的精确有理对偶证书 (识别常数版):

     M5 = 67/12 + 1/36 = 101/18  (exact);
     M6 = 39/4 + 131/420 + 1/6 - 1/126 - 544/10^4 + 2/10^4
        = 3202427/315000  ({4,2} band top + transport);
     atoms a = 2673/5000, b = 13149/10000, c = 10303/5000;
     P(x) = [(x-a)(x-b)(x-c)]^2/(abc)^2, P >= 0, P(0) = 1;
     w0 = 281255854405058410769981/2753785207121825824389981,
     1 - 2 w0 >= 7957/10000,  1 - w0 >= 8978/10000.

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
    ((x - 2673/5000) * (x - 13149/10000) * (x - 10303/5000))^2
    = (131132628910563134494761/62500000000000000000000)
      * (1
         - (40865740767194551461/3125000000000000000
            * 62500000000000000000000/131132628910563134494761)*x
         + (39629877598393353/1250000000000000
            * 62500000000000000000000/131132628910563134494761)*x^2
         - (2387347458831/62500000000
            * 62500000000000000000000/131132628910563134494761)*x^3
         + (2431693249/100000000
            * 62500000000000000000000/131132628910563134494761)*x^4
         - (39101/5000
            * 62500000000000000000000/131132628910563134494761)*x^5
         + (62500000000000000000000/131132628910563134494761)*x^6)
    := by
  grind

/-- Nonnegativity of the numerator (a plain square). -/
theorem cert_nonneg (x : Rat) :
    0 ≤ ((x - 2673/5000) * (x - 13149/10000)
         * (x - 10303/5000))^2 :=
  Lean.Grind.OrderedRing.sq_nonneg

/-- Normalization: at x = 0 the squared cubic equals (abc)^2,
    i.e. P(0) = 1. -/
theorem cert_norm :
    ((0 - 2673/5000) * (0 - 13149/10000)
     * (0 - 10303/5000) : Rat)^2
    = 131132628910563134494761/62500000000000000000000 := by
  grind

/-- M5 is exact: 67/12 + 1/36 = 101/18 (identified C5 = 1/36). -/
theorem M5_exact : (67/12 : Rat) + 1/36 = 101/18 := by grind

/-- M6 at the {4,2} band top:
    39/4 + 131/420 + 6·(1/36) − 1/126 − 544/10⁴ + 2/10⁴. -/
theorem M6_corner :
    (39/4 : Rat) + 131/420 + 6*(1/36) - 1/126
      - 544/10000 + 2/10000 = 3202427/315000 := by
  grind

/-- Band monotonicity: y6 > 0, so w0 is increasing in M6 and the
    {4,2} band top is the binding corner (M5 is exact). -/
theorem band_monotone :
    (0 : Rat) < 62500000000000000000000/131132628910563134494761
    := by
  grind

/-- The corner evaluation: sum of y_k against the pinned moments
    (1, 1, 4/3, 2, 13/4) and (M5, M6) equals the certified w0. -/
theorem w0_corner :
    (1 : Rat)
      - (40865740767194551461/3125000000000000000
         * 62500000000000000000000/131132628910563134494761) * 1
      + (39629877598393353/1250000000000000
         * 62500000000000000000000/131132628910563134494761) * (4/3)
      - (2387347458831/62500000000
         * 62500000000000000000000/131132628910563134494761) * 2
      + (2431693249/100000000
         * 62500000000000000000000/131132628910563134494761) * (13/4)
      - (39101/5000
         * 62500000000000000000000/131132628910563134494761)
        * (101/18)
      + (62500000000000000000000/131132628910563134494761)
        * (3202427/315000)
    = 281255854405058410769981/2753785207121825824389981 := by
  grind

/-- Headline: simple-zeros constant, 1 - 2 w0 >= 0.7957. -/
theorem headline_simple :
    (7957/10000 : Rat)
      ≤ 1 - 2 * (281255854405058410769981/2753785207121825824389981)
    := by
  grind

/-- Headline: distinct-zeros constant, 1 - w0 >= 0.8978. -/
theorem headline_distinct :
    (8978/10000 : Rat)
      ≤ 1 - 281255854405058410769981/2753785207121825824389981 := by
  grind

/- ── Expansion, pairing arithmetic, anchor instances ────────── -/

/-- The certificate cubic's square, expanded monic (paper App. A6):
    ring identity, numerator side. -/
theorem cert_expand_monic (x : Rat) :
    ((x - 2673/5000) * (x - 13149/10000) * (x - 10303/5000))^2 =
      x^6 - (39101/5000)*x^5 + (2431693249/100000000)*x^4
      - (2387347458831/62500000000)*x^3
      + (39629877598393353/1250000000000000)*x^2
      - (40865740767194551461/3125000000000000000)*x
      + 131132628910563134494761/62500000000000000000000 := by
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
