/- Anchors.lean — core Lean only (no mathlib; `grind` on Rat).

   Anchor-assembly identities of the paper "More than 0.7947 of
   the zeros of the Riemann zeta function are simple and on the
   critical line" (paper §5).  Key structural fact: the geometric
   anchors t_adj, t_opp CANCEL in the assembled moment
   expressions, which are therefore exact rational identities
   independent of the anchor values.

   锚装配恒等式 (论文 §5). 结构要点: 几何锚 t_adj, t_opp 在装配后
   的矩表达式中相互抵消, 故所得为与锚值无关的精确有理恒等式.

   m4:  1 + 2 + (2t_a + t_o) + (1/4 − 2t_a − t_o)      = 13/4
   m5:  1 + 10/3 + (10t_a + 5t_o) + 5(1/4 − 2t_a − t_o) = 67/12
   m6 pair+4 layer: 1 + 5 + (30t_a + 15t_o) + 15(1/4 − 2t_a − t_o)
                                                        = 39/4
   (paper §5: 9.7500 = 39/4.)                                    -/

-- Φ₄ as a function of the anchors.
def phi4 (ta to : Rat) : Rat := 1/4 - 2*ta - to

-- m₄ assembly: anchor-free value 13/4.
theorem m4_assembly (ta to : Rat) :
    1 + 2 + (2*ta + to) + phi4 ta to = 13/4 := by
  unfold phi4; grind

-- m₅ anchor-exact part: anchor-free value 67/12.
theorem m5_assembly (ta to : Rat) :
    1 + 10/3 + (10*ta + 5*to) + 5 * phi4 ta to = 67/12 := by
  unfold phi4; grind

-- m₆ pair+four layer: anchor-free value 39/4 (= 9.7500).
theorem m6_pair_layer (ta to : Rat) :
    1 + 5 + (30*ta + 15*to) + 15 * phi4 ta to = 39/4 := by
  unfold phi4; grind

-- Consumption calibration constants (13/18, 31/36 exactness).
theorem consume_13_18 : (1 : Rat) - 2 * (5/36) = 13/18 := by grind
theorem consume_31_36 : (1 : Rat) - 5/36 = 31/36 := by grind
