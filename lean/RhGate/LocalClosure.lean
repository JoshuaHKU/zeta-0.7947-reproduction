/- LocalClosure.lean — core Lean only (kernel `decide` on Nat).

   Instance of the coincidence-counting local law (paper §5,
   local-closure proposition): for the affine chain
   n_{t+1} = a_t n_t + j_t (mod p) the count N(j) of n_1 with all
   points prime to p equals p − D(j), where
   D = #distinct{d_t mod p} and d_t = B_t · A_t⁻¹ (normalised
   offsets; A_{t+1} = a_t A_t, B_{t+1} = a_t B_t + j_t).

   重合计数局部律的实例 (论文 §5 局部闭包命题): 仿射链的容许计数
   N(j) = p − D(j), D 为规范化位移的相异个数.

   Verified by complete enumeration at b = 4, p = 5 over the full
   j-cube (125 lock vectors) — a decidable proposition with no
   axioms beyond core.  (A draft omitting the A_t⁻¹ division was
   caught by the exact-enumeration proxy: 56/125 violations;
   forward-recorded as the register's 29th conversion.)          -/

def powMod (x n p : Nat) : Nat :=
  match n with
  | 0 => 1 % p
  | Nat.succ m => (x * powMod x m p) % p

/-- Modular inverse of a unit via Fermat: x⁻¹ = x^(p−2) mod p. -/
def invMod (x p : Nat) : Nat := powMod x (p - 2) p

/-- Trajectory of the affine chain started at n1. -/
def chainPts (p n1 : Nat) (a j : List Nat) : List Nat :=
  let step := fun (st : Nat × List Nat) (aj : Nat × Nat) =>
    let nxt := (aj.1 * st.1 + aj.2) % p
    (nxt, st.2 ++ [nxt])
  ((a.zip j).foldl step (n1, [n1])).2

/-- N(j): the all-unit chain count over n1. -/
def chainCount (p : Nat) (a j : List Nat) : Nat :=
  (List.range p).foldl (fun acc n1 =>
    if (chainPts p n1 a j).all (fun x => x % p ≠ 0)
    then acc + 1 else acc) 0

/-- D(j): distinct normalised offsets d_t = B_t · A_t⁻¹ mod p,
    with B_t the n1 = 0 trajectory and A_t the coefficient
    products. -/
def distinctOffsets (p : Nat) (a j : List Nat) : Nat :=
  let Bs := chainPts p 0 a j
  let As := a.foldl (fun (l : List Nat) ai =>
    l ++ [(l.getLast! * ai) % p]) [1]
  let ds := (Bs.zip As).map
    (fun BA => (BA.1 * invMod BA.2 p) % p)
  ds.eraseDups.length

/-- The pointwise identity N = p − D over the full 5³ lock cube
    at b = 4, p = 5, coefficients (2,3,4). -/
def identity_b4_p5 : Bool :=
  (List.range 5).all (fun j1 =>
    (List.range 5).all (fun j2 =>
      (List.range 5).all (fun j3 =>
        chainCount 5 [2,3,4] [j1,j2,j3]
          = 5 - distinctOffsets 5 [2,3,4] [j1,j2,j3])))

theorem local_closure_b4_p5 : identity_b4_p5 = true := by decide
