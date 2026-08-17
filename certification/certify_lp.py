# -*- coding: utf-8 -*-
"""Exact rational dual (SOS) certificate for the consumption LP.

Certificate polynomial (atoms a=27/50, b=263/200, c=103/50):

    P(x) = [(x-a)(x-b)(x-c)]^2 / (abc)^2 .

P is a perfect square (hence >= 0 on all of R, SOS trivially), P(0)=1,
its x^5-coefficient is negative and x^6-coefficient positive, so for
ANY probability measure nu on [0,oo) with
    m1=1, m2=4/3, m3=2, m4=13/4  (pinned),
    m5 >= M5,  m6 <= M6,
one has  nu({0}) <= int P dnu = sum_k y_k m_k <= y0 + y1 + (4/3)y2
+ 2 y3 + (13/4) y4 + y5 M5 + y6 M6  (y5<=0, y6>0 give the right
inequality directions).  No support bound and no LP solver enter: the
bound is exact rational arithmetic.  Verified here at BOTH corners of
the correlated C5-band (the bound is affine in the band parameter d,
so the corner maximum is the band maximum).

Moment inputs (preprint, corrected grouped-midpoint constants):
    M5(d) = 67/12 + 0.0278 + d,       |d| <= 0.0001
    M6(d) = 39/4 + 131/420 + 6*(0.0278+d) - 0.0552 - 0.0078 + 0.0018
with {2,2,2} = 131/420 EXACT (exact_t222.py), {4,2} and {6} at their
corrected bands (the 0.0018 = 0.0002+0.0008+0.0008 independent band;
see paper S[s:conv] for the midpoint protocol and D10 for the retired
endpoint-grid values).

Output: exact rational w0-bound and the theorem constants
    1-2w0 >= 0.7947,  1-w0 >= 0.8973 .

Statements mirrored in core Lean (lean/RhGate/Certificate.lean).
"""
from fractions import Fraction as F

def polymul(p,q):
    r=[F(0)]*(len(p)+len(q)-1)
    for i,pi in enumerate(p):
        for j,qj in enumerate(q): r[i+j]+=pi*qj
    return r

a,b,c = F(27,50), F(263,200), F(103,50)
cub = polymul(polymul([-a,F(1)],[-b,F(1)]),[-c,F(1)])
num = polymul(cub,cub)
n2 = (a*b*c)**2
y = [ck/n2 for ck in num]                     # y0..y6

assert y[0] == 1, "P(0) != 1"
assert y[5] < 0 and y[6] > 0, "sign conditions fail"
# perfect square by construction; independent spot check at x=1/3, 7/5:
for x in (F(1,3), F(7,5)):
    Px = sum(y[k]*x**k for k in range(7))
    Qx = ((x-a)*(x-b)*(x-c))**2/n2
    assert Px == Qx and Px >= 0

mom = [F(1),F(1),F(4,3),F(2),F(13,4)]
C5, dC5, band6 = F(278,10000), F(1,10000), F(18,10000)
t222 = 5*F(3,70)+6*F(1,90)+3*F(1,180)+F(1,70)
assert t222 == F(131,420)

worst = None
for d in (-dC5, dC5):                          # affine in d => corners
    M5 = F(67,12) + C5 + d
    M6 = F(39,4) + t222 + 6*(C5+d) + F(-552,10000) + F(-78,10000) + band6
    w0 = sum(y[k]*mom[k] for k in range(5)) + y[5]*M5 + y[6]*M6
    print(f"corner d={float(d):+.4f}:  w0 <= {w0}  = {float(w0):.9f}")
    if worst is None or w0 > worst: worst = w0

print("\nband-worst  w0 <=", worst, "=", float(worst))
print("1-2w0 =", 1-2*worst, "=", float(1-2*worst))
print("1-w0  =", 1-worst,  "=", float(1-worst))
assert worst == F(1153107070889, 11233957316589), "w0 mismatch vs paper"
assert 1-2*worst >= F(7947,10000), "0.7947 fails"
assert 1-worst  >= F(8973,10000), "0.8973 fails"
print("\nCERTIFIED (exact rational arithmetic, unbounded support):")
print("  N0^s/N >= 1-2w0 >= 0.7947      N_d/N >= 1-w0 >= 0.8973")

# 13/18 certificate re-check (Lemma 3.1 polynomial):
Q21 = polymul([F(3,2),-F(21,8),F(1)],[F(3,2),-F(21,8),F(1)])
yQ = [ck/F(9,4) for ck in Q21]
EQ = sum(yQ[k]*mom[k] for k in range(5))
assert EQ == F(5,36) and yQ[0]==1
print("  13/18 certificate: E[Q] at (1,4/3,2,13/4) = 5/36 exact  OK")
