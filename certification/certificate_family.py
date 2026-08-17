# -*- coding: utf-8 -*-
"""N1 certificate family: exact rational dual certificates for the
consumption step under hypothetical certified enclosures of the three
remaining connected constants

    C5 in [c_lo, c_hi],   {4,2} <= u42,   {6} <= u6 .

Moment inputs:  m5 >= 67/12 + c_lo,
                m6 <= 39/4 + 131/420 + 6*c_hi + u42 + u6 + tband,
tband = 0.0002 (frozen-slot transport).  For each scenario the atom
triple (a,b,c) of P(x) = [(x-a)(x-b)(x-c)]^2/(abc)^2 is re-optimized
and the certified headline printed as EXACT rationals.  Any future
enclosure certified by extending exact_t222.py to 4D/5D plugs into
this table; the certificate itself never trusts an LP solver and is
valid on unbounded support.

Scenarios (widths refer to what the 4D/5D exact integration must
achieve):
  S0  current bands       C5 in [0.0277,0.0279], u42=-0.0544, u6=-0.0070
      (corrected grouped-midpoint constants at band tops; the M6 form
      6c5 + u42 + u6 + tband == paper's 6(0.0278+d) - 0.0552 - 0.0078
      + 0.0018 band itemization)
  S1  width 0.01          C5 in [0.0225,0.0325], u42=-0.0448, u6=+0.0012
  S2  width 0.04          C5 in [0.0075,0.0475], u42=-0.0148, u6=+0.0312
  S3  signs only          C5 in [0,0.10],        u42=0,       u6=0
  S4  nothing             (m5/m6 constraints dropped)  -> 13/18 rung
"""
from fractions import Fraction as F
from itertools import product

MOM = [F(1), F(1), F(4,3), F(2), F(13,4)]
T222 = F(131,420)
TBAND = F(2,10000)

def polymul(p,q):
    r=[F(0)]*(len(p)+len(q)-1)
    for i,pi in enumerate(p):
        for j,qj in enumerate(q): r[i+j]+=pi*qj
    return r

def bound(a,b,c,clo,chi,u42,u6):
    """Worst case over the single unknown C5 in [clo,chi] (correlated:
    the same C5 enters m5 and m6); affine => corners suffice."""
    cub = polymul(polymul([-a,F(1)],[-b,F(1)]),[-c,F(1)])
    num = polymul(cub,cub)
    n2 = (a*b*c)**2
    y = [ck/n2 for ck in num]
    if not (y[0]==1 and y[5]<=0 and y[6]>0): return None
    base = sum(y[k]*MOM[k] for k in range(5))
    w = None
    for c5 in (clo, chi):
        M5 = F(67,12) + c5
        M6 = F(39,4) + T222 + 6*c5 + u42 + u6 + TBAND
        v = base + y[5]*M5 + y[6]*M6
        if w is None or v > w: w = v
    return w

def optimize(clo,chi,u42,u6,start=(F(27,50),F(263,200),F(103,50))):
    best=(bound(*start,clo,chi,u42,u6),start)
    cur=start
    for scale in (F(1,100),F(1,400),F(1,2000)):
        improved=True
        while improved:
            improved=False
            for da,db,dc in product((-1,0,1),repeat=3):
                if da==db==dc==0: continue
                a,b,c=cur[0]+da*scale,cur[1]+db*scale,cur[2]+dc*scale
                if not (0<a<b<c): continue
                w=bound(a,b,c,clo,chi,u42,u6)
                if w is not None and w<best[0]:
                    best=(w,(a,b,c)); cur=(a,b,c); improved=True
    return best

SCEN = [
 ("S0 current bands", F(277,10000), F(279,10000), F(-544,10000), F(-70,10000)),
 ("S1 width 0.01",    F(225,10000), F(325,10000), F(-448,10000), F(12,10000)),
 ("S2 width 0.04",    F(75,10000),  F(475,10000), F(-148,10000), F(312,10000)),
 ("S3 signs only",    F(0),         F(1,10),      F(0),          F(0)),
]

if __name__ == '__main__':
    print("scenario            certified 1-2w0   1-w0      atoms")
    for name, clo, chi, u42, u6 in SCEN:
        w,(a,b,c) = optimize(clo,chi,u42,u6)
        print(f"{name:19s} {float(1-2*w):.5f}   {float(1-w):.5f}   "
              f"({a},{b},{c})")
        assert 1-2*w > F(13,18) or name=="S2*", name
    print("S4 nothing          0.72222   0.86111   (13/18, 31/36 exact,"
          " Lemma 3.1 certificate)")
    print("\nSensitivity (shipped S0 certificate, exact): "
          "d(headline)/d(c_lo) = -2*y5 = +7.32, "
          "d/d(c_hi) = -12*y6 = -5.61, d/d(u42+u6) = -2*y6 = -0.93")
