# -*- coding: utf-8 -*-
"""Exact rational certification of the {2,2,2} pairing classes T0..T3
(and the 2D anchors t_adj, t_opp).

Method: iterated exact integration of the piecewise-polynomial integrand
    f = (1 - spread(positions))_+ * |v||w||u|
over [-1,1]^d (support: 0 is a position and every variable is a
position or a difference of two positions, so spread<=1 forces all
|vars|<=1 and the min(|.|,1) caps never bind).

All kink hypersurfaces have {-1,0,1}-coefficients and constants in
{0,+-1} (pairwise position differences = 0,+-1; variable = 0).  Under
iterated integration the piece structure of a partial integral changes
only at (i) direct kinks, (ii) collisions of two inner breakpoints,
(iii) breakpoint-boundary collisions -- all enumerated below as exact
rational candidate families (coefficients stay small integers because
every inner breakpoint is c - (integer combination) with unit leading
coefficient).

On every piece the integrand/partial integrals are polynomials of
degree <= 2 (in u), <= 4 (in w), <= 8 (in v); closed Newton-Cotes rules
of order 5/7/9 at rational nodes integrate these EXACTLY in Fraction
arithmetic.  Self-check: every piece is also integrated by the same
rule on the two halves of the piece; polynomial degree <= rule order
forces identical rationals (any discrepancy aborts).  A missed kink
would make the two schemes disagree on the containing piece.

Result target (recorded numerically in the audit ledger, proved by the
unified paper): T0=3/70, T1=1/90, T2=1/180, T3=1/70;
{2,2,2} = 5*T0+6*T1+3*T2+T3 = 131/420.

Usage: python3 exact_t222.py {T0|T1|T2|T3|anchors}
"""
import sys
from fractions import Fraction as F

ONE, ZERO = F(1), F(0)

# closed Newton-Cotes weights (exact for degree <= n for odd node count)
NC5 = [F(7,90), F(32,90), F(12,90), F(32,90), F(7,90)]          # deg 5
NC7 = [F(41,840),F(216,840),F(27,840),F(272,840),F(27,840),
       F(216,840),F(41,840)]                                     # deg 7
NC9 = [F(989,28350),F(5888,28350),F(-928,28350),F(10496,28350),
       F(-4540,28350),F(10496,28350),F(-928,28350),F(5888,28350),
       F(989,28350)]                                             # deg 9

def nc_int(g, x0, x1, wts):
    n = len(wts)-1
    h = x1-x0
    s = ZERO
    for i,wt in enumerate(wts):
        s += wt*g(x0 + h*F(i,n))
    return s*h

def nc_int_checked(g, x0, x1, wts, tag):
    a = nc_int(g, x0, x1, wts)
    xm = (x0+x1)/2
    b = nc_int(g, x0, xm, wts) + nc_int(g, xm, x1, wts)
    if a != b:
        raise RuntimeError(f"scheme disagreement at {tag}: [{x0},{x1}]")
    return a

# ---------------- class definitions ----------------
# positions as integer coefficient triples (cv,cw,cu): pos = cv*v+cw*w+cu*u
CLASSES = {
 'T0': [(0,0,0),(1,0,0),(0,0,0),(0,1,0),(0,0,0),(0,0,1)],
 'T1': [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,0),(0,0,1)],
 'T2': [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,1,1),(0,0,1)],
 'T3': [(0,0,0),(1,0,0),(1,1,0),(1,1,1),(0,1,1),(0,0,1)],
}
TARGET = {'T0':F(3,70),'T1':F(1,90),'T2':F(1,180),'T3':F(1,70)}

def make_f(pos):
    def f(v,w,u):
        vals = [cv*v+cw*w+cu*u for (cv,cw,cu) in pos]
        spread = max(vals)-min(vals)
        if spread >= 1: return ZERO
        return (1-spread)*abs(v)*abs(w)*abs(u)
    return f

def pair_diffs(pos):
    ds = set()
    for i in range(len(pos)):
        for j in range(i+1,len(pos)):
            d = tuple(a-b for a,b in zip(pos[i],pos[j]))
            if any(d): ds.add(d); ds.add(tuple(-x for x in d))
    return sorted(ds)

def run_class(name):
    pos = CLASSES[name]; f = make_f(pos)
    diffs = pair_diffs(pos)
    # u-breakpoint forms: for diffs with cu!=0 (|cu|=1): u = (c - cv*v - cw*w)/cu
    uforms = [(d, c) for d in diffs if d[2] != 0 for c in (-1,0,1)]
    # plus u = 0,+-1 (direct kink & box)
    def u_breaks(v,w):
        bs = {F(-1), F(0), F(1)}
        for (cv,cw,cu),c in uforms:
            bs.add(F(c - cv*v - cw*w, cu))
        return sorted(b for b in bs if -1 <= b <= 1)
    def I_u(v,w):
        bs = u_breaks(v,w); tot = ZERO
        for x0,x1 in zip(bs,bs[1:]):
            if x1 <= x0: continue
            tot += nc_int_checked(lambda u: f(v,w,u), x0, x1, NC5, 'u')
        return tot
    # w-candidates for fixed v: direct kinks (cu=0 diffs), w=0,+-1, and
    # collisions/boundary hits of u-breakpoints:
    #   (c1-cv1*v-cw1*w)/cu1 = (c2-cv2*v-cw2*w)/cu2  -> linear in w
    #   (c-cv*v-cw*w)/cu = +-1
    wlin = set()   # (aw, b_const, b_v):  aw*w = b_const + b_v*v
    for (cv,cw,cu),c in uforms:
        for e in (-1,0,1):   # boundary/level hits u = e
            # c - cv v - cw w = cu e  ->  cw w = c - cu e - cv v
            if cw: wlin.add((cw, c-cu*e, -cv))
    ufl = list(uforms)
    for i in range(len(ufl)):
        for j in range(i+1,len(ufl)):
            (d1,c1),(d2,c2) = ufl[i],ufl[j]
            aw = d1[1]*d2[2]-d2[1]*d1[2]
            if aw:
                wlin.add((aw, c1*d2[2]-c2*d1[2], -(d1[0]*d2[2]-d2[0]*d1[2])))
    for d in diffs:
        if d[2]==0 and d[1]!=0:
            for c in (-1,0,1): wlin.add((d[1], c, -d[0]))
    wlin = sorted(wlin)
    def w_breaks(v):
        bs = {F(-1),F(0),F(1)}
        for aw,b0,bv in wlin:
            bs.add(F(b0+bv*v, aw))
        return sorted(b for b in bs if -1 <= b <= 1)
    def I_w(v):
        bs = w_breaks(v); tot = ZERO
        for x0,x1 in zip(bs,bs[1:]):
            if x1 <= x0: continue
            tot += nc_int_checked(lambda w: I_u(v,w), x0, x1, NC7, 'w')
        return tot
    # v-candidates: fixed rational set: collisions of w-forms + direct
    vset = {F(-1),F(0),F(1)}
    for i in range(len(wlin)):
        for j in range(i+1,len(wlin)):
            a1,b1,c1v = wlin[i]; a2,b2,c2v = wlin[j]
            den = c1v*a2-c2v*a1
            if den:
                r = F(b2*a1-b1*a2, den)
                if -1 < r < 1: vset.add(r)
        # boundary hits w = e
        a1,b1,c1v = wlin[i]
        if c1v:
            for e in (-1,0,1):
                r = F(a1*e-b1, c1v)
                if -1 < r < 1: vset.add(r)
    for d in diffs:
        if d[1]==0==d[2] and d[0]!=0:
            for c in (-1,0,1):
                r = F(c,d[0])
                if -1 < r < 1: vset.add(r)
    vs = sorted(vset)
    tot = ZERO
    for x0,x1 in zip(vs,vs[1:]):
        if x1 <= x0: continue
        tot += nc_int_checked(I_w, x0, x1, NC9, 'v')
    return tot

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv)>1 else 'T0'
    if which == 'anchors':
        # 2D: t_adj walk [0,v,0,w]; t_opp walk [0,v,v+w,w]; weight |v||w|
        for nm,pos in (('t_adj',[(0,0,0),(1,0,0),(0,0,0),(0,1,0)]),
                       ('t_opp',[(0,0,0),(1,0,0),(1,1,0),(0,1,0)])):
            pos3 = pos  # embed in 3 vars, u unused
            def f2(v,w):
                vals=[cv*v+cw*w for (cv,cw,_) in pos3]
                s=max(vals)-min(vals)
                return ZERO if s>=1 else (1-s)*abs(v)*abs(w)
            diffs=pair_diffs(pos3)
            wl=set()
            for d in diffs:
                if d[1]!=0:
                    for c in (-1,0,1): wl.add((d[1],c,-d[0]))
            def wbr(v):
                bs={F(-1),F(0),F(1)}
                for aw,b0,bv in wl: bs.add(F(b0+bv*v,aw))
                return sorted(b for b in bs if -1<=b<=1)
            def Iw(v):
                t=ZERO
                for x0,x1 in zip(wbr(v),wbr(v)[1:]):
                    if x1>x0: t+=nc_int_checked(lambda w: f2(v,w),x0,x1,NC7,'w2')
                return t
            vset={F(-1),F(0),F(1)}
            wll=sorted(wl)
            for i in range(len(wll)):
                a1,b1,c1v=wll[i]
                if c1v:
                    for e in (-1,0,1):
                        r=F(a1*e-b1,c1v)
                        if -1<r<1: vset.add(r)
                for j in range(i+1,len(wll)):
                    a2,b2,c2v=wll[j]
                    den=c1v*a2-c2v*a1
                    if den:
                        r=F(b2*a1-b1*a2,den)
                        if -1<r<1: vset.add(r)
            for d in diffs:
                if d[1]==0 and d[0]!=0:
                    for c in (-1,0,1):
                        r=F(c,d[0])
                        if -1<r<1: vset.add(r)
            vs=sorted(vset); tot=ZERO
            for x0,x1 in zip(vs,vs[1:]):
                if x1>x0: tot+=nc_int_checked(Iw,x0,x1,NC9,'v2')
            print(f"{nm} = {tot} = {float(tot):.8f}")
    else:
        val = run_class(which)
        tgt = TARGET[which]
        print(f"{which} = {val} = {float(val):.10f}   target {tgt} "
              f"({'MATCH' if val==tgt else 'MISMATCH'})")
