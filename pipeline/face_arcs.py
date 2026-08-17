# -*- coding: utf-8 -*-
"""Face 5 -- well-spacing lemma and hybrid size-control (Sec. 7).

(a) Well-spacing (Lemma 7.1): the n-side image of an m-side Q-arc
    lies in at most b2 arms of exact image spacing b1*b2/N with
    total count <= 2*wid*N + b2.  PASS: zero violations over all
    mixed centers.
(b) Hybrid size control: per-(side, Q) strata, budget =
    min(union large sieve, K * global minor sup); assembly over
    strata against the measured mixed mass.  Both branches are
    classical.  PASS: tightness <= 100x per family (archive
    records 24-31x).
"""
import sys

import numpy as np

from common import build_spectra, arc_quality

P_CUT = 40
FAMS = [(5, 3), (5, 7), (3, 2)]


def classify(am2, an2, geo, b1, b2):
    """Per-bin arc classification: 0 = both minor, 1 = mixed,
    2 = both major; mixed centers keyed by (side, Q, center)."""
    N = geo['N']
    wid_m, wid_n = P_CUT / geo['Ym'], P_CUT / geo['Yn']
    cls = np.zeros(len(am2), dtype=np.int8)
    centers = {}
    for j in range(len(am2)):
        gm = (b2 * j / N) % 1.0
        gn = (b1 * j / N) % 1.0
        Qm = arc_quality(gm, P_CUT, wid_m)
        Qn = arc_quality(gn, P_CUT, wid_n)
        if Qm and Qn:
            cls[j] = 2
        elif Qm or Qn:
            cls[j] = 1
            if Qm:
                centers.setdefault(
                    ('m', Qm, round(gm * Qm) / Qm), []).append(j)
            else:
                centers.setdefault(
                    ('n', Qn, round(gn * Qn) / Qn), []).append(j)
    return cls, centers


def face(T):
    ok_all = True
    for (b1, b2) in FAMS:
        am2, an2, geo = build_spectra(T, b1, b2)
        cls, centers = classify(am2, an2, geo, b1, b2)
        N = geo['N']
        # (a) well-spacing: arms, arm step, count.
        bad = 0
        for (side, Q, c), js in centers.items():
            js = sorted(js)
            bb = b2 if side == 'm' else b1
            wid = P_CUT / (geo['Ym'] if side == 'm' else geo['Yn'])
            arms = {}
            for j in js:
                arms.setdefault(j % bb, []).append(j)
            if len(arms) > bb or len(js) > 2 * wid * N + bb + 1:
                bad += 1
            for jl in arms.values():
                if any((y - x) % bb for x, y in zip(jl, jl[1:])):
                    bad += 1
        # (b) hybrid size assembly against measured mixed mass.
        mix = 0.0
        for j in np.nonzero(cls == 1)[0]:
            w = 2.0 if 0 < j < len(am2) - 1 else 1.0
            mix += w * am2[j] * an2[j] / N
        lam_m = float(am2.sum()) / N
        lam_n = float(an2.sum()) / N
        gm_all = (b2 * np.arange(len(am2)) / N) % 1.0
        gn_all = (b1 * np.arange(len(am2)) / N) % 1.0
        maj_m = np.array([bool(arc_quality(g, P_CUT,
                                           P_CUT / geo['Ym']))
                          for g in gm_all])
        maj_n = np.array([bool(arc_quality(g, P_CUT,
                                           P_CUT / geo['Yn']))
                          for g in gn_all])
        sup_min_m = float(am2[~maj_m].max())
        sup_min_n = float(an2[~maj_n].max())
        strata = {}
        for (side, Q, c), js in centers.items():
            s = strata.setdefault((side, Q), [0.0, 0])
            arr = am2 if side == 'm' else an2
            s[0] = max(s[0], float(arr[np.array(js)].max()))
            s[1] += len(js)
        asm = 0.0
        for (side, Q), (mx, cnt) in strata.items():
            if side == 'm':
                budget = min((geo['Yn'] + N / b1) * lam_n / N,
                             cnt * sup_min_n / N)
            else:
                budget = min((geo['Ym'] + N / b2) * lam_m / N,
                             cnt * sup_min_m / N)
            asm += 2 * mx * budget
        rat = asm / max(mix, 1e-30)
        good = bad == 0 and rat <= 100
        ok_all &= good
        print(f"  ({b1},{b2}): spacing violations {bad}, "
              f"size tightness {rat:.1f}x "
              f"{'PASS' if good else 'FIRE'}")
    print(f"[arc faces] {'ALL PASS' if ok_all else 'FIRED'}")
    return ok_all


if __name__ == "__main__":
    face(float(sys.argv[1]) if len(sys.argv) > 1 else 9600.0)
