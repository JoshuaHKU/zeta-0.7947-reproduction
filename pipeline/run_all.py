# -*- coding: utf-8 -*-
"""Complete gate suite for the 0.7947 paper (preprint-0.8).

Usage: python3 run_all.py [T]      (default T = 9600; ~3 min)

Order: exact identities (local closure, Parseval, Bell gates),
per-bin SW face, dispersion face, arc faces, five- and six-fold
product faces, and the consumption LP with correlated bands.
"""
import sys
import time

import face_local_closure
import face_parseval
import face_sw
import face_dispersion
import face_arcs
import face_o5_gate
import face_b5_frame
import face_b6_frame
import face_lp_full


def main():
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 9600.0
    t0 = time.time()
    print(f"=== 0.7947 gate suite (T = {T:.0f}) ===")
    face_local_closure.main()
    ok = face_parseval.face(T)
    ok &= face_sw.face(T)
    ok &= face_dispersion.face(T)
    ok &= face_arcs.face(T)
    ok &= face_o5_gate.face()
    ok &= face_b5_frame.face(T)
    ok &= face_b6_frame.face(T)
    ok &= face_lp_full.face()
    print(f"=== {'ALL GATES PASS' if ok else 'A GATE FIRED'} "
          f"({time.time()-t0:.0f}s) ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
