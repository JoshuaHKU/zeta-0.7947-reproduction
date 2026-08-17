# Reproduction pipeline: "More than 0.7947 of the zeros..."

Complete standalone gate suite for the paper (numpy + scipy).

    python3 run_all.py            # full suite at T = 9600, ~3 min

| Module | Paper item | Criterion |
|---|---|---|
| face_local_closure | local law, all cycle lengths (paper §5) | exact, 7 configs |
| face_parseval | Parseval identity (paper §6) | exact ~1e-16 |
| face_sw | per-bin Siegel-Walfisz | RMS < 6e-3 at T >= 9600 |
| face_dispersion | comb identification | median tracking < 4% |
| face_arcs | well-spacing + size control | 0 violations; <= 100x |
| face_o5_gate | Bell(5) bookkeeping | exact 52; chords 10/5 |
| face_b5_frame | five-fold product identity | tracking <= 5% |
| face_b6_frame | Bell(6) + six-fold product | exact 203; <= 5% |
| face_lp_full | consumption LP grid face (paper §9) | edge 0.79472/0.89736 |

Height-trend tables (T = 2400..153600: the T^{-1/2} dispersion
law, the tenfold SW drop, product faces to 0.1%) reproduce by
running individual faces at each height. Registry: 194 checks,
160 passed, 34 fired-and-converted, forward-recorded (paper §11,
Appendix).
