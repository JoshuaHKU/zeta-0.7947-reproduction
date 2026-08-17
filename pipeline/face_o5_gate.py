# -*- coding: utf-8 -*-
"""Face O5 -- the Bell(5) bookkeeping gate (Prop. pair-layer).

Enumerates all 52 set partitions of the 5-cycle, verifies the
class-count table (1/10/15/10/10/5/1), and the pentagon chord
classification of {2,2,1}: 10 non-crossing (t_adj) + 5 crossing
(t_opp).  These multiplicities, with the machine-exact frozen-slot
reductions and the certified b=4 anchors, close the pair layer:
anchor-exact part = 67/12.  PASS: all counts match.
"""


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def crossing(a, b):
    a1, a2 = sorted(a)
    b1, b2 = sorted(b)
    return (a1 < b1 < a2 < b2) or (b1 < a1 < b2 < a2)


def face():
    parts = list(set_partitions(list(range(5))))
    counts, adj, opp = {}, 0, 0
    for p in parts:
        sig = tuple(sorted((len(b) for b in p), reverse=True))
        counts[sig] = counts.get(sig, 0) + 1
        if sig == (2, 2, 1):
            pr = [b for b in p if len(b) == 2]
            if crossing(pr[0], pr[1]):
                opp += 1
            else:
                adj += 1
    expect = {(1, 1, 1, 1, 1): 1, (2, 1, 1, 1): 10, (2, 2, 1): 15,
              (3, 1, 1): 10, (3, 2): 10, (4, 1): 5, (5,): 1}
    ok = (len(parts) == 52 and counts == expect
          and (adj, opp) == (10, 5))
    print(f"[o5 gate] Bell(5) = {len(parts)}, classes "
          f"{'match' if counts == expect else 'MISMATCH'}, "
          f"chords {adj}/{opp} {'PASS' if ok else 'FIRE'}")
    return ok


if __name__ == "__main__":
    face()
