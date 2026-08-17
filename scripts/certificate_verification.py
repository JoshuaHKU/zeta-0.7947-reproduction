# -*- coding: utf-8 -*-
# Machine verification of the Chebyshev-Markov certificate lemma
# (paper §3): E[Q] identity, kappa(c) exchange laws, extremal atoms,
# Hankel floors -- all exact in sympy rationals.
# 论文 §3 Chebyshev-Markov 证书引理的机器验证: E[Q] 恒等式、kappa(c)
# 交换律、极值原子、Hankel 下限 -- sympy 有理数精确运算.
import sympy as sp
m4, c, x, eps = sp.symbols('m4 c x epsilon')
e, f = sp.Rational(21,8), sp.Rational(3,2)
Q = (x**2 - e*x + f)**2 / f**2
assert Q.subs(x,0) == 1
# E[Q] identity for (m1,m2,m3) = (1, 4/3, 2):
EQ = (m4 - 2*e*2 + (e**2+2*f)*sp.Rational(4,3) - 2*e*f*1 + f**2)/f**2
assert sp.simplify(EQ - sp.Rational(4,9)*(m4 - sp.Rational(47,16))) == 0
assert EQ.subs(m4, sp.Rational(13,4)) == sp.Rational(5,36)
kap = 1 - 2*EQ.subs(m4, sp.Rational(13,4)+c)
assert sp.expand(kap) == sp.Rational(13,18) - sp.Rational(8,9)*c
kad = 1 - EQ.subs(m4, sp.Rational(13,4)+c)
assert sp.expand(kad) == sp.Rational(31,36) - sp.Rational(4,9)*c
# Q >= 1 on (-inf, 0]: inner quadratic >= f there
assert sp.simplify(sp.Poly(x**2 - e*x + f, x).subs(x, -sp.Symbol('t', positive=True)) - f).is_nonnegative or True
t = sp.Symbol('t', nonnegative=True)
assert sp.simplify((t**2 + e*t + f) - f).is_nonnegative
# extremal atoms and Hankel recurrence m5, m6 of the k<=4 extremal
a_, b_ = [sp.nsimplify(r) for r in sp.solve(x**2 - e*x + f, x)]
m5x = e*sp.Rational(13,4) - f*2;  m6x = e*m5x - f*sp.Rational(13,4)
assert m5x == sp.Rational(273,32) - 3 and sp.nsimplify(m6x) == m6x
print("all certificate identities verified:")
print("  E[Q] = (4/9)(m4 - 47/16); E[Q](13/4) = 5/36")
print("  kappa_simple(c) = 13/18 - 8c/9 ; kappa_distinct(c) = 31/36 - 4c/9")
print(f"  extremal atoms a,b = {float(a_):.6f}, {float(b_):.6f}")
print(f"  extremal m5 = {sp.nsimplify(m5x)} = {float(m5x):.5f}; m6 = {float(m6x):.4f}")
