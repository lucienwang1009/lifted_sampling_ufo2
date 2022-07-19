from symengine import Rational, var
from sympy import Poly

Rational = Rational
Poly = Poly


def create_vars(conf):
    return var(conf)


def expand(polynomial):
    return polynomial.expand()


def coeff_monomial(polynomial, monomial) -> Rational:
    return polynomial.as_coefficients_dict()[monomial]


# from gmpy2 import mpq
# from sympy import Poly, symbols
# Rational = mpq
# Poly = Poly
#
#
# def create_vars(conf):
#     return symbols(conf)
#
#
# def expand(polynomial):
#     return Poly(polynomial)
#
#
# def coeff_monomial(polynomial, monomial) -> Rational:
#     return polynomial.coeff_monomial(monomial)
