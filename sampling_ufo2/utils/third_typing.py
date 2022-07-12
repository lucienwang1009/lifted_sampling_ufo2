from typing import TypeVar
from gmpy2 import mpq
from sympy import Poly

RingElement = TypeVar('RingElement', Poly, mpq)
