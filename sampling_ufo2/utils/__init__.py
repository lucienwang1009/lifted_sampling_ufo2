import numpy as np

from .multinomial import MultinomialCoefficients, multinomial
from .tree_sum import TreeSumContext, tree_sum
from .polynomial import Rational, expand, coeff_monomial, create_vars
from .third_typing import RingElement


AUXILIARY_PRED_NAME = 'aux'
TSEITIN_PRED_NAME = 'tseitin'
SKOLEM_PRED_NAME = 'skolem'
EVIDOM_PRED_NAME = 'evidom'
PREDS_FOR_EXISTENTIAL = [
    TSEITIN_PRED_NAME, SKOLEM_PRED_NAME, EVIDOM_PRED_NAME
]


def format_np_complex(num: np.ndarray) -> str:
    return '{num.real:+0.04f}+{num.imag:+0.04f}j'.format(num=num)


__all__ = [
    "MultinomialCoefficients",
    "multinomial",
    "TreeSumContext",
    'RingElement',
    'Rational',
    'expand',
    'coeff_monomial',
    'create_vars',
    "tree_sum",
    "format_np_complex",
    "AUXILIARY_PRED_NAME",
    "TSEITIN_PRED_NAME",
    "SKOLEM_PRED_NAME",
    "EVIDOM_PRED_NAME",
    "PREDS_FOR_EXISTENTIAL"
]
