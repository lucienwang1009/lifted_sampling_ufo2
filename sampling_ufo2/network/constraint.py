import numpy as np
import cmath

from abc import ABC
from typing import List, Dict, Tuple
from logzero import logger
from itertools import product
from copy import deepcopy

from sampling_ufo2.fol.syntax import Pred, Const, QuantifiedFormula
from sampling_ufo2.network.mln import MLN, ComplexMLN


class Constraint(ABC):
    pass


class TreeConstraint(Constraint):
    def __init__(self, pred: Pred):
        """
        Tree constraint

        :param pred Pred: the relation that forms a tree
        """
        super().__init__()
        self.pred = pred

    def __str__(self):
        return "Tree({})".format(self.pred)


class ArborescenceConstraint(Constraint):
    def __init__(self, pred: Pred, root_pred: Pred):
        """
        Arborescence (directed tree) constraint

        :param pred Pred: the relation that forms a arborescence
        :param root Const: the constant that represents the root of the tree
        """
        self.pred = pred
        self.root_pred = root_pred

    def __str__(self):
        return "DTree({}, {})".format(self.pred, self.root_pred)


class CardinalityConstraint(Constraint):
    def __init__(self, pred2card: Dict[Pred, int]):
        super().__init__()
        self.pred2card = pred2card

    def preds(self):
        return list(self.pred2card.keys())

    def check(self, cards: List[int]):
        return cards == tuple(self.pred2card.values())

    def dft(self, mln: MLN) -> Tuple[ComplexMLN, np.ndarray, np.ndarray]:
        new_formulas = []
        dft_domain = []
        top_weights = []
        M = []
        for pred in self.preds():
            cnf = QuantifiedFormula.from_pred(pred, list(mln.vars()))
            new_formulas.append(cnf)
            D_f = mln.domain_size() ** pred.arity
            dft_domain.append(range(D_f + 1))
            M.append(D_f + 1)

        new_weights = [[]] * len(new_formulas)
        M = np.array(M)
        logger.debug('dft domain: %s', dft_domain)
        d = np.prod(M)
        for i in product(*dft_domain):
            weight_for_constraint_formulas = complex(
                0, -2 * cmath.pi) * np.array(i) / M
            for k, f in enumerate(new_formulas):
                new_weights[k].append(weight_for_constraint_formulas[k])
            if not self.check(i):
                continue
            top_w = []
            for j in product(*dft_domain):
                top_w.append(cmath.exp(
                    complex(0, 2 * cmath.pi *
                            np.dot(np.array(i), np.array(j) / M))
                ))
            top_weights.append(top_w)
        top_weights = np.array(top_weights, dtype=np.complex256)
        new_weights = [np.array(w, dtype=np.complex256) for w in new_weights]
        formulas = mln.formulas + new_formulas
        weights = [
            np.tile(np.complex256(weight), int(d)) for weight in mln.weights
        ]
        weights.extend(new_weights)
        return ComplexMLN(formulas, weights, deepcopy(mln.domain)), top_weights, M

    def __str__(self):
        s = ''
        for pred, card in self.pred2card.items():
            s += '|{}| = {}\n'.format(pred, card)
        return s
