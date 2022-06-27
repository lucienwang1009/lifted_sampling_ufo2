import functools
import operator

from typing import Callable, Dict, FrozenSet
from nnf import Var, NNF
from nnf.amc import eval
from sympy import Poly

# NOTE: presume we use rational field
from gmpy2 import mpq

from sampling_ufo2.fol.utils import to_nnf, to_nnf_var
from sampling_ufo2.fol.syntax import CNF, Lit, Pred


def poly_ring_wmc(node: NNF,
                  weights: Callable[[Var], Poly]) -> Poly:
    """Model counting of sd-DNNF sentences, weighted by variables.

    :param node: The sentence to measure.
    :param weights: A dictionary mapping variable nodes to weights.
    """
    # General ×
    # Non-idempotent +
    # Non-neutral +
    # = sd-DNNF
    return eval(node, operator.add, operator.mul, mpq(0), mpq(1), weights)


class WMC(object):
    def __init__(self, gnd_formula: CNF,
                 get_weight: Callable[[Pred], Poly],
                 ignore_cell_weight: bool = True):
        """
        Compute WMC of gnd_formula

        :param gnd_formula CNF: gounding formula in the form of CNF
        :param get_weight Callable[[Pred], Poly]: weighting function
        :param ignore_cell_weight bool: whether set the weight of unary and reflexive binary atom to 1
        """
        self.gnd_formula: CNF = gnd_formula
        self.get_weight: Callable[Pred, Poly] = get_weight
        self.nnf: NNF = to_nnf(gnd_formula)
        self.ignore_cell_weight = ignore_cell_weight
        self.var_weights: Dict[Var, Poly] = self._get_var_weights(
            self.get_weight
        )

        # for checking if the sentence is satisfiable (w.r.t. given evidences)
        self.dummy_var_weights = self._get_var_weights(
            lambda pred: (mpq(1), mpq(1))
        )

    def _get_var_weights(self, get_weight):
        weights = dict()
        for atom in self.gnd_formula.atoms():
            var = to_nnf_var(Lit(atom))
            if self.ignore_cell_weight and \
                    (len(atom.args) == 1 or all(arg == atom.args[0] for arg in atom.args)):
                weights[var], weights[var.negate()] = (mpq(1), mpq(1))
            else:
                weights[var], weights[var.negate()] = get_weight(atom.pred)
        return weights

    @functools.lru_cache(maxsize=None)
    def satisfiable(self, evidences: FrozenSet[Lit] = None) -> bool:
        mc = self._wmc_internal(evidences, self.dummy_var_weights)
        if mc == mpq(0):
            return False
        else:
            return True

    @functools.lru_cache(maxsize=None)
    def wmc(self, evidences: FrozenSet[Lit] = None,
            get_weight: Callable[[Pred], Poly] = None) -> Poly:
        if get_weight is not None:
            var_weights = self._get_var_weights(get_weight)
        else:
            var_weights = self.var_weights
        return self._wmc_internal(evidences, var_weights)

    def _wmc_internal(self, evidences=None, var_weights=None) -> Poly:
        """
        Compute WMC w.r.t. evidences

        :param evidences FrozenSet[Lit]:
        :param evidence_weight bool: whether multiply the weight of evidence
        :rtype Poly: WMC value
        """

        if evidences is None:
            evidences_vars = set()
        else:
            evidences_vars = set(
                to_nnf_var(lit) for lit in evidences
            )

        def weights_fn(var):
            if var in evidences_vars:
                return self.var_weights[var]
            elif var.negate() in evidences_vars:
                return mpq(0)
            return var_weights[var]
        return poly_ring_wmc(self.nnf, weights_fn)
