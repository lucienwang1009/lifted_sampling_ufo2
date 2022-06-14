import numpy as np
import functools

from typing import Callable, Dict, FrozenSet
from nnf import Var, NNF
from nnf.amc import eval

from sampling_ufo2.fol.utils import to_nnf, to_nnf_var
from sampling_ufo2.fol.syntax import CNF, Lit, Pred


def np_wmc(node: NNF, weights: Callable[[Var], np.ndarray]) -> np.ndarray:
    """Model counting of sd-DNNF sentences, weighted by variables.

    :param node: The sentence to measure.
    :param weights: A dictionary mapping variable nodes to weights.
    """
    # General ×
    # Non-idempotent +
    # Non-neutral +
    # = sd-DNNF
    return eval(node, np.add, np.multiply, 0.0, 1.0, weights)


class WMC(object):
    def __init__(self, gnd_formula: CNF, get_weight: Callable[[Pred], np.ndarray], ignore_cell_weight: bool = True):
        """
        Compute WMC of gnd_formula

        :param gnd_formula CNF: gounding formula in the form of CNF
        :param get_weight Callable[[Pred], np.ndarray]: weighting function
        :param ignore_cell_weight bool: whether set the weight of unary and reflexive binary atom to 1
        """
        self.gnd_formula: CNF = gnd_formula
        self.get_weight: Callable[Pred, np.ndarray] = get_weight
        self.nnf: NNF = to_nnf(gnd_formula)
        self.ignore_cell_weight = ignore_cell_weight
        self.var_weights: Dict[Var, np.ndarray] = self._get_var_weights(
            self.get_weight
        )

        # for checking if the sentence is satisfiable (w.r.t. given evidences)
        self.dummy_var_weights = self._get_var_weights(lambda pred: (1, 1))

    def _get_var_weights(self, get_weight):
        weights = dict()
        for atom in self.gnd_formula.atoms():
            var = to_nnf_var(Lit(atom))
            if self.ignore_cell_weight and \
                    (len(atom.args) == 1 or all(arg == atom.args[0] for arg in atom.args)):
                weights[var], weights[var.negate()] = (1, 1)
            else:
                weights[var], weights[var.negate()] = get_weight(atom.pred)
        return weights

    @functools.lru_cache(maxsize=None)
    def satisfiable(self, evidences: FrozenSet[Lit] = None) -> bool:
        mc = self._wmc_internal(evidences, self.dummy_var_weights)
        if np.all(mc == 0):
            return False
        else:
            return True

    @functools.lru_cache(maxsize=None)
    def wmc(self, evidences: FrozenSet[Lit] = None, get_weight: Callable[[Pred], np.ndarray] = None) -> np.ndarray:
        if get_weight is not None:
            var_weights = self._get_var_weights(get_weight)
        else:
            var_weights = self.var_weights
        return self._wmc_internal(evidences, var_weights)

    def _wmc_internal(self, evidences=None, var_weights=None) -> np.ndarray:
        """
        Compute WMC w.r.t. evidences

        :param evidences FrozenSet[Lit]:
        :param evidence_weight bool: whether multiply the weight of evidence
        :rtype np.ndarray: WMC value
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
                return 0
            return var_weights[var]
        return np_wmc(self.nnf, weights_fn)
