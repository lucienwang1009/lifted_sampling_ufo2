import random
import numpy as np
import functools

from typing import Dict, FrozenSet, List, Tuple, Callable
from itertools import product
from nnf import Var, NNF
from nnf.amc import eval

from sampling_ufo2.fol.utils import to_nnf, to_nnf_var
from sampling_ufo2.fol.syntax import CNF, Atom, Lit, Pred


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
        self.var_weights: Dict[Var, np.ndarray] = self._get_var_weights()

    def _get_var_weights(self):
        weights = dict()
        for atom in self.gnd_formula.atoms():
            var = to_nnf_var(Lit(atom))
            if self.ignore_cell_weight and \
                    (len(atom.args) == 1 or all(arg == atom.args[0] for arg in atom.args)):
                weights[var], weights[var.negate()] = (1, 1)
            else:
                weights[var], weights[var.negate()] = self.get_weight(atom.pred)
        return weights

    @functools.lru_cache(maxsize=None)
    def wmc(self, evidences: FrozenSet[Lit] = None) -> np.ndarray:
        """
        Compute WMC w.r.t. evidences

        :param evidences FrozenSet[Lit]:
        :param evidence_weight bool: whether multiply the weight of evidence
        :rtype np.ndarray: WMC value
        """
        evidences_vars = set(
            to_nnf_var(lit) for lit in evidences
        )

        def weights_fn(var):
            if var in evidences_vars:
                return self.var_weights[var]
            elif var.negate() in evidences_vars:
                return 0
            return self.var_weights[var]
        return np_wmc(self.nnf, weights_fn)


class WMCSampler(object):
    def __init__(self, wmc: WMC, evidences: FrozenSet[Lit]):
        self.wmc: WMC = wmc
        self.evidences: FrozenSet[Lit] = evidences
        self.unknown_atoms: List[Atom] = self._get_unknown_atoms()
        self.code: List[Tuple[bool]]
        self.dist: List[np.ndarray]
        # dist = [world: [weight_dim: ]]
        self.codes, self.dist = self._get_distribution()
        # logger.debug(
        #     'Evidences: %s', self.evidences
        # )
        # logger.debug(
        #     'Unknown atoms: %s', self.unknown_atoms
        # )
        # logger.debug('Distribution: %s', self.dist)

    def _get_unknown_atoms(self) -> FrozenSet[Atom]:
        unknown_atoms = []
        for atom in self.wmc.gnd_formula.atoms():
            if Lit(atom) not in self.evidences and \
                    Lit(atom, False) not in self.evidences:
                unknown_atoms.append(atom)
        return unknown_atoms

    def _get_distribution(self):
        codes = []
        dist = []
        for code in product(*([[True, False]] * len(self.unknown_atoms))):
            evidences_tmp = set(self.evidences)
            for idx, flag in enumerate(code):
                lit = Lit(self.unknown_atoms[idx])
                if not flag:
                    lit = lit.negate()
                evidences_tmp.add(lit)
            weight = self.wmc.wmc(frozenset(evidences_tmp))
            if np.all(weight != 0):
                dist.append(weight)
                codes.append(code)
        return codes, dist

    def decode(self, code: Tuple[bool]) -> FrozenSet[Atom]:
        decoded_atoms = set()
        for idx, atom in enumerate(self.unknown_atoms):
            if code[idx]:
                decoded_atoms.add(atom)
        return frozenset(decoded_atoms)

    def sample(self, k: int = 1) -> List[FrozenSet[Atom]]:
        if len(self.unknown_vars) == 0:
            return [frozenset()] * k
        samples = []
        sample_codes = random.choices(self.codes, self.dist, k=k)
        for code in sample_codes:
            samples.append(self.decode(code))
        return samples
