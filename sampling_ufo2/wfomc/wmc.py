import pandas as pd

from typing import Callable, FrozenSet, List

from sampling_ufo2.fol.syntax import CNF, Lit, Pred, Atom
from sampling_ufo2.utils import RingElement, Rational


class WMC(object):
    def __init__(self, gnd_formula: CNF,
                 get_weight: Callable[[Pred], RingElement],
                 ignore_cell_weight: bool = True):
        """
        Compute WMC of gnd_formula

        :param gnd_formula CNF: gounding formula in the form of CNF
        :param get_weight Callable[[Pred], RingElement]: weighting function
        :param ignore_cell_weight bool: whether set the weight of unary and reflexive binary atom to 1
        """
        self.gnd_formula: CNF = gnd_formula
        self.atoms: List[Atom] = list(self.gnd_formula.atoms())
        self.get_weight: Callable[Pred, RingElement] = get_weight
        self.ignore_cell_weight = ignore_cell_weight
        self.model_table: pd.DataFrame
        self.build_models()

    def build_models(self):
        table = []
        for model in self.gnd_formula.models():
            weight = Rational(1, 1)
            values = {}
            for lit in model:
                values[lit.atom] = lit.positive
                if (not self.ignore_cell_weight) or \
                        (not (len(lit.atom.args) == 1 or all(arg == lit.atom.args[0] for arg in lit.atom.args))):
                    weight *= (self.get_weight(lit.pred())[0] if lit.positive else
                               self.get_weight(lit.pred())[1])
            table.append([values[atom] for atom in self.atoms] + [weight])
        self.model_table = pd.DataFrame(
            table, columns=self.atoms + ['weight']
        )

    def conditional_on(self, evidences: FrozenSet[Lit] = None) -> pd.DataFrame:
        table = self.model_table
        for e in evidences:
            if e.atom in table.columns:
                if e.positive:
                    table = table[table[e.atom]]
                else:
                    table = table[~table[e.atom]]
                if table.empty:
                    return table
        return table
