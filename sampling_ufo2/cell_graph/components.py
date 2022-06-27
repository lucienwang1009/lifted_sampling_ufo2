from __future__ import annotations

import pandas as pd
import functools

from typing import FrozenSet, List, Set, Tuple
from dataclasses import dataclass, field
from itertools import product
from logzero import logger
from sympy import Poly
from gmpy2 import mpq

from sampling_ufo2.fol.syntax import Atom, Lit, Pred, Term, a, b, x
from sampling_ufo2.wfomc.wmc import WMC


@dataclass(frozen=True)
class Cell(object):
    """
    In other words, the Unary types
    """
    code: Tuple[bool] = field(hash=False, compare=False)
    preds: Tuple[Pred] = field(hash=False, compare=False)
    # for hashing
    _identifier: FrozenSet[Tuple[Pred, bool]] = field(
        default=None, repr=False, init=False, hash=True, compare=True)

    def __post_init__(self):
        object.__setattr__(self, '_identifier',
                           frozenset(zip(self.preds, self.code)))

    def get_evidences(self, term: Term) -> FrozenSet[Lit]:
        evidences = set()
        for i, p in enumerate(self.preds):
            atom = p(*([term] * p.arity))
            if (self.code[i]):
                evidences.add(Lit(atom))
            else:
                evidences.add(Lit(atom, False))
        return frozenset(evidences)

    def is_positive(self, pred: Pred) -> bool:
        return self.code[self.preds.index(pred)]

    def negate(self, pred: Pred) -> Cell:
        idx = self.preds.index(pred)
        new_code = list(self.code)
        new_code[idx] = not new_code[idx]
        return Cell(tuple(new_code), self.preds)

    def drop_pred(self, pred: Pred) -> Cell:
        new_code, new_preds = zip(
            *[(c, p) for c, p in zip(self.code, self.preds) if p != pred])
        return Cell(tuple(new_code), tuple(new_preds))

    def __str__(self):
        evidences: Set[Lit] = self.get_evidences(x)
        return '^'.join(str(lit) for lit in evidences)

    def __repr__(self):
        return self.__str__()


class BtypeTable(object):
    def __init__(self, wmc: WMC, cell_1: Cell, cell_2: Cell):
        """
        The table containing the weight of all B-types.
        Note the order of cell_1 and cell_2 matters!
        """
        self.wmc: WMC = wmc
        self.cell_1: Cell = cell_1
        self.cell_2: Cell = cell_2

        self.evidences: FrozenSet[Lit] = frozenset(
            self.cell_1.get_evidences(a).union(
                self.cell_2.get_evidences(b)
            )
        )
        self.unknown_atoms: List[Atom] = self._get_unknown_atoms()
        self.table: pd.DataFrame = self.build_table()
        # logger.debug('Btype table:\n%s', self.table.to_markdown())

    def build_table(self) -> None:
        table = []
        for code in product(*([[True, False]] * len(self.unknown_atoms))):
            evidences = set(self.evidences)
            for idx, flag in enumerate(code):
                lit = Lit(self.unknown_atoms[idx])
                if not flag:
                    lit = lit.negate()
                evidences.add(lit)
            if not self.wmc.satisfiable(frozenset(evidences)):
                continue
            weight = self.wmc.wmc(frozenset(evidences))
            table.append(list(code) + [weight])
        table = pd.DataFrame(table, columns=self.unknown_atoms + ['weight'])
        return table

    def _condition(self, evidences: FrozenSet[Lit] = None) -> pd.DataFrame:
        if evidences is None or len(evidences) == 0:
            return self.table

        df = self.table
        for lit in evidences:
            if lit.positive:
                df = df[df[lit.atom]]
            else:
                df = df[~df[lit.atom]]
        return df

    @functools.lru_cache(maxsize=None)
    def get_weight(self, evidences: FrozenSet[Lit] = None) -> Poly:
        df = self._condition(evidences)
        if len(df) == 0:
            logger.warning(
                'Cell pair (%s, %s) with evidences %s is not satisfiable',
                self.cell_1, self.cell_2, evidences
            )
            return mpq(0)
        return functools.reduce(
            lambda a, b: a + b,
            df.weight
        )

    @functools.lru_cache(maxsize=None)
    def get_btypes(self, evidences: FrozenSet[Lit] = None) -> Tuple[FrozenSet[Lit], Poly]:
        btypes = []
        df = self._condition(evidences)
        if len(df) == 0:
            logger.warning(
                'Cell pair (%s, %s) with evidences %s is not satisfiable',
                self.cell_1, self.cell_2, evidences
            )
            return btypes
        for r in df.iterrows():
            btype = set()
            for k, v in r[1].items():
                if k == 'weight':
                    weight = v
                else:
                    if v:
                        btype.add(Lit(k))
                    else:
                        btype.add(Lit(k, False))
            btypes.append((frozenset(btype), weight))
        return btypes

    def satisfiable(self, evidences: FrozenSet[Lit] = None) -> bool:
        evidences = self.evidences.union(evidences)
        return self.wmc.satisfiable(evidences)

    def _get_unknown_atoms(self) -> FrozenSet[Atom]:
        unknown_atoms = []
        for atom in self.wmc.gnd_formula.atoms():
            if Lit(atom) not in self.evidences and \
                    Lit(atom, False) not in self.evidences:
                unknown_atoms.append(atom)
        return unknown_atoms
