from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Callable, Dict, FrozenSet, List, Set, Tuple
from dataclasses import dataclass, field
from logzero import logger
from itertools import product

from sampling_ufo2.wfomc.wmc import WMC, WMCSampler
from sampling_ufo2.fol.syntax import Lit, Pred, Term, CNF, x, a, b, c, tautology, AndCNF
from sampling_ufo2.fol.utils import ground_FO2
from sampling_ufo2.utils import format_np_complex


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


class CellGraph(object):
    """
    Cell graph that handles cells (U-types) and the wmc between them.
    """

    def __init__(self, sentence: CNF, get_weight: Callable[[Pred], Tuple[np.ndarray]], conditional_formulas: List[CNF] = None):
        """
        Cell graph that handles cells (U-types) and the WMC between them

        :param sentence CNF: the sentence in the form of CNF
        :param get_weight Callable[[Pred], Tuple[np.ndarray]]: the weighting function
        :param conditional_formulas List[CNF]: the optional conditional formula appended in WMC computing
        """
        self.sentence: CNF = sentence
        self.get_weight = get_weight
        self.preds: Tuple[Pred] = tuple(self.sentence.preds())
        logger.debug('prednames: %s', self.preds)

        gnd_formula_ab1 = ground_FO2(self.sentence, a, b)
        gnd_formula_ab2 = ground_FO2(self.sentence, b, a)
        self.gnd_formula_ab: CNF = AndCNF(
            gnd_formula_ab1, gnd_formula_ab2)
        self.gnd_formula_cc: CNF = ground_FO2(self.sentence, c)
        logger.debug('ground a b: %s', self.gnd_formula_ab)

        self.ab_wmcs = {
            tautology: WMC(self.gnd_formula_ab, self.get_weight)
        }
        self.cc_wmc = WMC(self.gnd_formula_cc, self.get_weight)
        # build cells
        self.cells: List[Cell] = self._build_cells()
        # filter cells
        logger.info('before filtering, the number of cells: %s',
                    len(self.cells))
        self.cells = list(filter(self._valid_cell, self.cells))
        logger.info('after filtering, the number of cells: %s',
                    len(self.cells))

        if conditional_formulas:
            for conditional_formula in conditional_formulas:
                self.ab_wmcs[conditional_formula] = WMC(
                    AndCNF(
                        self.gnd_formula_ab, conditional_formula
                    ),
                    self.get_weight
                )

        self.cell_weights: Dict[Cell, np.ndarray]
        self.edge_weights: Dict[FrozenSet[Cell],
                                Dict[CNF, np.ndarray]]
        self.samplers: Dict[FrozenSet[Cell],
                            Dict[CNF, WMCSampler]]
        self.cell_weights = self._compute_cell_weights()
        self.edge_weights, self.samplers = self._compute_edge_weight()

    def show(self):
        logger.info(str(self))

    def __str__(self):
        s = 'CellGraph:\n'
        s += 'predicates: {}\n'.format(self.preds)
        cell_weight_df = []
        edge_weight_df = []
        for idx1, cell1 in enumerate(self.cells):
            # NOTE just for debug, thus summation is ok
            cell_weight_df.append(
                [str(cell1), format_np_complex(
                    np.sum(self.get_cell_weight(cell1)))]
            )
            edge_weight = []
            for idx2, cell2 in enumerate(self.cells):
                if idx1 < idx2:
                    edge_weight.append(0)
                    continue
                edge_weight.append(
                    format_np_complex(np.sum(self.get_edge_weight(
                        frozenset((cell1, cell2))
                    )))
                )
            edge_weight_df.append(edge_weight)
        cell_str = [str(cell) for cell in self.cells]
        cell_weight_df = pd.DataFrame(cell_weight_df, index=None,
                                      columns=['Cell', 'Weight'])
        edge_weight_df = pd.DataFrame(edge_weight_df, index=cell_str,
                                      columns=cell_str)
        s += 'cell weights: \n'
        s += cell_weight_df.to_markdown() + '\n'
        s += 'edge weights: \n'
        s += edge_weight_df.to_markdown()
        return s

    def __repr__(self):
        return str(self)

    def get_cells(self, cell_filter: Callable[[Cell], bool] = None):
        if cell_filter is None:
            return self.cells
        return list(filter(cell_filter, self.cells))

    def get_cell_weight(self, cell: Cell) -> np.ndarray:
        if cell not in self.cell_weights:
            logger.warning(
                "Cell %s not found", cell
            )
            return 0
        return self.cell_weights.get(cell)

    def get_edge_weight(self, cells: FrozenSet[Cell], conditional_formula: CNF = None) -> np.ndarray:
        if cells not in self.edge_weights:
            raise RuntimeError(
                "Cells (%s) not found", cells
            )
            return 0
        if conditional_formula is None:
            conditional_formula = tautology
        return self.edge_weights.get(cells).get(conditional_formula)

    def _build_cells(self):
        cells = []
        n_preds = len(self.preds)
        for i in product(*([[True, False]] * n_preds)):
            cells.append(Cell(i, self.preds))
        return cells

    def _valid_cell(self, cell: Cell):
        '''
        Any cell with zero w and zero wmc with all assignments for other variables
        should be removed
        '''
        evidences = cell.get_evidences(c)
        weight = self.cc_wmc.wmc(evidences)
        if np.all(weight == 0):
            return False
        return True

    def _compute_cell_weights(self):
        weights = dict()
        for cell in self.cells:
            weight = 1.0
            for i, pred in zip(cell.code, cell.preds):
                if i:
                    weight = np.multiply(
                        weight, self.get_weight(pred)[0])
                else:
                    weight = np.multiply(
                        weight, self.get_weight(pred)[1])
            # NOTE: Avoid modifying the weight
            weight.flags.writeable = False
            weights[cell] = weight
        return weights

    def _compute_edge_weight(self):
        weights = dict()
        samplers = dict()
        for i, cell in enumerate(self.cells):
            for j, other_cell in enumerate(self.cells):
                if i > j:
                    continue
                edge_weights = dict()
                edge_samplers = dict()
                evidences = cell.get_evidences(a)
                evidences = evidences.union(other_cell.get_evidences(b))
                for conditional_formula, wmc in self.ab_wmcs.items():
                    wmc_val = wmc.wmc(evidences)
                    # NOTE: Avoid modifying the weight
                    wmc_val.flags.writeable = False
                    edge_weights[conditional_formula] = wmc_val
                    edge_samplers[conditional_formula] = WMCSampler(
                        wmc, evidences)
                samplers[frozenset((cell, other_cell))] = edge_samplers
                weights[frozenset((cell, other_cell))] = edge_weights
        return weights, samplers
