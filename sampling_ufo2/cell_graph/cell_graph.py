from __future__ import annotations

import pandas as pd
import functools

from typing import Callable, Dict, FrozenSet, List, Tuple
from logzero import logger
from sympy import Poly

from sampling_ufo2.wfomc.wmc import WMC
from sampling_ufo2.fol.syntax import AndCNF, CNF, Lit, Pred, a, b, c
from sampling_ufo2.fol.utils import ground_FO2
from sampling_ufo2.utils import Rational
from .components import Cell, BtypeTable


class CellGraph(object):
    """
    Cell graph that handles cells (U-types) and the wmc between them.
    """

    def __init__(self, sentence: CNF,
                 get_weight: Callable[[Pred], Tuple[Poly, Poly]]):
        """
        Cell graph that handles cells (U-types) and the WMC between them

        :param sentence CNF: the sentence in the form of CNF
        :param get_weight Callable[[Pred], Tuple[mpq, mpq]]: the weighting function
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

        self.cc_wmc = WMC(self.gnd_formula_cc, self.get_weight)
        self.ab_wmc = WMC(
            self.gnd_formula_ab, self.get_weight
        )
        # build cells
        self.cells: List[Cell] = self._build_cells()
        # filter cells
        logger.info('the number of valid cells: %s',
                    len(self.cells))

        self.cell_weights: Dict[Cell, Poly]
        self.edge_tables: Dict[Tuple[Cell, Cell], BtypeTable]
        self.cell_weights = self._compute_cell_weights()
        self.edge_tables = self._build_edge_tables()

    def show(self):
        logger.info(str(self))

    def __str__(self):
        s = 'CellGraph:\n'
        s += 'predicates: {}\n'.format(self.preds)
        cell_weight_df = []
        edge_weight_df = []
        for idx1, cell1 in enumerate(self.cells):
            cell_weight_df.append(
                [str(cell1), self.get_cell_weight(cell1)]
            )
            edge_weight = []
            for idx2, cell2 in enumerate(self.cells):
                if idx1 < idx2:
                    edge_weight.append(0)
                    continue
                edge_weight.append(
                    self.get_edge_weight(
                        (cell1, cell2))
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

    @functools.lru_cache(maxsize=None, typed=True)
    def get_cell_weight(self, cell: Cell) -> Poly:
        if cell not in self.cell_weights:
            logger.warning(
                "Cell %s not found", cell
            )
            return 0
        return self.cell_weights.get(cell)

    def _check_existence(self, cells: Tuple[Cell, Cell]):
        if cells not in self.edge_tables:
            raise ValueError(
                "Cells (%s) not found, note that the order of cells matters!", cells
            )

    @functools.lru_cache(maxsize=None, typed=True)
    def get_edge_weight(self, cells: Tuple[Cell, Cell],
                        evidences: FrozenSet[Lit] = None) -> Poly:
        self._check_existence(cells)
        return self.edge_tables.get(cells).get_weight(evidences)

    @functools.lru_cache(maxsize=None, typed=True)
    def satisfiable(self, cells: Tuple[Cell, Cell],
                    evidences: FrozenSet[Lit] = None) -> bool:
        self._check_existence(cells)
        return self.edge_tables.get(cells).satisfiable(evidences)

    @functools.lru_cache(maxsize=None)
    def get_btypes(self, cells: Tuple[Cell, Cell],
                   evidences: FrozenSet[Lit] = None) -> Tuple[FrozenSet[Lit], Poly]:
        self._check_existence(cells)
        return self.edge_tables.get(cells).get_btypes(evidences)

    def _build_cells(self):
        cells = []
        code = {}
        for model in self.gnd_formula_cc.models():
            for lit in model:
                code[lit.pred()] = lit.positive
            cells.append(Cell(tuple(code[p] for p in self.preds), self.preds))
        return cells

    def _compute_cell_weights(self):
        weights = dict()
        for cell in self.cells:
            weight = Rational(1, 1)
            for i, pred in zip(cell.code, cell.preds):
                if i:
                    weight = weight * self.get_weight(pred)[0]
                else:
                    weight = weight * self.get_weight(pred)[1]
            weights[cell] = weight
        return weights

    def _build_edge_tables(self):
        tables = dict()
        for i, cell in enumerate(self.cells):
            for j, other_cell in enumerate(self.cells):
                tables[(cell, other_cell)] = BtypeTable(
                    self.ab_wmc, cell, other_cell
                )
        return tables
