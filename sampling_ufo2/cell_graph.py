import numpy as np

from typing import List, Tuple, Set, Dict, FrozenSet, Union
from dataclasses import dataclass
from logzero import logger
from itertools import product

from sampling_ufo2.context import Context
from sampling_ufo2.wfomc.wmc import WMC, WMCSampler
from sampling_ufo2.fol.syntax import Lit, Pred, Term, x, a, b, c


@dataclass(frozen=True)
class Cell(object):
    """
    Unary types
    """
    code: List[bool]
    preds: Tuple[Pred]

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

    def __str__(self):
        evidences: Set[Lit] = self.get_evidences(x)
        return ' ^ '.join(str(lit) for lit in evidences)

    def __repr__(self):
        return self.__str__()


class CellGraph(object):
    def __init__(self, context: Context):
        self.context = context
        logger.debug('prednames: %s', self.context.preds)
        # ground the sentence to (a,b) and (c,c)
        logger.debug('ground a b: %s', self.context.gnd_formula_ab)
        self.get_weight = self.context.get_weight_fn()
        self.wmc_ab = WMC(self.context.gnd_formula_ab, self.get_weight)
        logger.debug('ground c c: %s', self.context.gnd_formula_cc)
        self.wmc_cc = WMC(self.context.gnd_formula_cc, self.get_weight)

        if self.context.contain_tree_constraint():
            logger.debug('Ground positive a b: %s',
                         self.context.gnd_formula_ab_p)
            logger.debug('Ground negative a b: %s',
                         self.context.gnd_formula_ab_n)
            self.wmc_ab_p = WMC(self.context.gnd_formula_ab_p,
                                self.get_weight)
            self.wmc_ab_n = WMC(self.context.gnd_formula_ab_n,
                                self.get_weight)
        # build cells
        self.cells = self._build_cells()
        # filter cells
        logger.info('before filtering, the number of cells: %s',
                    len(self.cells))
        self.cells = list(filter(self._valid_cell, self.cells))
        logger.info('after filtering, the number of cells: %s',
                    len(self.cells))

        self.cell_weights: Dict[Cell, np.ndarray]
        self.edge_weights: Dict[FrozenSet[Cell],
                                Union[np.ndarray, Tuple[np.ndarray]]]
        self.samplers: Dict[FrozenSet[Cell],
                            Union[WMCSampler, Tuple[WMCSampler]]]
        self.cell_weights = self._compute_cell_weights()
        self.edge_weights, self.samplers = self._compute_edge_weight()

    def show(self):
        logger.info('predicates: %s', self.context.preds)
        for cell in self.cells:
            logger.info('weight for cell %s: %s',
                        cell, self.cell_weights.get(cell))
            # data = pd.DataFrame(self.r[d], self.cells, self.cells)
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     logger.info('%s-th r:\n%s', d, data)
        logger.info('r: \n%s', self.edge_weights)

    def get_cell_weight(self, cell: Cell) -> np.ndarray:
        if cell not in self.cell_weights:
            raise RuntimeError(
                "Cell %s not found", cell
            )
        return self.cell_weights.get(cell)

    def get_edge_weight(self, cells: FrozenSet[Cell]) -> Union[np.ndarray, Tuple[np.ndarray]]:
        if cells not in self.edge_weights:
            raise RuntimeError(
                "Cells (%s) not found", cells
            )
        return self.edge_weights.get(cells)

    def _build_cells(self):
        cells = []
        n_preds = len(self.context.preds)
        for i in product(*([[True, False]] * n_preds)):
            cells.append(Cell(i, self.context.preds))
        return cells

    def _valid_cell(self, cell: Cell):
        '''
        Any cell with zero w and zero wmc with all assignments for other variables
        should be removed
        '''
        evidences1 = cell.get_evidences(a)
        evidences2 = cell.get_evidences(b)
        evidences3 = cell.get_evidences(c)
        weight1 = self.wmc_ab.wmc(evidences1)
        weight2 = self.wmc_ab.wmc(evidences2)
        weight3 = self.wmc_cc.wmc(evidences3)
        if any(np.all(weight == 0) for weight in [
            weight1, weight2, weight3
        ]):
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
            weights[cell] = weight
        return weights

    def _compute_edge_weight(self):
        weights = dict()
        samplers = dict()
        for i, cell in enumerate(self.cells):
            # logger.debug('evidence: %s', cell.get_evidences(a))
            for j, other_cell in enumerate(self.cells):
                if i > j:
                    continue
                # logger.debug('other evidence: %s', other_cell.get_evidences(b))
                evidences = cell.get_evidences(a)
                evidences = evidences.union(other_cell.get_evidences(b))
                if self.context.contain_tree_constraint():
                    edge_samplers = (
                        WMCSampler(self.wmc_ab_p, evidences),
                        WMCSampler(self.wmc_ab_n, evidences)
                    )
                    edge_weights = (
                        self.wmc_ab_p.wmc(evidences),
                        self.wmc_ab_n.wmc(evidences)
                    )
                    samplers[frozenset((cell, other_cell))] = edge_samplers
                    weights[frozenset((cell, other_cell))] = edge_weights
                else:
                    samplers[frozenset((cell, other_cell))] = WMCSampler(
                        self.wmc_ab, evidences
                    )
                    weights[frozenset((cell, other_cell))
                            ] = self.wmc_ab.wmc(evidences)
        return weights, samplers
