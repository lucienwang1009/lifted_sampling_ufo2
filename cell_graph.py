import math
import pandas as pd

from copy import deepcopy
from collections import defaultdict
from pracmln import MLN
from logzero import logger
from nnf import amc, Var

from atom import Atom
from utils import to_nnf


class Cell(object):
    def __init__(self, code, n_preds, s=None, w=None, inherent_weight=None):
        self.code = code
        self.n_preds = n_preds
        self.s = s
        self.w = w
        self.inherent_weight = inherent_weight

    def __eq__(self, other):
        return self.code == other.code

    def __hash__(self):
        return hash(self.code)

    def __str__(self):
        return format(self.code, 'b').zfill(self.n_preds)[::-1]

    def __repr__(self):
        return self.__str__()


class CellGraph(object):
    def __init__(self, context):
        self.context = context
        logger.debug('prednames: %s', self.context.preds)
        # ground the sentence to (a,b) and (c,c)
        gnd_formula1 = self.context.ground_formula('a', 'b')
        gnd_formula2 = self.context.ground_formula('c', 'c')
        # nnf for two different cases
        self.nnf1 = to_nnf(gnd_formula1)
        self.nnf2 = to_nnf(gnd_formula2)
        self.var_weights1 = self._get_var_weights(self.nnf1)
        self.var_weights2 = self._get_var_weights(self.nnf2)
        # build cells
        self.cells = self._build_cells()
        # evidence caches
        self.same_assign_atoms = [
            Atom(predname=p.name, args=(['c'] * len(p.argdoms)))
            for p in self.context.preds
        ]
        # logger.debug('same assignment evidences: %s', self.same_assign_evidences)
        self.left_assign_atoms = [
            Atom(predname=p.name, args=(['a'] * len(p.argdoms)))
            for p in self.context.preds
        ]
        # logger.debug('left assignment evidences: %s', self.left_assign_evidences)
        self.right_assign_atoms = [
            Atom(predname=p.name, args=(['b'] * len(p.argdoms)))
            for p in self.context.preds
        ]
        # compute inherent weight for each cell
        self._compute_inherent_weight()
        # compute w first for filtering the cell
        self._compute_w()
        # filter cells
        logger.debug('before filtering, the number of cells: %s', len(self.cells))
        self.cells = list(filter(self._valid_cell, self.cells))
        logger.debug('after filtering, the number of cells: %s', len(self.cells))
        self._compute_s()
        self.r = self._compute_r()

    def show(self):
        logger.info('predicates: %s', self.context.preds)
        for cell in self.cells:
            logger.info('inherent weight, s and w for cell %s: %s, %s, %s',
                        cell, cell.inherent_weight, cell.s, cell.w)
        data = pd.DataFrame(self.r, self.cells, self.cells)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info('r:\n%s', data)

    def _build_cells(self):
        cells = []
        n_preds = len(self.context.preds)
        for i in range(2 ** n_preds):
            cells.append(Cell(i, len(self.context.preds)))
        return cells

    def _get_var_weights(self, nnf):
        weights = {}
        for varname in nnf.vars():
            var = Var(varname)
            w = self.context.get_weight(Atom.from_var(var).predname)
            weights[var] = w[0]
            weights[var.negate()] = w[1]
        return weights

    def _wmc(self, nnf, var_weights, evidences=None):
        def weights_fn(var):
            if var in evidences:
                return 1
            elif var.negate() in evidences:
                return 0
            return var_weights[var]
        return amc.WMC(nnf, weights_fn)

    def _get_evidences(self, cell, atoms):
        evidences = set()
        for i in range(len(self.context.preds)):
            if (cell.code & (1 << i)):
                evidences.add(atoms[i].to_var())
            else:
                evidences.add(atoms[i].to_var().negate())
        return evidences

    def _valid_cell(self, cell):
        '''
        Any cell with zero w and zero wmc with all assignments for other variables
        should be removed
        '''
        evidences1 = self._get_evidences(cell, self.left_assign_atoms)
        evidences2 = self._get_evidences(cell, self.right_assign_atoms)
        weight1 = self._wmc(self.nnf1, self.var_weights1, evidences1)
        weight2 = self._wmc(self.nnf1, self.var_weights1, evidences2)
        if weight1 * weight2 * cell.w == 0:
            logger.debug('filtered cell: %s', cell)
            return False
        return True

    def _compute_inherent_weight(self):
        for cell in self.cells:
            weight = 1
            for i in range(len(self.context.preds)):
                if (cell.code & (1 << i)):
                    weight *= self.context.get_weight(self.context.preds[i].name)[0]
                else:
                    weight *= self.context.get_weight(self.context.preds[i].name)[1]
            cell.inherent_weight = weight

    def _compute_r(self):
        r = defaultdict(dict)
        for cell in self.cells:
            for other_cell in self.cells:
                if cell == other_cell:
                    continue
                evidences = self._get_evidences(cell, self.left_assign_atoms)
                evidences = evidences.union(
                    self._get_evidences(other_cell, self.right_assign_atoms)
                )
                r[cell][other_cell] = self._wmc(self.nnf1, self.var_weights1, evidences)
        return r

    def _compute_s(self):
        for cell in self.cells:
            evidences = self._get_evidences(cell, self.left_assign_atoms)
            evidences = evidences.union(
                self._get_evidences(cell, self.right_assign_atoms)
            )
            cell.s = self._wmc(self.nnf1, self.var_weights1, evidences)

    def _compute_w(self):
        for cell in self.cells:
            evidences = self._get_evidences(cell, self.same_assign_atoms)
            cell.w = self._wmc(self.nnf2, self.var_weights2, evidences)
