import pandas as pd

from collections import defaultdict
from logzero import logger

from wmc import WMC, WMCSampler
from atom import Atom


class Cell(object):
    def __init__(self, code, context, s=None, w=None, inherent_weight=None):
        self.code = code
        self.context = context
        self.s = s
        self.w = w
        self.inherent_weight = inherent_weight
        self.w_sampler = None
        self.s_sampler = None

    def decode(self, const, include_negative=True):
        evidences = set()
        for i, p in enumerate(self.context.preds):
            var = Atom(predname=p.name, args=([const] * len(p.argdoms))).to_var()
            if (self.code & (1 << i)):
                evidences.add(var)
            elif include_negative:
                evidences.add(var.negate())
        return evidences

    def __eq__(self, other):
        return self.code == other.code

    def __hash__(self):
        return hash(self.code)

    def __str__(self):
        return format(self.code, 'b').zfill(len(self.context.preds))[::-1]

    def __repr__(self):
        return self.__str__()


class CellGraph(object):
    def __init__(self, context):
        self.context = context
        logger.debug('prednames: %s', self.context.preds)
        # ground the sentence to (a,b) and (c,c)
        self.wmc_ab = WMC(self.context.gnd_formula_ab, self.context.get_weight)
        self.wmc_cc = WMC(self.context.gnd_formula_cc, self.context.get_weight)
        # build cells
        self.cells = self._build_cells()
        # compute inherent weight for each cell
        self._compute_inherent_weight()
        # compute w first for filtering the cell
        self._compute_w()
        # filter cells
        logger.debug('before filtering, the number of cells: %s', len(self.cells))
        self.cells = list(filter(self._valid_cell, self.cells))
        logger.debug('after filtering, the number of cells: %s', len(self.cells))
        self._compute_s()
        self.r, self.r_samplers = self._compute_r()

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
            cells.append(Cell(i, self.context))
        return cells

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
        evidences1 = cell.decode('a')
        evidences2 = cell.decode('b')
        # evidences1 = self._get_evidences(cell, self.context.left_gnd_atoms)
        # evidences2 = self._get_evidences(cell, self.context.right_gnd_atoms)
        weight1 = self.wmc_ab.wmc(evidences1)
        weight2 = self.wmc_ab.wmc(evidences2)
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
        samplers = defaultdict(dict)
        for cell in self.cells:
            for other_cell in self.cells:
                if cell == other_cell:
                    continue
                evidences = cell.decode('a')
                evidences = evidences.union(other_cell.decode('b'))
                # evidences = self._get_evidences(cell, self.context.left_gnd_atoms)
                # evidences = evidences.union(
                #     self._get_evidences(other_cell, self.context.right_gnd_atoms)
                # )
                r[cell][other_cell] = self.wmc_ab.wmc(evidences)
                samplers[cell][other_cell] = WMCSampler(self.wmc_ab, evidences)
        return r, samplers

    def _compute_s(self):
        for cell in self.cells:
            evidences = cell.decode('a')
            evidences = evidences.union(cell.decode('b'))
            # evidences = self._get_evidences(cell, self.context.left_gnd_atoms)
            # evidences = evidences.union(
            #     self._get_evidences(cell, self.context.right_gnd_atoms)
            # )
            cell.s = self.wmc_ab.wmc(evidences)
            cell.s_sampler = WMCSampler(self.wmc_ab, evidences)

    def _compute_w(self):
        for cell in self.cells:
            evidences = cell.decode('c')
            # evidences = self._get_evidences(cell, self.context.same_gnd_atoms)
            cell.w = self.wmc_cc.wmc(evidences)
            cell.w_sampler = WMCSampler(self.wmc_cc, evidences)
