from logzero import logger
from pracmln import MLN

from wmc import WMC, WMCSampler
from context import Context
from atom import Atom


class Cell(object):
    def __init__(self, code, context, inherent_weight=None):
        self.code = code
        self.context = context
        self.inherent_weight = inherent_weight

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
        self.wmc_ab = WMC(self.context.gnd_formula_ab, self.context)
        logger.debug('ground a b: %s', self.wmc_ab.gnd_formula)
        self.wmc_cc = WMC(self.context.gnd_formula_cc, self.context)
        logger.debug('ground c c: %s', self.wmc_cc.gnd_formula)
        # build cells
        self.cells = self._build_cells()
        # filter cells
        logger.debug('before filtering, the number of cells: %s', len(self.cells))
        self.cells = list(filter(self._valid_cell, self.cells))
        logger.debug('after filtering, the number of cells: %s', len(self.cells))

        # compute weight for each cell and each pair of cells
        self._compute_cell_weight()
        # [weight dim: [cell1: [cell2: r]]]
        self.edge_weight, self.samplers = self._compute_edge_weight()

    def show(self):
        logger.info('predicates: %s', self.context.preds)
        for cell in self.cells:
            logger.info('weight for cell %s: %s, %s',
                        cell, cell.inherent_weight)
        for k in range(self.context.w_dim):
            # data = pd.DataFrame(self.r[d], self.cells, self.cells)
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     logger.info('%s-th r:\n%s', d, data)
            logger.info('%s-th r: \n%s', k, self.edge_weight[k])

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
        evidences3 = cell.decode('c')
        # evidences1 = self._get_evidences(cell, self.context.left_gnd_atoms)
        # evidences2 = self._get_evidences(cell, self.context.right_gnd_atoms)
        # any index is ok
        weight1 = self.wmc_ab.wmc(evidences1)
        weight2 = self.wmc_ab.wmc(evidences2)
        weight3 = self.wmc_cc.wmc(evidences3)
        if weight1 * weight2 * weight3 == 0:
            logger.debug('filtered cell: %s', cell)
            return False
        return True

    def _compute_cell_weight(self):
        for cell in self.cells:
            weights = []
            for d in range(self.context.w_dim):
                weight = 1
                for i in range(len(self.context.preds)):
                    if (cell.code & (1 << i)):
                        weight *= self.context.get_weight(self.context.preds[i].name, d)[0]
                    else:
                        weight *= self.context.get_weight(self.context.preds[i].name, d)[1]
                weights.append(weight)
            cell.inherent_weight = weights

    # def _compute_s(self):
    #     for cell in self.cells:
    #         cell.s = []
    #         evidences = cell.decode('a')
    #         evidences = evidences.union(cell.decode('b'))
    #         cell.s_sampler = WMCSampler(self.wmc_ab, evidences, self.context)
    #         for d in range(self.context.w_dim):
    #             cell.s.append(self.wmc_ab.wmc(evidences, d))

    def _compute_edge_weight(self):
        w = [{}] * self.context.w_dim
        samplers = {}
        for i, cell in enumerate(self.cells):
            for j, other_cell in enumerate(self.cells):
                if i > j:
                    continue
                evidences = cell.decode('a')
                evidences = evidences.union(other_cell.decode('b'))
                samplers[frozenset((cell, other_cell))] = WMCSampler(
                    self.wmc_ab, evidences, self.context
                )
                for d in range(self.context.w_dim):
                    tmp = self.wmc_ab.wmc(evidences, d)
                    w[d][frozenset((cell, other_cell))] = tmp
        return w, samplers


if __name__ == '__main__':
    mln = MLN.load('./models/friendsmoker-complex.mln', grammar='StandardGrammar')
    g = CellGraph(Context(mln))
    g.show()
