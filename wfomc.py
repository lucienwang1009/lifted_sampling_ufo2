import itertools

from pracmln import MLN
from sympy import ntheory
from logzero import logger

from context import Context
from cell_graph import CellGraph


def product_wmc(cell_graph, partition):
    res = 1
    for i, cell_i in enumerate(cell_graph.cells):
        n_i = partition[i]
        res *= (cell_i.inherent_weight ** n_i)
        res *= (cell_i.w ** n_i)
        res *= ((cell_i.s * cell_i.s) ** (n_i * (n_i - 1) / 2))
        for j in range(i + 1, len(cell_graph.cells)):
            n_j = partition[j]
            cell_j = cell_graph.cells[j]
            res *= ((cell_graph.r[cell_i][cell_j] * cell_graph.r[cell_j][cell_i]) ** (n_i * n_j))
    return res


def wfomc(mln):
    assert len(mln.domains) == 1
    context = Context(mln)
    cell_graph = CellGraph(context)
    cell_graph.show()
    domain_size = len(list(mln.domains.values())[0])
    n_cells = len(cell_graph.cells)

    iterator = ntheory.multinomial.multinomial_coefficients_iterator(
        n_cells, domain_size
    )
    res = 0
    for partition, coef in iterator:
        config_wfomc = (coef * product_wmc(cell_graph, partition))
        logger.debug('coefficient and wfomc in the configuration %s: %s, %s',
                     partition, coef, config_wfomc)
        res += config_wfomc
    return res


if __name__ == '__main__':
    mln = MLN.load('./models/friendsmoker.mln', grammar='StandardGrammar')
    wfomc = wfomc(mln)
    print(wfomc)
