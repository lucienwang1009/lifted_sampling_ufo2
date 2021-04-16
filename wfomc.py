import os
import argparse
import logging

import logzero
from pracmln import MLN
from sympy import ntheory
from logzero import logger

from context import Context
from cell_graph import CellGraph


def product_wmc(cell_graph, partition, index):
    res = 1
    for i, cell_i in enumerate(cell_graph.cells):
        n_i = partition[i]
        res *= (cell_i.w[index] ** n_i)
        res *= (cell_i.s[index] ** (n_i * (n_i - 1) / 2))
        for j in range(i + 1, len(cell_graph.cells)):
            n_j = partition[j]
            cell_j = cell_graph.cells[j]
            res *= (cell_graph.r[index][cell_i][cell_j] ** (n_i * n_j))
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
    for d in range(context.w_dim):
        for partition, coef in iterator:
            config_wfomc = (coef * product_wmc(cell_graph, partition, d))
            res += config_wfomc
    return res


def parse_args():
    parser = argparse.ArgumentParser(
        description='WFOMC for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    mln = MLN.load(args.input, grammar='StandardGrammar')
    res = wfomc(mln)
    logger.info('WFOMC: %s', res)
