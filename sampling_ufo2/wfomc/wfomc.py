import os
import argparse
import logging
import numpy as np
import logzero

from logzero import logger
from typing import Tuple, List
from collections import namedtuple
from itertools import product

from sampling_ufo2.utils import MultinomialCoefficients, tree_sum, ArborescenceSumContext
from sampling_ufo2.cell_graph import CellGraph, Cell
from sampling_ufo2.context import Context
from sampling_ufo2.parser import parse_mln_constraint
from sampling_ufo2.network.constraint import ArborescenceConstraint


ConfigWeight = namedtuple(
    'ConfigWeight', ['weight', 'A', 'force_edges']
)


def assign_cell(cell_graph: CellGraph, partition: Tuple[int]) -> Tuple[List[Cell], np.ndarray]:
    cell_assignment = list()
    w = 1
    for i, n in enumerate(partition):
        for j in range(n):
            cell_assignment.append(cell_graph.cells[i])
            w *= cell_graph.get_cell_weight(
                cell_assignment[-1])
    return cell_assignment, w


def get_config_weight(context: Context, cell_graph: CellGraph, partition: Tuple[int]) -> np.ndarray:
    # exact one root element for arborescence
    if context.contain_arborescence_constraint():
        n_roots = 0
        for idx, cell in enumerate(cell_graph.cells):
            if cell.is_positive(context.tree_constraint.root_pred):
                n_roots += partition[idx]
        if n_roots != 1:
            return ConfigWeight(
                np.array(0.0, dtype=context.dtype), None, None
            )

    domain_size = len(context.domain)
    if context.contain_tree_constraint():
        cell_assignment, w = assign_cell(cell_graph, partition)
        # assign each element a cell type
        A = np.zeros([domain_size, domain_size, context.weight_dims],
                     dtype=context.dtype)
        f = []
        r = 1
        for i in range(domain_size):
            for j in range(domain_size):
                if i >= j:
                    continue
                edge_weight = cell_graph.get_edge_weight(frozenset((
                    cell_assignment[i], cell_assignment[j]
                )))
                if np.all(edge_weight[1] == 0):
                    # WMC(\phi \land \not R(a,b)) = 0
                    f.append((i, j))
                    A[i, j, :] = 1.0
                    r *= edge_weight[0]
                else:
                    # WMC(\phi \land \not R(a,b)) != 0
                    A[i, j, :] = edge_weight[0] / edge_weight[1]
                    r *= edge_weight[1]
        ts = tree_sum(A, f)
        logger.debug('tree sum:%s, inherent_weight: %s, r:%s',
                     ts, w, r)
        res = ts * w * r
        return ConfigWeight(
            res, A, f
        )
    elif context.contain_arborescence_constraint():
        cell_assignment, w = assign_cell(cell_graph, partition)
        tree_constraint: ArborescenceConstraint = context.tree_constraint
        root = None
        for i in range(domain_size):
            if cell_assignment[i].is_positive(tree_constraint.root_pred):
                root = i
        # assign each element a cell type
        A = np.zeros([domain_size, domain_size, context.weight_dims],
                     dtype=context.dtype)
        f = []
        F = []
        r = 1
        for i in range(domain_size):
            for j in range(domain_size):
                if i >= j:
                    continue
                # WMC(f(a,b) ^ R(a,b) ^ !R(b,a)), WMC(f(a,b) ^ !R(a,b) ^ !R(b,a)), WMC(f(a,b) ^ !R(a,b) ^ R(b,a))
                ab_p, neg, ba_p, ab_or = cell_graph.get_edge_weight(frozenset((
                    cell_assignment[i], cell_assignment[j]
                )))
                if np.any(neg == 0):
                    if np.any(ab_or == 0):
                        continue
                    if np.any(ba_p == 0):
                        f.append((i, j))
                    elif np.any(ab_p == 0):
                        f.append((j, i))
                    else:
                        F.append([(i, j), (j, i)])
                    # r *= ab_or
                    A[i, j, :] = ab_p  # / ab_or
                    A[j, i, :] = ba_p  # / ab_or
                else:
                    r *= neg
                    A[i, j, :] = ab_p / neg
                    A[j, i, :] = ba_p / neg
        logger.info('Undirected contraction edges: %s', F)
        ts = 0
        if F:
            for i in product(*F):
                tmp_f = f + list(i)
                ts += ArborescenceSumContext(A, root, tmp_f).tree_sum()
        else:
            ts = ArborescenceSumContext(A, root, f).tree_sum()
        res = ts * w * r
        logger.debug('partition: %s, tree sum:%s, inherent_weight: %s, r:%s, res: %s',
                     partition, ts, w, r, res)
        return ConfigWeight(
            res, A, f
        )
    else:
        res = 1
        for i, cell_i in enumerate(cell_graph.cells):
            n_i = partition[i]
            if n_i == 0:
                continue

            res *= np.power(cell_graph.get_cell_weight(cell_i), n_i)
            res *= np.power(cell_graph.get_edge_weight(frozenset((cell_i, cell_i))),
                            n_i * (n_i - 1) / 2)
            for j in range(i + 1, len(cell_graph.cells)):
                n_j = partition[j]
                if n_j == 0:
                    continue
                cell_j = cell_graph.cells[j]
                res *= np.power(cell_graph.get_edge_weight(frozenset((cell_i, cell_j))),
                                n_i * n_j)
        return ConfigWeight(res, None, None)


def wfomc(mln, tree_constraint, cardinality_constraint):
    if len(mln.vars()) > 2:
        raise RuntimeError(
            "Support at most two variables, i.e., FO2"
        )

    context = Context(mln, tree_constraint, cardinality_constraint)
    cell_graph = CellGraph(context)
    cell_graph.show()

    domain_size = len(context.domain)
    n_cells = len(cell_graph.cells)
    multinomial_coefficients = MultinomialCoefficients(domain_size, n_cells)

    res = 0
    for partition in multinomial_coefficients:
        coef = multinomial_coefficients.coef(partition)
        wfomc_partition = get_config_weight(
            context, cell_graph, partition).weight

        if context.contain_cardinality_constraint():
            res = res + coef * \
                np.sum(np.dot(context.top_weights, wfomc_partition))
        else:
            res = res + coef * np.sum(wfomc_partition)

    if context.contain_cardinality_constraint():
        res = np.divide(res, context.reverse_dft_coef)
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
    mln, tree_constraint, cardinality_constraint = parse_mln_constraint(
        args.input)
    res = wfomc(mln, tree_constraint, cardinality_constraint)
    logger.info('WFOMC: %s', res)
