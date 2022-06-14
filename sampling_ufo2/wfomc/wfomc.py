import os
import argparse
import logging
import numpy as np
import logzero

from logzero import logger
from typing import Tuple, List, Dict
from collections import namedtuple, defaultdict
from itertools import product
from copy import deepcopy

from sampling_ufo2.fol.syntax import CNF
from sampling_ufo2.utils import MultinomialCoefficients, multinomial, tree_sum
from sampling_ufo2.cell_graph import CellGraph, Cell
from sampling_ufo2.context import Context
from sampling_ufo2.parser import parse_mln_constraint
from sampling_ufo2.network import MLN, TreeConstraint, CardinalityConstraint


ConfigResult = namedtuple(
    'ConfigResult', ['weight', 'A', 'force_edges']
)


class WFOMC(object):
    def __init__(self, mln: MLN, tree_constraint: TreeConstraint,
                 cardinality_constraint: CardinalityConstraint):
        super().__init__()
        if len(mln.vars()) > 2:
            raise RuntimeError(
                "Support at most two variables, i.e., FO2"
            )
        self.mln: MLN = mln
        self.tree_constraint: TreeConstraint = tree_constraint
        self.cardinality_constraint: CardinalityConstraint = cardinality_constraint

        self.context: Context = Context(
            mln, tree_constraint, cardinality_constraint)
        self.domain_size: int = len(self.context.domain)

        if logzero.loglevel == logging.DEBUG:
            self.context.cell_graph.show()
        # pre compute the pascal triangle for computing multinomial coefficient
        MultinomialCoefficients.precompute(self.domain_size)
        if self.context.contain_existential_quantifier():
            # { (cell, n, k): wfomc }
            self.existential_weights = self._precompute_ext_weight(
                self.domain_size - 1)
            logger.debug('Pre-computed weights for existential quantified formula: %s',
                         self.existential_weights)

    def _assign_cell(self, cell_graph: CellGraph,
                     partition: Dict[Cell, int]) -> Tuple[List[Cell], np.ndarray]:
        cell_assignment = list()
        w = 1
        for cell, n in partition.items():
            for j in range(n):
                cell_assignment.append(cell)
                w = w * cell_graph.get_cell_weight(cell)
        return cell_assignment, w

    def _get_config_weight_standard(self, cell_graph: CellGraph, partition: Dict[Cell, int]) -> np.ndarray:
        res = 1
        for i, cell_i in enumerate(partition):
            n_i = partition[cell_i]
            if n_i == 0:
                continue
            res = res * np.power(
                cell_graph.get_cell_weight(cell_i), n_i
            )
            res = res * np.power(cell_graph.get_edge_weight(
                frozenset((cell_i, cell_i))
            ), n_i * (n_i - 1) / 2)
            for j, cell_j in enumerate(partition):
                if j <= i:
                    continue
                n_j = partition[cell_j]
                if n_j == 0:
                    continue
                res = res * np.power(cell_graph.get_edge_weight(
                    frozenset((cell_i, cell_j))
                ), n_i * n_j)
        return res

    def _precompute_ext_weight(self, domain_size):
        # { (cell, n, k): wfomc }
        existential_weights = defaultdict(lambda: 0)
        cell_graph = self.context.skolem_cell_graph
        cells = cell_graph.get_cells()
        if logzero.loglevel == logging.DEBUG:
            cell_graph.show()
        # NOTE: only one existential quantifier
        skolem_pred = self.context.skolem_preds[0]
        for partition in multinomial(len(cells), domain_size):
            res = self._get_config_weight_standard(
                cell_graph, dict(zip(cells, partition)))
            # if self.context.contain_cardinality_constraint():
            #     res = np.sum(np.dot(self.context.top_weights, res))
            ext_n_k = defaultdict(lambda: [0, 0])
            for cell, n in zip(cells, partition):
                cell_dropped_skolem = cell.drop_pred(skolem_pred)
                if cell.is_positive(skolem_pred):
                    ext_n_k[cell_dropped_skolem][0] += n
                else:
                    ext_n_k[cell_dropped_skolem][0] += n
                    ext_n_k[cell_dropped_skolem][1] += n
            # logger.debug('%s: %s', ext_n_k, res)
            for ks in product(
                *list(
                    range(k_min, n + 1) for n, k_min in ext_n_k.values()
                )
            ):
                coef = 1
                for k, (n, k_min) in zip(ks, ext_n_k.values()):
                    coef *= MultinomialCoefficients.coef(
                        (k_min, k - k_min)
                    )
                existential_weights[
                    frozenset(
                        zip(ext_n_k.keys(), [n for n, k in ext_n_k.values()], ks))
                ] += coef * res
        return existential_weights

    def get_config_result_existential(self, partition: Dict[Cell, int]) -> ConfigResult:
        ext_pred = self.context.ext_preds[0]
        cell_graph = self.context.cell_graph
        cells, config = list(zip(*partition.items()))
        config = list(config)
        tseitin_pred = self.context.ext_preds[0]
        already_satisfied_ext = [cell.is_positive(
            tseitin_pred) for cell in cells]
        k = [n if not already_satisfied_ext[idx]
             else 0 for idx, n in enumerate(config)]
        # all elements are satisfied the existential quantified formula
        if all(already_satisfied_ext[i] for i, n in enumerate(config) if n != 0):
            # the weight is exactly that of sentence without existential quantifiers
            logger.info('All cells satisfies existential quantifier')
            res = self._get_config_weight_standard(cell_graph, partition)
            return ConfigResult(res, None, None)
        # choose any element
        for i, satisfied in enumerate(already_satisfied_ext):
            if config[i] > 0:  # and not satisfied:
                selected_idx = i
                break
        selected_cell = cells[selected_idx]
        logger.debug('select cell: %s', selected_cell)
        # domain recursion
        config[selected_idx] -= 1
        k[selected_idx] -= 1
        edge_weights: Dict[CNF, Dict[CNF, np.ndarray]] = dict()
        # filter the impossible existential B-types
        # NOTE: here the order of cells is required
        for cell in cells:
            weights = dict()
            for ext_btype in self.context.ext_btypes:
                weight = cell_graph.get_edge_weight(
                    frozenset((selected_cell, cell)),
                    ext_btype.get_conditional_formula()
                )
                if np.all(weight != 0):
                    weights[ext_btype] = weight
            edge_weights[cell] = weights
        res = 0
        logger.debug('possible edges: %s', edge_weights)
        logger.debug('cell configuration: %s, k: %s', config, k)
        for reduce_configs in product(
                *list(multinomial(
                    len(edge_weights[cell]), total_num
                ) for cell, total_num in zip(cells, config))
        ):
            logger.debug('reduced config: %s', reduce_configs)
            reduced_k = deepcopy(k)
            W_e = cell_graph.get_cell_weight(selected_cell)
            satisfied = False
            for idx, (reduce_config, edge_weight) in enumerate(
                    zip(reduce_configs, edge_weights.values())
            ):
                W_e = W_e * MultinomialCoefficients.coef(reduce_config)
                for num, ext_btype in zip(reduce_config, edge_weight.keys()):
                    ab_p, ba_p = ext_btype.is_positive(ext_pred)
                    if num > 0 and ab_p:
                        satisfied = True
                    W_e = W_e * np.power(edge_weight[ext_btype], num)
                    if ba_p:
                        reduced_k[idx] = max(reduced_k[idx] - num, 0)
            if satisfied or already_satisfied_ext[selected_idx]:
                logger.debug('W_e: %s', W_e)
                logger.debug('reduced wfomc for cells=%s, n=%s, k=%s: ',
                             cells, config, reduced_k)
                reduced_wfomc = self.existential_weights[frozenset(
                    zip(cells, config, reduced_k))]
                logger.debug('%s', reduced_wfomc)
                res += (W_e * reduced_wfomc)
        return ConfigResult(res, None, None)

    def get_config_result_tree(self, partition: Dict[Cell, int]) -> ConfigResult:
        cell_assignment, w = self._assign_cell(
            self.context.cell_graph, partition)
        # assign each element a cell type
        A = np.zeros([self.domain_size, self.domain_size, self.context.weight_dims],
                     dtype=self.context.dtype)
        f = []
        r = 1
        for i in range(self.domain_size):
            for j in range(self.domain_size):
                if i >= j:
                    continue
                edge = frozenset((
                    cell_assignment[i], cell_assignment[j],
                ))
                ab_p = self.context.cell_graph.get_edge_weight(
                    edge, self.context.tree_condition_ab_p)
                ab_n = self.context.cell_graph.get_edge_weight(
                    edge, self.context.tree_condition_ab_n)
                if np.all(ab_n == 0):
                    # WMC(\phi \land \not R(a,b)) = 0
                    f.append((i, j))
                    A[i, j, :] = 1.0
                    r = r * ab_p
                else:
                    # WMC(\phi \land \not R(a,b)) != 0
                    A[i, j, :] = ab_p / ab_n
                    r = r * ab_n
        ts = tree_sum(A, f)
        logger.debug('tree sum:%s, inherent_weight: %s, r:%s',
                     ts, w, r)
        res = ts * w * r
        return ConfigResult(
            res, A, f
        )

    def get_config_result(self, partition: Dict[Cell, int]) -> ConfigResult:
        if self.context.contain_existential_quantifier():
            return self.get_config_result_existential(partition)
        elif self.context.contain_tree_constraint():
            return self.get_config_result_tree(partition)
        else:
            res = self._get_config_weight_standard(
                self.context.cell_graph, partition)
            return ConfigResult(res, None, None)

    def compute(self):
        cells = self.context.cell_graph.get_cells()
        n_cells = len(cells)

        res = 0
        for partition in multinomial(n_cells, self.domain_size):
            coef = MultinomialCoefficients.coef(partition)
            partition_dict = dict(zip(cells, partition))
            logger.debug(
                '=' * 15 + ' Compute WFOMC for the partition %s ' + '=' * 15,
                partition_dict
            )
            wfomc_partition = self.get_config_result(partition_dict).weight

            if self.context.contain_cardinality_constraint():
                wfomc_partition = coef * \
                    np.sum(np.dot(self.context.top_weights, wfomc_partition))
                res = res + wfomc_partition
            else:
                wfomc_partition = coef * np.sum(wfomc_partition)
                res = res + wfomc_partition

            if self.context.contain_cardinality_constraint():
                logger.debug('Weight of the partition: %s',
                             wfomc_partition / self.context.reverse_dft_coef)
            else:
                logger.debug('Weight of the partition: %s',
                             wfomc_partition)

        if self.context.contain_cardinality_constraint():
            res = np.divide(res, self.context.reverse_dft_coef)
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
    # import sys
    # sys.setrecursionlimit(int(1e6))
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
    wfomc = WFOMC(mln, tree_constraint, cardinality_constraint)
    logger.info('WFOMC: %s', wfomc.compute())
