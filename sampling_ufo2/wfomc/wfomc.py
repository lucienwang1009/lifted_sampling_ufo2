import os
import argparse
import logging
import numpy as np
import logzero

from logzero import logger
from typing import Callable, Dict, FrozenSet, List, Tuple
from collections import namedtuple, defaultdict
from itertools import product
from copy import deepcopy
from gmpy2 import mpq
from sympy import symbols, Poly

from sampling_ufo2.utils import MultinomialCoefficients, multinomial, tree_sum
from sampling_ufo2.cell_graph import CellGraph, Cell
from sampling_ufo2.context import WFOMCContext
from sampling_ufo2.parser import parse_mln_constraint
from sampling_ufo2.network import MLN, TreeConstraint, CardinalityConstraint
from sampling_ufo2.fol.syntax import CNF, Lit, Pred


WFOMCConfigResult = namedtuple(
    'WFOMCConfigResult', ['weight', 'A', 'force_edges']
)


def assign_cell(cell_graph: CellGraph,
                config: Dict[Cell, int]) -> Tuple[List[Cell], Poly]:
    cell_assignment = list()
    w = 1
    for cell, n in config.items():
        for j in range(n):
            cell_assignment.append(cell)
            w = w * cell_graph.get_cell_weight(cell)
    return cell_assignment, w


def get_config_weight_standard(context: WFOMCContext, cell_graph: CellGraph,
                               cell_config: Dict[Cell, int]) -> Poly:
    res = 1
    for i, (cell_i, n_i) in enumerate(cell_config.items()):
        if n_i == 0:
            continue
        res = res * cell_graph.get_cell_weight(cell_i) ** n_i
        res = res * cell_graph.get_edge_weight(
            (cell_i, cell_i)
        ) ** (n_i * (n_i - 1) // 2)
        for j, (cell_j, n_j) in enumerate(cell_config.items()):
            if j <= i:
                continue
            if n_j == 0:
                continue
            res = res * cell_graph.get_edge_weight(
                (cell_i, cell_j)
            ) ** (n_i * n_j)
    return res


def get_config_weight_tree(context: WFOMCContext, cell_graph: CellGraph,
                           cell_config: Dict[Cell, int]) -> Poly:
    # assign each element a cell type
    cell_assignment, cell_weight = assign_cell(cell_graph, cell_config)
    domain_size = len(context.domain)
    A = np.zeros(
        [domain_size, domain_size],
        dtype=np.object
    )
    force_edges = []
    r = mpq(1)
    for i in range(domain_size):
        for j in range(domain_size):
            if i >= j:
                continue
            edge = (cell_assignment[i], cell_assignment[j])
            ab_p = cell_graph.get_edge_weight(
                edge, context.tree_p_evidence
            )
            ab_n = cell_graph.get_edge_weight(
                edge, context.tree_n_evidence
            )
            if ab_n == 0:
                # WMC(\phi \land \not R(a,b)) = 0
                force_edges.append((i, j))
                A[i, j] = mpq(1)
                r = r * ab_p
            else:
                # WMC(\phi \land \not R(a,b)) != 0
                A[i, j] = ab_p / ab_n
                r = r * ab_n
    ts = tree_sum(A, force_edges)
    res = ts * cell_weight * r
    return res


def wfomc(context: WFOMCContext):
    ccpred2weight = {}
    if context.contain_cardinality_constraint():
        pred2card = cardinality_constraint.pred2card
        syms = symbols('x0:{}'.format(len(pred2card)))
        monomial = mpq(1)
        for sym, (pred, card) in zip(syms, context.cardinality_constraint.pred2card):
            weight = context.get_weight(pred)
            ccpred2weight[pred] = (Poly(weight[0] * sym), weight[1])
            monomial = monomial * (sym ** card)

    def get_weight_new(pred: Pred) -> Poly:
        if pred in ccpred2weight:
            return ccpred2weight[pred]
        return context.get_weight(pred)

    cell_graph = CellGraph(context.sentence, get_weight_new)
    cell_graph.show()
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    domain_size = len(context.domain)
    MultinomialCoefficients.setup(domain_size)

    res = mpq(0)
    for partition in multinomial(n_cells, domain_size):
        coef = MultinomialCoefficients.coef(partition)
        cell_config = dict(zip(cells, partition))
        logger.debug(
            '=' * 15 + ' Compute WFOMC for the partition %s ' + '=' * 15,
            cell_config
        )
        if tree_constraint is not None:
            res = res + coef * \
                get_config_weight_tree(context, cell_graph, cell_config)
        else:
            res = res + coef * \
                get_config_weight_standard(context, cell_graph, cell_config)

    if cardinality_constraint is not None:
        res = Poly(res, syms)
        res = res.coeff_monomial(monomial)
    return res


class WFOMC(object):
    def __init__(self, sentence: CNF, get_weight: Callable[[Pred], Tuple[mpq, mpq]],
                 tree_constraint: TreeConstraint,
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
        MultinomialCoefficients.setup(self.domain_size)
        if self.context.contain_existential_quantifier():
            # { (cell, n, k): wfomc }
            self.existential_weights = self.precompute_ext_weight(
                self.domain_size - 1
            )
            logger.debug('Pre-computed weights for existential quantified formula: %s',
                         self.existential_weights)

    def assign_cell(self, cell_graph: CellGraph,
                    config: Dict[Cell, int]) -> Tuple[List[Cell], Poly]:
        cell_assignment = list()
        w = 1
        for cell, n in config.items():
            for j in range(n):
                cell_assignment.append(cell)
                w = w * cell_graph.get_cell_weight(cell)
        return cell_assignment, w

    def _get_config_weight_standard(self, cell_graph: CellGraph,
                                    cell_config: Dict[Cell, int]) -> Poly:
        res = 1
        for i, (cell_i, n_i) in enumerate(cell_config.items()):
            if n_i == 0:
                continue
            res = res * cell_graph.get_cell_weight(cell_i) ** n_i
            res = res * cell_graph.get_edge_weight(
                (cell_i, cell_i)
            ) ** (n_i * (n_i - 1) // 2)
            for j, (cell_j, n_j) in enumerate(cell_config.items()):
                if j <= i:
                    continue
                if n_j == 0:
                    continue
                res = res * cell_graph.get_edge_weight(
                    (cell_i, cell_j)
                ) ** (n_i * n_j)
        return res

    def precompute_ext_weight(self, domain_size: int) -> Dict[FrozenSet[Tuple[Cell, int, int]], np.ndarray]:
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
                cell_graph, dict(zip(cells, partition))
            )
            ext_n_k = defaultdict(lambda: [0, 0])
            for cell, n in zip(cells, partition):
                cell_dropped_skolem = cell.drop_pred(skolem_pred)
                if cell.is_positive(skolem_pred):
                    ext_n_k[cell_dropped_skolem][0] += n
                else:
                    ext_n_k[cell_dropped_skolem][0] += n
                    ext_n_k[cell_dropped_skolem][1] += n
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
                        zip(ext_n_k.keys(), [
                            n for n, k in ext_n_k.values()], ks)
                    )
                ] += coef * res
        return existential_weights

    def _get_config_result_existential(self, cell_config: Dict[Cell, int]) -> WFOMCConfigResult:
        ext_pred = self.context.ext_preds[0]
        cell_graph = self.context.cell_graph
        cells, config = list(zip(*cell_config.items()))
        config = list(config)
        tseitin_pred = self.context.ext_preds[0]
        already_satisfied_ext = [cell.is_positive(
            tseitin_pred) for cell in cells]
        k = [n if not already_satisfied_ext[idx]
             else 0 for idx, n in enumerate(config)]
        # all elements are satisfied the existential quantified formula
        if all(already_satisfied_ext[i] for i, n in enumerate(config) if n != 0):
            # the weight is exactly that of sentence without existential quantifiers
            logger.debug('All cells satisfies existential quantifier')
            res = self._get_config_weight_standard(cell_graph, cell_config)
            return WFOMCConfigResult(res, None, None)
        # choose any element
        for i, satisfied in enumerate(already_satisfied_ext):
            if config[i] > 0 and not satisfied:
                selected_idx = i
                break
        selected_cell = cells[selected_idx]
        logger.debug('select cell: %s', selected_cell)
        # domain recursion
        config[selected_idx] -= 1
        k[selected_idx] -= 1
        edge_weights: Dict[Cell, Dict[FrozenSet[Lit], np.ndarray]] = dict()
        # filter the impossible existential B-types
        # NOTE: here the order of cells matters
        for cell in cells:
            cell_pair = (selected_cell, cell)
            weights = dict()
            for ext_btype in self.context.ext_btypes:
                evidences = ext_btype.get_evidences()
                if cell_graph.satisfiable(cell_pair, evidences):
                    weights[ext_btype] = cell_graph.get_edge_weight(
                        cell_pair, evidences
                    )
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
            if satisfied:  # or already_satisfied_ext[selected_idx]:
                logger.debug('W_e: %s', W_e)
                logger.debug('reduced wfomc for cells=%s, n=%s, k=%s: ',
                             cells, config, reduced_k)
                reduced_wfomc = self.existential_weights[frozenset(
                    zip(cells, config, reduced_k)
                )]
                logger.debug('%s', reduced_wfomc)
                res += (W_e * reduced_wfomc)
        return WFOMCConfigResult(res, None, None)

    def _get_config_result_tree(self, cell_config: Dict[Cell, int]) -> WFOMCConfigResult:
        cell_assignment, w = self.assign_cell(
            self.context.cell_graph, cell_config
        )
        # assign each element a cell type
        A = np.zeros(
            [self.domain_size, self.domain_size, self.context.weight_dims],
            dtype=self.context.dtype
        )
        f = []
        r = 1
        for i in range(self.domain_size):
            for j in range(self.domain_size):
                if i >= j:
                    continue
                edge = (cell_assignment[i], cell_assignment[j])
                ab_p = self.context.cell_graph.get_edge_weight(
                    edge, self.context.tree_p_evidence
                )
                ab_n = self.context.cell_graph.get_edge_weight(
                    edge, self.context.tree_n_evidence
                )
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
        return WFOMCConfigResult(
            res, A, f
        )

    def get_config_result(self, cell_config: Dict[Cell, int]) -> WFOMCConfigResult:
        if self.context.contain_existential_quantifier():
            return self._get_config_result_existential(cell_config)
        elif self.context.contain_tree_constraint():
            return self._get_config_result_tree(cell_config)
        else:
            res = self._get_config_weight_standard(
                self.context.cell_graph, cell_config
            )
            return WFOMCConfigResult(res, None, None)

    def compute(self):
        cells = self.context.cell_graph.get_cells()
        n_cells = len(cells)

        res = 0
        for partition in multinomial(n_cells, self.domain_size):
            coef = MultinomialCoefficients.coef(partition)
            cell_config = dict(zip(cells, partition))
            logger.debug(
                '=' * 15 + ' Compute WFOMC for the partition %s ' + '=' * 15,
                cell_config
            )
            config_weight = self.get_config_result(cell_config).weight

            if self.context.contain_cardinality_constraint():
                config_weight = coef * \
                    np.sum(np.dot(self.context.top_weights, config_weight))
                res = res + config_weight
            else:
                config_weight = coef * config_weight
                res = res + config_weight

            if self.context.contain_cardinality_constraint():
                logger.debug('Weight of the config: %s',
                             config_weight / self.context.reverse_dft_coef)
            else:
                logger.debug('Weight of the config: %s',
                             config_weight)

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
        args.input
    )
    # algorithm = WFOMCAlgorithm(mln, tree_constraint, cardinality_constraint)
    # res = algorithm.compute()
    context = WFOMCContext(mln, tree_constraint, cardinality_constraint)
    res = wfomc(context)
    logger.info('WFOMC (arbitrary precision): %s', res)
    logger.info('WFOMC (round): %s', float(res))
