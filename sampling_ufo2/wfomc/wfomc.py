import os
import argparse
import logging
import numpy as np
import logzero

from logzero import logger
from typing import Callable, Dict, FrozenSet, List, Set, Tuple
from collections import namedtuple, defaultdict
from itertools import product
from copy import deepcopy
from functools import partial
from contexttimer import Timer

from sampling_ufo2.utils import MultinomialCoefficients, multinomial, tree_sum, \
    RingElement, PREDS_FOR_EXISTENTIAL, Rational, create_vars, expand, coeff_monomial
from sampling_ufo2.cell_graph import CellGraph, Cell
from sampling_ufo2.context import WFOMCContext, EBType
from sampling_ufo2.parser import parse_mln_constraint
from sampling_ufo2.network import CardinalityConstraint, TreeConstraint
from sampling_ufo2.fol.syntax import CNF, Const, Lit, Pred, a, b


WFOMCConfigResult = namedtuple(
    'WFOMCConfigResult', ['weight', 'A', 'force_edges']
)


def precompute_ext_weight(cell_graph: CellGraph, domain_size: int,
                          context: WFOMCContext) \
        -> Dict[FrozenSet[Tuple[Cell, FrozenSet[Pred], int]], RingElement]:
    existential_weights = defaultdict(lambda: Rational(0, 1))
    cells = cell_graph.get_cells()
    eu_configs = []
    for cell in cells:
        config = []
        for domain_pred, tseitin_preds in context.domain_to_evidence_preds.items():
            if cell.is_positive(domain_pred):
                config.append((
                    cell.drop_preds(
                        prefixes=PREDS_FOR_EXISTENTIAL), tseitin_preds
                ))
        eu_configs.append(config)

    cell_weights = []
    edge_weights = []
    for i, cell_i in enumerate(cells):
        cell_weights.append(cell_graph.get_cell_weight(cell_i))
        edge_weight = []
        for j, cell_j in enumerate(cells):
            if i > j:
                edge_weight.append(1)
            else:
                edge_weight.append(cell_graph.get_edge_weight(
                    (cell_i, cell_j)
                ))
        edge_weights.append(edge_weight)

    for partition in multinomial(len(cells), domain_size):
        # res = get_config_weight_standard(
        #     cell_graph, dict(zip(cells, partition))
        # )
        res = get_config_weight_standard_faster(
            partition, cell_weights, edge_weights
        )
        eu_config = defaultdict(lambda: 0)
        for idx, n in enumerate(partition):
            for config in eu_configs[idx]:
                eu_config[config] += n
        eu_config = dict(
            (k, v) for k, v in eu_config.items() if v > 0
        )
        existential_weights[
            frozenset((*k, v) for k, v in eu_config.items())
        ] += (Rational(MultinomialCoefficients.coef(partition), 1) * res)
    # remove duplications
    for eu_config in existential_weights.keys():
        dup_factor = Rational(MultinomialCoefficients.coef(
            tuple(c[2] for c in eu_config)
        ), 1)
        existential_weights[eu_config] /= dup_factor
    return existential_weights


def count_distribution(sentence: CNF,
                       get_weight: Callable[[Pred], Tuple[RingElement, RingElement]],
                       domain: Set[Const],
                       preds: List[Pred],
                       tree_constraint: TreeConstraint = None,
                       cardinality_constraint: CardinalityConstraint = None) -> Dict[Tuple[int, ...], RingElement]:
    pred2weight = {}
    pred2sym = {}
    if cardinality_constraint is not None:
        preds = list(set(preds).union(
            cardinality_constraint.pred2card.keys()
        ))

    # syms = symbols('x0:{}'.format(len(preds)))
    syms = create_vars('x0:{}'.format(len(preds)))
    for sym, pred in zip(syms, preds):
        if pred in pred2weight:
            continue
        weight = get_weight(pred)
        pred2weight[pred] = (weight[0] * sym, weight[1])
        pred2sym[pred] = sym

    def get_weight_new(pred: Pred) -> RingElement:
        if pred in pred2weight:
            return pred2weight[pred]
        return get_weight(pred)

    res = standard_wfomc(sentence, get_weight_new, domain, tree_constraint)

    pred2sym_index = dict(
        (pred, res.gens.index(pred2sym[pred])) for pred in preds
    )
    count_dist = {}
    for degrees, coef in res.as_dict().items():
        if cardinality_constraint is None or all(
            degrees[pred2sym_index[pred] == card]
            for pred, card in cardinality_constraint.pred2card
        ):
            count_dist[degrees] = coef
    return count_dist


def assign_cell(cell_graph: CellGraph,
                config: Dict[Cell, int]) -> Tuple[List[Cell], RingElement]:
    cell_assignment = list()
    w = 1
    for cell, n in config.items():
        for j in range(n):
            cell_assignment.append(cell)
            w = w * cell_graph.get_cell_weight(cell)
    return cell_assignment, w


def get_config_weight_standard_faster(config: List[int],
                                      cell_weights: List[RingElement],
                                      edge_weights: List[List[RingElement]]) -> RingElement:
    res = Rational(1, 1)
    for i, n_i in enumerate(config):
        if n_i == 0:
            continue
        n_i = Rational(n_i, 1)
        res *= cell_weights[i] ** n_i
        res *= edge_weights[i][i] ** (n_i * (n_i - 1) // Rational(2, 1))
        for j, n_j in enumerate(config):
            if j <= i:
                continue
            if n_j == 0:
                continue
            n_j = Rational(n_j, 1)
            res *= edge_weights[i][j] ** (n_i * n_j)
    return res


def get_config_weight_standard(cell_graph: CellGraph,
                               cell_config: Dict[Cell, int]) -> RingElement:
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


def get_config_weight_tree(cell_graph: CellGraph, cell_config: Dict[Cell, int],
                           tree_constraint: TreeConstraint) -> RingElement:
    # assign each element a cell type
    cell_assignment, cell_weight = assign_cell(cell_graph, cell_config)
    domain_size = sum(n for c, n in cell_config.items())
    A = np.zeros(
        [domain_size, domain_size],
        dtype=np.object
    )
    force_edges = []
    r = Rational(1, 1)
    for i in range(domain_size):
        for j in range(domain_size):
            if i >= j:
                continue
            edge = (cell_assignment[i], cell_assignment[j])
            ab_p = cell_graph.get_edge_weight(
                edge, frozenset([Lit(tree_constraint.pred(a, b))])
            )
            ab_n = cell_graph.get_edge_weight(
                edge, frozenset([Lit(tree_constraint.pred(a, b), False)])
            )
            if ab_n == 0:
                # WMC(\phi \land \not R(a,b)) = 0
                force_edges.append((i, j))
                A[i, j] = Rational(1, 1)
                r = r * ab_p
            else:
                # WMC(\phi \land \not R(a,b)) != 0
                A[i, j] = ab_p / ab_n
                r = r * ab_n
    ts = tree_sum(A, force_edges)
    res = ts * cell_weight * r
    return res


class ExtConfig(object):
    def __init__(self, cell_config: Dict[Cell, int],
                 tseitin_to_extpred: Dict[Pred, Pred]):
        self.cell_config = cell_config
        self.cells = list(self.cell_config.keys())
        self.tseitin_to_extpred = tseitin_to_extpred
        self.eu_config = defaultdict(lambda: 0)
        for cell, num in self.cell_config.items():
            eetype = set()
            for tseitin, ext in tseitin_to_extpred.items():
                if not cell.is_positive(ext):
                    eetype.add(tseitin)
            self.eu_config[(cell, frozenset(eetype))] = num
        logger.debug('initial eu_config: %s', self.eu_config)

    def all_satisfied(self):
        for cell in self.cells:
            if self.eu_config[(cell, frozenset())] != self.cell_config[cell]:
                return False
        return True

    def select_eutype(self):
        assert not self.all_satisfied()
        for (cell, eetype), num in self.eu_config.items():
            if len(eetype) != 0 and num > 0:
                return cell, eetype

    def reduce_element(self, cell, eetype):
        self.eu_config[(cell, eetype)] -= 1

    def reduce_eetype(self, eetype, eb_config):
        if len(eetype) == 0:
            return eetype
        for ebtype, num in eb_config.items():
            if num > 0:
                # ab_p
                eetype = eetype.difference(set(
                    tseitin for tseitin in eetype
                    if ebtype.is_positive(self.tseitin_to_extpred[tseitin])[0]
                ))
            if len(eetype) == 0:
                return eetype
        return eetype

    def satisfied(self, eetype, eb_config):
        assert len(eetype) > 0
        if len(self.reduce_eetype(eetype, eb_config)) == 0:
            return True
        return False

    def iter_eb_config(
        self,
        ebtype_weights: Dict[Cell, Dict[EBType, RingElement]]
    ):
        for raw_config in product(
                *list(multinomial(
                    len(ebtype_weights[cell]), num
                ) for (cell, _), num in self.eu_config.items())
        ):
            eb_config = defaultdict(dict)
            for i, (cell, eetype) in enumerate(self.eu_config.keys()):
                config = raw_config[i]
                for j, ebtype in enumerate(ebtype_weights[cell].keys()):
                    eb_config[(cell, eetype)][ebtype] = config[j]
            yield eb_config

    def reduce_eu_config(self, eb_config):
        reduced_eu_config = deepcopy(self.eu_config)
        for (cell, eetype), config in eb_config.items():
            for ebtype, num in config.items():
                reduced_eetype = eetype.difference(set(
                    tseitin for tseitin in eetype
                    if ebtype.is_positive(self.tseitin_to_extpred[tseitin])[1]
                ))
                reduced_eu_config[(cell, eetype)] -= num
                reduced_eu_config[(cell, reduced_eetype)] += num
        return frozenset(
            (*k, v) for k, v in reduced_eu_config.items() if v > 0
        )

    def __str__(self):
        s = ''
        for (cell, eetype), num in self.eu_config.items():
            s += 'Cell {}, {}: {}\n'.format(cell, list(eetype), num)
        return s

    def __repr__(self):
        return str(self)


def get_config_weight_existential(cell_graph: CellGraph,
                                  cell_config: Dict[Cell, int],
                                  ext_weights,
                                  context: WFOMCContext) -> WFOMCConfigResult:
    ext_config = ExtConfig(cell_config, context.tseitin_to_extpred)
    logger.debug('ext config: \n%s', ext_config)
    if ext_config.all_satisfied():
        logger.debug('All cells satisfies existential quantifier')
        res = get_config_weight_standard(cell_graph, cell_config)
        return res

    selected_cell, selected_eetype = ext_config.select_eutype()
    logger.debug('select cell: %s, ee type: %s',
                 selected_cell, selected_eetype)
    ext_config.reduce_element(selected_cell, selected_eetype)

    ebtype_weights: Dict[Cell, Dict[EBType, RingElement]] = dict()
    # filter all impossible EB-types
    # NOTE: here the order of cells matters
    for cell in cell_graph.cells:
        cell_pair = (selected_cell, cell)
        weights = dict()
        for ebtype in context.ebtypes:
            evidences = ebtype.get_evidences()
            if cell_graph.satisfiable(cell_pair, evidences):
                weights[ebtype] = cell_graph.get_edge_weight(
                    cell_pair, evidences
                )
        ebtype_weights[cell] = weights

    res = Rational(0, 1)
    for eb_config in ext_config.iter_eb_config(
        ebtype_weights
    ):
        w = Rational(1, 1)
        # logger.debug('eb_config: \t%s\n eb_config_per_cell: \t%s\n overall_eb_config:\t%s',
        #              eb_config, eb_config_per_cell, overall_eb_config)
        eb_config_per_cell = defaultdict(lambda: defaultdict(lambda: 0))
        overall_eb_config = defaultdict(lambda: 0)
        for (cell, eetype), config in eb_config.items():
            for ebtype, num in config.items():
                eb_config_per_cell[cell][ebtype] += num
                overall_eb_config[ebtype] += num

        if not ext_config.satisfied(selected_eetype, overall_eb_config):
            continue

        for _, config in eb_config.items():
            w *= Rational(MultinomialCoefficients.coef(tuple(config.values())), 1)
        for cell, config in eb_config_per_cell.items():
            for ebtype, num in config.items():
                w *= (ebtype_weights[cell][ebtype] ** Rational(num, 1))

        reduced_eu_config = ext_config.reduce_eu_config(eb_config)
        res += (w * ext_weights[reduced_eu_config])
        logger.debug('reduced eu_config %s:\t%s', reduced_eu_config,
                     ext_weights[reduced_eu_config])
    res *= cell_graph.get_cell_weight(selected_cell)
    return res


def existentially_wfomc(context: WFOMCContext,
                        get_weight: Callable[[Pred], Tuple[RingElement, RingElement]]) \
        -> RingElement:
    # here the sentence is the conjunction of universally quantified formula
    cell_graph = CellGraph(context.sentence, get_weight)
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    domain_size = len(context.domain)
    MultinomialCoefficients.setup(domain_size)

    skolem_cell_graph = CellGraph(context.skolemized_sentence, get_weight)
    with Timer() as t:
        ext_weights = precompute_ext_weight(
            skolem_cell_graph, domain_size - 1, context
        )
    logger.info('Elapsed time for pre-computing weight: %s', t.elapsed)

    logger.debug(ext_weights)

    res = Rational(0, 1)
    for partition in multinomial(n_cells, domain_size):
        coef = MultinomialCoefficients.coef(partition)
        cell_config = dict(zip(cells, partition))
        logger.debug(
            '=' * 15 + ' Compute WFOMC for the partition %s ' + '=' * 15,
            cell_config
        )
        weight = get_config_weight_existential(
            cell_graph, cell_config, ext_weights, context
        )
        res = res + coef * weight
        logger.debug(
            'Weight = %s', weight
        )
        logger.debug(
            '=' * 100
        )
    return res


def standard_wfomc(sentence: CNF, get_weight: Callable[[Pred], Tuple[RingElement, RingElement]],
                   domain: Set[Const], tree_constraint: TreeConstraint = None) -> RingElement:
    cell_graph = CellGraph(sentence, get_weight)
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)

    if tree_constraint is not None:
        get_config_weight = partial(
            get_config_weight_tree, tree_constraint=tree_constraint
        )
    else:
        get_config_weight = get_config_weight_standard

    res = Rational(0, 1)
    for partition in multinomial(n_cells, domain_size):
        coef = MultinomialCoefficients.coef(partition)
        cell_config = dict(zip(cells, partition))
        logger.debug(
            '=' * 15 + ' Compute WFOMC for the partition %s ' + '=' * 15,
            cell_config
        )
        res = res + coef * get_config_weight(
            cell_graph, cell_config
        )
    return res


def wfomc(context: WFOMCContext) -> Rational:
    ccpred2weight = {}
    if context.contain_cardinality_constraint():
        pred2card = context.cardinality_constraint.pred2card
        # syms = symbols('x0:{}'.format(len(pred2card)))
        syms = create_vars('x0:{}'.format(len(pred2card)))
        monomial = Rational(1, 1)
        for sym, (pred, card) in zip(syms, context.cardinality_constraint.pred2card):
            weight = context.get_weight(pred)
            ccpred2weight[pred] = (weight[0] * sym, weight[1])
            monomial = monomial * (sym ** Rational(card, 1))

    def get_weight_new(pred: Pred) -> RingElement:
        if pred in ccpred2weight:
            return ccpred2weight[pred]
        return context.get_weight(pred)

    if context.contain_existential_quantifier():
        res = existentially_wfomc(context, get_weight_new)
    else:
        res = standard_wfomc(
            context.sentence,
            get_weight_new,
            context.domain,
            context.tree_constraint
        )

    if context.contain_cardinality_constraint():
        res = expand(res)
        logger.debug('wfomc polynomial: %s', res)
        res = coeff_monomial(res, monomial)
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
    context = WFOMCContext(mln, tree_constraint, cardinality_constraint)
    # print(count_distribution(
    #     context.skolemized_sentence,
    #     context.get_weight,
    #     list(context.domain)[:-1],
    #     list(context.tseitin_to_extpred.keys()),
    #     context.tree_constraint,
    #     context.cardinality_constraint
    # ))
    res = wfomc(context)
    logger.info('WFOMC (arbitrary precision): %s', res)
    logger.info('WFOMC (round): %s', float(res))
