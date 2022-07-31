import random
import numpy as np
import argparse
import os
import logzero
import logging
import pickle

from logzero import logger
from contexttimer import Timer
from typing import List, Set, FrozenSet, Tuple, Dict
from collections import defaultdict

from sampling_ufo2.fol.syntax import Atom, Const, Lit, a, b
from sampling_ufo2.utils import MultinomialCoefficients, multinomial, TreeSumContext, \
    Rational, RingElement, coeff_monomial, round_rational, choices, \
    PREDS_FOR_EXISTENTIAL, expand
from sampling_ufo2.parser import parse_mln_constraint
from sampling_ufo2.cell_graph import Cell, CellGraph
from sampling_ufo2.context import WFOMCContext, EBType, DRWFOMCContext
from sampling_ufo2.network import MLN, TreeConstraint, CardinalityConstraint

from sampling_ufo2.wfomc import cc_weighting_fn, precompute_ext_weight, \
    assign_cell, get_config_weight_standard
from sampling_ufo2.existential_context import ExistentialContext


class Sampler(object):
    def __init__(self, mln: MLN, tree_constraint: TreeConstraint,
                 cardinality_constraint: CardinalityConstraint):
        if mln.contain_existential_quantifier():
            self.context: DRWFOMCContext = DRWFOMCContext(
                mln, tree_constraint, cardinality_constraint
            )
        else:
            self.context: WFOMCContext = WFOMCContext(
                mln, tree_constraint, cardinality_constraint
            )
        get_weight = self.context.get_weight
        if self.context.contain_cardinality_constraint():
            get_weight, self.monomial = cc_weighting_fn(
                get_weight,
                self.context.cardinality_constraint.pred2card
            )

        self.domain: List[Const] = list(self.context.domain)
        self.domain_size: int = len(self.domain)
        logger.debug('domain: %s', self.domain)
        self.cell_graph: CellGraph = CellGraph(
            self.context.sentence, get_weight
        )
        MultinomialCoefficients.setup(self.domain_size)
        self.configs, self.weights = self._get_config_weights(
            self.cell_graph, self.domain_size)

        if self.context.contain_existential_quantifier():
            # Precomputed weights for existential quantifier
            self.uni_cell_graph: CellGraph = CellGraph(
                self.context.uni_sentence, get_weight
            )
            # self.uni_cell_graph.show()
            self.configs, self.weights = self._adjust_config_weights(
                self.configs, self.weights,
                self.cell_graph, self.uni_cell_graph
            )
            self.cell_graph = self.uni_cell_graph
        wfomc = sum(self.weights)
        if wfomc == 0:
            raise RuntimeError(
                'Unsatisfiable MLN!'
            )
        round_val = round_rational(wfomc)
        logger.info('wfomc (round):%s (exp(%s))',
                    round_val, round_val.ln())
        logger.debug('Configuration weight (round): %s', list(zip(self.configs, [
            round_rational(w) for w in self.weights
        ])))
        self.cells = self.cell_graph.get_cells()

        if self.context.contain_existential_quantifier():
            self.skolem_cell_graph: CellGraph = CellGraph(
                self.context.partial_skolem_sentence, get_weight
            )
            # Precomputed weights for existential quantifier
            self.existential_weights: Dict[
                FrozenSet[Tuple[Cell, int, int]], RingElement
            ] = dict()
            logger.info('Pre-compute the weights for existential quantifiers')
            for n in range(1, self.domain_size):
                self.existential_weights.update(
                    precompute_ext_weight(
                        self.skolem_cell_graph, n, self.context)
                )
            logger.debug('pre-computed weights for existential quantifiers:\n%s',
                         self.existential_weights)

        # for measuring performance
        self.t_sampling = 0
        self.t_assigning = 0
        self.t_sampling_models = 0

    def _adjust_config_weights(self, configs: List[Tuple[int, ...]],
                               weights: List[RingElement],
                               src_cell_graph: CellGraph,
                               dest_cell_graph: CellGraph) -> \
            Tuple[List[Tuple[int, ...]], List[Rational]]:
        src_cells = src_cell_graph.get_cells()
        dest_cells = dest_cell_graph.get_cells()
        mapping_mat = np.zeros(
            (len(src_cells), len(dest_cells)), dtype=np.int32)
        for idx, cell in enumerate(src_cells):
            dest_idx = dest_cells.index(
                cell.drop_preds(prefixes=PREDS_FOR_EXISTENTIAL)
            )
            mapping_mat[idx, dest_idx] = 1

        adjusted_config_weight = defaultdict(lambda: Rational(0, 1))
        for config, weight in zip(configs, weights):
            adjusted_config_weight[tuple(np.dot(
                config, mapping_mat).tolist())] += weight
        return list(adjusted_config_weight.keys()), \
            list(adjusted_config_weight.values())

    def _sample_ext_evidences(self, cell_assignment: List[Cell],
                              cell_weight: RingElement) \
            -> Dict[Tuple[int, int], FrozenSet[Lit]]:
        ext_config = ExistentialContext(
            cell_assignment, self.context.tseitin_to_extpred)

        # Get the total weight of the current  configuration
        cell_config = tuple(cell_assignment.count(cell) for cell in self.cells)
        total_weight = self.weights[self.configs.index(
            cell_config)] / MultinomialCoefficients.coef(cell_config)

        pair_evidences: Dict[Tuple[int, int],
                             FrozenSet[Lit]] = dict()
        q = Rational(1, 1)
        while not ext_config.all_satisfied():
            selected_cell, selected_eetype = ext_config.select_eutype()
            selected_idx = ext_config.reduce_element(
                selected_cell, selected_eetype)
            logger.debug('select element: %s, cell: %s, ee type: %s',
                         selected_idx, selected_cell, selected_eetype)

            ebtype_weights: Dict[Cell, Dict[EBType, RingElement]] = dict()
            # filter all impossible EB-types
            for cell in self.cells:
                cell_pair = (selected_cell, cell)
                weights = dict()
                for ebtype in self.context.ebtypes:
                    evidences = ebtype.get_evidences()
                    if self.cell_graph.satisfiable(cell_pair, evidences):
                        weights[ebtype] = self.cell_graph.get_edge_weight(
                            cell_pair, evidences
                        )
                ebtype_weights[cell] = weights

            for eb_config in ext_config.iter_eb_config(
                ebtype_weights
            ):
                utype_weight = self.cell_graph.get_cell_weight(selected_cell)
                eb_config_per_cell = defaultdict(
                    lambda: defaultdict(lambda: 0))
                overall_eb_config = defaultdict(lambda: 0)
                for (cell, eetype), config in eb_config.items():
                    for ebtype, num in config.items():
                        eb_config_per_cell[cell][ebtype] += num
                        overall_eb_config[ebtype] += num

                if not ext_config.satisfied(selected_eetype, overall_eb_config):
                    continue

                coeff = Rational(1, 1)
                for _, config in eb_config.items():
                    coeff *= Rational(MultinomialCoefficients.coef(
                        tuple(config.values())), 1)

                total_weight_ebtype = Rational(1, 1)
                for cell, config in eb_config_per_cell.items():
                    for ebtype, num in config.items():
                        total_weight_ebtype *= (
                            ebtype_weights[cell][ebtype] ** Rational(num, 1))

                reduced_eu_config = ext_config.reduce_eu_config(eb_config)
                reduced_weight = self.existential_weights[reduced_eu_config]
                # print(q, total_weight_ebtype, utype_weight, coeff,
                #       reduced_weight)
                # print(expand(q * total_weight_ebtype * utype_weight * coeff *
                #       reduced_weight))
                w = self._get_weight_poly(
                    q * total_weight_ebtype * utype_weight * coeff *
                    reduced_weight)
                # print(w, total_weight)
                if random.random() < w / total_weight:
                    ebtype_indices = ext_config.sample_and_update(eb_config)
                    logger.debug('sampled evidences in this step:')
                    for ebtype, indices in ebtype_indices.items():
                        for idx in indices:
                            # NOTE: the element order in pair evidences matters!
                            if selected_idx < idx:
                                pair_evidences[(selected_idx, idx)
                                               ] = ebtype.get_evidences()
                            else:
                                pair_evidences[(idx, selected_idx)
                                               ] = ebtype.get_evidences(True)
                            logger.debug('(%s, %s): %s',
                                         selected_idx, idx, ebtype)
                    # Now the ebtype assignement has been determined!
                    total_weight = w / coeff
                    q *= (utype_weight * total_weight_ebtype)
                    break
                else:
                    total_weight -= w
        return pair_evidences

    def _compute_wmc_prod(
        self, cell_assignment: List[Cell],
        pair_evidences: Dict[Tuple[int, int], FrozenSet[Lit]] = None
    ) -> List[RingElement]:
        wmc_prod = [Rational(1, 1)]
        n_elements = len(cell_assignment)
        # compute from back to front
        for i in range(n_elements - 1, -1, -1):
            for j in range(n_elements - 1, max(i, 0), -1):
                cell_pair = (cell_assignment[i], cell_assignment[j])
                pair = (i, j)
                if pair_evidences is not None and pair in pair_evidences:
                    edge_weight = self.cell_graph.get_edge_weight(
                        cell_pair, frozenset(pair_evidences[pair])
                    )
                else:
                    edge_weight = self.cell_graph.get_edge_weight(cell_pair)
                prod = wmc_prod[0] * edge_weight
                wmc_prod.insert(0, prod)
        return wmc_prod

    def _get_unary_atoms(self, cell_assignment: List[Cell]) -> Set[Atom]:
        sampled_atoms = set()
        for idx, cell in enumerate(cell_assignment):
            evidences = cell.get_evidences(self.domain[idx])
            positive_lits = filter(lambda lit: lit.positive, evidences)
            sampled_atoms.update(set(
                lit.atom for lit in positive_lits
            ))
        return sampled_atoms

    def _get_weight_poly(self, weight: RingElement):
        if self.context.contain_cardinality_constraint():
            return coeff_monomial(expand(weight), self.monomial)
        return weight

    def _sample_binary_atoms(self, cell_assignment: List[Cell],
                             cell_weight: RingElement,
                             binary_evidences: FrozenSet[Lit] = None,
                             pair_evidences: Dict[Tuple[int, int],
                                                  FrozenSet[Lit]] = None) -> Set[Atom]:
        # NOTE: here the element order matters in pair_evidences!!!
        if pair_evidences is None:
            pair_evidences = defaultdict(list)
            if binary_evidences is not None:
                for evidence in binary_evidences:
                    # NOTE: we always deal with the index of domain elements here!
                    pair_index = tuple(self.domain.index(c)
                                       for c in evidence.atom.args)
                    assert len(pair_index) == 2
                    if pair_index[0] < pair_index[1]:
                        evidence = Lit(evidence.pred()(
                            a, b), evidence.positive)
                    else:
                        pair_index = (pair_index[1], pair_index[0])
                        evidence = Lit(evidence.pred()(
                            b, a), evidence.positive)
                    pair_evidences[pair_index].append(evidence)
        wmc_prod = self._compute_wmc_prod(cell_assignment, pair_evidences)
        total_weight = self._get_weight_poly(cell_weight * wmc_prod[0])
        q = Rational(1, 1)
        idx = 1
        sampled_atoms = set()
        for i, cell_1 in enumerate(cell_assignment):
            for j, cell_2 in enumerate(cell_assignment):
                if i >= j:
                    continue
                logger.debug('Sample the atom consisting of %s(%s) and %s(%s)',
                             i, self.domain[i], j, self.domain[j])
                # go through all btypes
                evidences = None
                if (i, j) in pair_evidences:
                    evidences = frozenset(pair_evidences[(i, j)])
                btypes_with_weight = self.cell_graph.get_btypes(
                    (cell_1, cell_2), evidences
                )
                # compute the sampling distribution
                for btype, btype_weight in btypes_with_weight:
                    gamma_w = self._get_weight_poly(
                        cell_weight * q * btype_weight * wmc_prod[idx]
                    )
                    if random.random() < gamma_w / total_weight:
                        sampled_raw_atoms = [
                            lit.atom for lit in btype if lit.positive]
                        r_hat = btype_weight
                        total_weight = gamma_w
                        break
                    else:
                        total_weight -= gamma_w
                # replace to real domain elements
                sampled_atoms_replaced = set(
                    self._replace_consts(
                        atom,
                        {a: self.domain[i], b: self.domain[j]}
                    ) for atom in sampled_raw_atoms
                )
                sampled_atoms.update(sampled_atoms_replaced)
                # update q
                q *= r_hat
                # move forward
                idx += 1
                logger.debug(
                    'sampled atoms at this step: %s', sampled_atoms_replaced
                )
                logger.debug('updated q: %s', q)
        return sampled_atoms

    def sample_on_config(self, config):
        logger.debug('sample on cell configuration %s', config)
        # shuffle domain elements
        random.shuffle(self.domain)
        # for tree axiom
        A = None
        force_edges = None
        with Timer() as t:
            cell_assignment, cell_weight = assign_cell(
                self.cell_graph, dict(
                    zip(self.cells, config))
            )
            sampled_atoms: Set = self._remove_aux_atoms(
                self._get_unary_atoms(cell_assignment)
            )
            logger.debug('initial unary atoms: %s', sampled_atoms)

            if self.context.contain_tree_constraint():
                config_result = self.wfomc.get_config_result_tree(
                    self.context, self.cell_graph, config)
                A, force_edges = config_result.A, config_result.force_edges
                TreeSumContext(A, force_edges)
            self.t_assigning += t.elapsed

        pair_evidences = None
        if self.context.contain_existential_quantifier():
            pair_evidences = self._sample_ext_evidences(
                cell_assignment, cell_weight)
            logger.debug('sampled existential quantified literals: %s',
                         pair_evidences)

        with Timer() as t:
            sampled_atoms.update(
                self._sample_binary_atoms(
                    cell_assignment, cell_weight,
                    pair_evidences=pair_evidences,
                )
            )
            # q = np.ones([self.context.weight_dims], dtype=self.context.dtype)
            # idx = 0
            # for i, cell_1 in enumerate(cell_assignment):
            #     for j, cell_2 in enumerate(cell_assignment):
            #         if i >= j:
            #             continue
            #         logger.debug('Sample the atom consisting of %s(%s) and %s(%s)',
            #                      i, self.domain[i], j, self.domain[j])
            #         # compute distribution
            #         sampler = self.cell_graph.samplers[frozenset(
            #             (cell_1, cell_2)
            #         )]
            #         dist = []
            #         raw_atoms = []
            #         r_hat = []
            #         if self.context.contain_tree_constraint():
            #             sampler_p, sampler_n = sampler
            #             len_p = len(sampler_p.dist)
            #             len_n = len(sampler_n.dist)
            #             connected = []
            #             disconnected_ts, connected_ts = tree_sum_context.try_connect(
            #                 (i, j)
            #             )
            #             logger.debug(
            #                 'Disconnect tree sum: %s, Connect tree sum: %s',
            #                 disconnected_ts, connected_ts
            #             )
            #             # if (i, j) is already contracted, (i, j) must be connected
            #             if tree_sum_context.is_contracted((i, j)):
            #                 valid_dist_indices = len_p
            #             else:
            #                 valid_dist_indices = len_p + len_n
            #             for gamma_idx in range(valid_dist_indices):
            #                 if gamma_idx < len_p:
            #                     gamma_dist = sampler_p.dist[gamma_idx]
            #                     tmp_ts = connected_ts
            #                     atom = sampler_p.decode(
            #                         sampler_p.codes[gamma_idx]
            #                     )
            #                     c = True
            #                 else:
            #                     if tree_sum_context.is_contracted((i, j)):
            #                         continue
            #                     gamma_dist = sampler_n.dist[gamma_idx - len_p]
            #                     tmp_ts = disconnected_ts
            #                     atom = sampler_n.decode(
            #                         sampler_n.codes[gamma_idx - len_p]
            #                     )
            #                     c = False
            #                 if self.context.contain_cardinality_constraint():
            #                     gamma_w = np.sum(
            #                         np.dot(
            #                             self.context.top_weights,
            #                             (cell_weight * q * gamma_dist *
            #                              wmc_prod[idx] * tmp_ts)
            #                         )
            #                     )
            #                     # NOTE: must be real number
            #                     gamma_w = gamma_w.real
            #                 else:
            #                     gamma_w = np.sum(
            #                         cell_weight * q * gamma_dist *
            #                         wmc_prod[idx] * tmp_ts
            #                     )
            #                 if gamma_w != 0:
            #                     dist.append(gamma_w)
            #                     raw_atoms.append(atom)
            #                     r_hat.append(gamma_dist)
            #                     connected.append(c)
            #         logger.debug('Distribution of each B-type:')
            #         for d, v in zip(dist, raw_atoms):
            #             logger.debug('%s %s', v, d)
            #         # sample
            #         sampled_idx = random.choices(
            #             range(len(dist)),
            #             weights=dist, k=1
            #         )[0]
            #         if self.context.contain_tree_constraint():
            #             if connected[sampled_idx]:
            #                 tree_sum_context.connected((i, j))
            #             else:
            #                 tree_sum_context.disconnect((i, j))
            #         sampled_raw_atoms = raw_atoms[sampled_idx]
            #         sampled_prob = r_hat[sampled_idx]
            #         # replace to real domain elements
            #         sampled_atoms_replaced = self._replace_consts(
            #             sampled_raw_atoms,
            #             {a: self.domain[i], b: self.domain[j]}
            #         )
            #         sampled_atoms = sampled_atoms.union(sampled_atoms_replaced)
            #         # update q
            #         q *= sampled_prob
            #         # move forward
            #         idx += 1
            #         logger.debug(
            #             'sampled atoms at this step: %s', sampled_atoms_replaced
            #         )
            #         logger.debug('updated q: %s', q)
            self.t_sampling_models += t.elapsed
        return self._remove_aux_atoms(sampled_atoms)

    def _remove_aux_atoms(self, atoms):
        # only return atoms with the predicate in the original MLN
        preds = self.context.mln.preds()
        return set(
            filter(lambda atom: atom.pred in preds, atoms)
        )

    def _replace_consts(self, term, replacement):
        if isinstance(term, Atom):
            args = [replacement.get(a) for a in term.args]
            return term.pred(*args)
        elif isinstance(term, Lit):
            args = [replacement.get(a) for a in term.atom.args]
            return Lit(term.atom.pred(*args), term.positive)
        else:
            raise RuntimeError(
                'Unknown type to replace constant %s', type(term)
            )

    def _get_config_weights(self, cell_graph: CellGraph, domain_size: int) \
            -> Tuple[List[Tuple[int, ...]], List[Rational]]:
        cell_weights, edge_weights = cell_graph.get_all_weights()
        cells = cell_graph.get_cells()
        configs = []
        weights = []
        for partition in multinomial(len(cells), domain_size):
            coef = MultinomialCoefficients.coef(partition)
            weight = get_config_weight_standard(
                cell_graph, dict(zip(cells, partition)))
            # weight = get_config_weight_standard_faster(
            #     partition, cell_weights, edge_weights
            # )
            if weight != 0:
                weight = coef * self._get_weight_poly(weight)
                configs.append(partition)
                weights.append(weight)
        return configs, weights

    def sample(self, k=1):
        samples = []
        sampled_idx = choices(
            list(range(len(self.configs))), weights=self.weights, k=k)

        self.t_assigning = 0
        self.t_sampling = 0
        self.t_sampling_models = 0
        # NOTE: can do it parallelly!
        for idx in sampled_idx:
            samples.append(self.sample_on_config(
                self.configs[idx]
            ))
        logger.info('elapsed time for assigning cell type: %s',
                    self.t_assigning)
        logger.info('elapsed time for sampling possible worlds: %s',
                    self.t_sampling_models)
        return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sampler for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--n_samples', '-k', type=int, required=True)
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points')
    parser.add_argument('--show_samples', '-s',
                        action='store_true', default=False)
    parser.add_argument('--debug', '-d', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
        args.show_samples = True
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')
    mln, tree_constraint, cardinality_constraint = parse_mln_constraint(
        args.input)

    with Timer() as total_t:
        with Timer() as t:
            sampler = Sampler(mln, tree_constraint, cardinality_constraint)
        logger.info('elapsed time for initializing sampler: %s', t.elapsed)
        samples = sampler.sample(args.n_samples)
        logger.info('total time for sampling: %s', total_t.elapsed)
    save_file = os.path.join(args.output_dir, 'samples.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(samples, f)
    logger.info('Samples are saved in %s', save_file)
    if args.show_samples:
        logger.info('Samples:')
        for s in samples:
            logger.info(sorted(str(i) for i in s))
