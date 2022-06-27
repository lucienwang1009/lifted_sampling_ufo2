import random
import numpy as np
import argparse
import os
import logzero
import logging
import pickle
import pandas as pd

from logzero import logger
from contexttimer import Timer
from typing import List, Set, FrozenSet, Tuple, Dict
from collections import defaultdict
from itertools import product

from sampling_ufo2.fol.syntax import Atom, Pred, Const, a, b, Lit
from sampling_ufo2.utils import MultinomialCoefficients, multinomial, TreeSumContext
from sampling_ufo2.parser import parse_mln_constraint
from sampling_ufo2.wfomc.wfomc import WFOMC
from sampling_ufo2.cell_graph import Cell, CellGraph
from sampling_ufo2.context import Context, ExtBType
from sampling_ufo2.network import MLN, TreeConstraint, CardinalityConstraint


class ExtConfig(object):
    def __init__(self, cell_assignment: List[Cell], ext_preds: List[Pred]):
        self.cell_assignment: List[Cell] = cell_assignment
        self.cells: List[Cell] = list(set(self.cell_assignment))
        self.ext_preds: List[Pred] = ext_preds
        assert len(self.ext_preds) == 1
        self.ext_pred = self.ext_preds[0]

        # [element_idx, cell, ext_pred1, ext_pred2, ..., ext_predm]
        self.config: pd.DataFrame
        df_list = []
        for cell in self.cell_assignment:
            row = [cell]
            for pred in self.ext_preds:
                if cell.is_positive(pred):
                    row.append(True)
                else:
                    row.append(False)
            df_list.append(row)
        self.config = pd.DataFrame(
            df_list,
            columns=['cell'] + self.ext_preds,
            index=range(len(self.cell_assignment))
        )

    def all_satisfied(self) -> bool:
        return self.config[self.ext_preds].all(None)

    def empty(self) -> bool:
        return self.config.empty

    def get_unsatisfied(self) -> int:
        assert not self.all_satisfied()
        return self.config[
            ~self.config[self.ext_preds].all(axis=1)
        ].index[0]

    def remove(self, index: int) -> None:
        self.config.drop(index, inplace=True)

    def cell_config(self) -> List[Tuple[Cell, int]]:
        return [(cell, n) for cell, (n, k) in self.config.items()]

    def size(self, cell: Cell) -> int:
        return (self.config.cell == cell).sum()

    def unsatisfied_size(self, cell: Cell) -> int:
        return (~self.config[self.config.cell == cell][self.ext_pred]).sum()

    def satisfied_size(self, cell: Cell) -> int:
        return self.config[self.config.cell == cell][self.ext_pred].sum()

    def unsatisfied(self, cell: Cell) -> List[int]:
        cell_elements = self.config[self.config.cell == cell]
        return cell_elements[~cell_elements[self.ext_pred]].index.to_list()

    def satisfied(self, cell: Cell) -> List[int]:
        cell_elements = self.config[self.config.cell == cell]
        return cell_elements[cell_elements[self.ext_pred]].index.to_list()

    def satisfies(self, index: int):
        self.config.at[index, self.ext_pred] = True

    def __str__(self):
        return self.config.to_markdown()

    def __repr__(self):
        return str(self)


class Sampler(object):
    def __init__(self, mln: MLN, tree_constraint: TreeConstraint, cardinality_constraint: CardinalityConstraint):
        with Timer() as t:
            self.wfomc: WFOMC = WFOMC(
                mln, tree_constraint, cardinality_constraint)
            self.context: Context = self.wfomc.context
            self.cell_graph: CellGraph = self.context.cell_graph
            if logzero.loglevel == logging.DEBUG:
                self.context.cell_graph.show()
            self.domain_size = len(self.context.domain)
            self.configs, self.weights = self._get_config_weights()
            logger.debug('Configuration weight: %s',
                         tuple(zip(self.configs, self.weights)))
            logger.info(
                'elapsed time for computing configuration weight: %s', t.elapsed
            )
        wfomc = np.sum(self.weights)
        assert np.abs(wfomc) > 1e-5, 'Input sentence is unsatisfiable'
        if self.context.contain_cardinality_constraint():
            logger.info('wfomc:%s', np.divide(
                wfomc, self.context.reverse_dft_coef)
            )
        else:
            logger.info('wfomc:%s', wfomc)
        logger.debug(list(zip(self.configs, self.weights)))
        assert len([i for i in self.weights if i < 0]) == 0, "negative weight"
        self.domain: List[Const] = list(self.context.domain)
        self.domain_size: int = len(self.domain)
        logger.debug('domain: %s', self.domain)

        if self.context.contain_existential_quantifier():
            # Precomputed weights for existential quantifier
            self.existential_weights: Dict[
                FrozenSet[Tuple[Cell, int, int]], np.ndarray
            ] = dict()
            logger.info('Pre-compute the weights for existential quantifiers')
            for n in range(1, self.domain_size):
                self.existential_weights.update(
                    self.wfomc.precompute_ext_weight(n)
                )

            self.n_cells = len(self.cell_graph.cells)
            # self.existential_weights[
            #     frozenset(zip(self.cell_graph.cells,
            #                   [0] * self.n_cells,
            #                   [0] * self.n_cells))
            # ] = 1
            logger.debug('pre-computed weights for existential quantifiers:\n%s',
                         self.existential_weights)

        # for measuring performance
        self.t_sampling = 0
        self.t_assigning = 0
        self.t_sampling_worlds = 0

    def _sample_ext_evidences(self, cell_assignment: List[Cell]) -> FrozenSet[Lit]:
        ext_config = ExtConfig(cell_assignment, self.context.ext_preds)
        logger.debug('initial configuration for existential quantified predicates:\n%s',
                     ext_config)
        ext_pred = self.context.ext_preds[0]
        cells = self.cell_graph.cells
        sampled_evidences: Set[Lit] = set()
        accum_weight = 1
        while(not ext_config.empty() and not ext_config.all_satisfied()):
            unsatisfied_index = ext_config.get_unsatisfied()
            logger.debug('select unsatisfied element: %s(%s)',
                         self.domain[unsatisfied_index], unsatisfied_index)
            ext_config.remove(unsatisfied_index)
            unsatisfied_cell = cell_assignment[unsatisfied_index]
            edge_weights: Dict[Cell, Dict[FrozenSet[Lit], np.ndarray]] = dict()
            # filter the impossible existential B-types
            for cell in cells:
                cell_pair = (unsatisfied_cell, cell)
                weights = dict()
                for ext_btype in self.context.ext_btypes:
                    evidences = ext_btype.get_evidences()
                    if self.cell_graph.satisfiable(cell_pair, evidences):
                        weights[ext_btype] = self.cell_graph.get_edge_weight(
                            cell_pair, evidences
                        )
                edge_weights[cell] = weights

            init_weight = self.cell_graph.get_cell_weight(unsatisfied_cell)
            # for cell in cells:
            #     init_weight = init_weight * np.power(
            #         self.cell_graph.get_edge_weight(
            #             (unsatisfied_cell, cell)
            #         ), ext_config.satisfied_size(cell)
            #     )
            dist: List = []
            reduced_configs: List[Dict[Cell,
                                       List[Tuple[ExtBType, int, int]]]] = []
            weights: List = []
            for unsatisfied_configs in product(
                    *list(multinomial(
                        len(edge_weights[cell]
                            ), ext_config.unsatisfied_size(cell)
                    ) for cell in cells)
            ):
                for satisfied_configs in product(
                    *list(multinomial(
                        len(edge_weights[cell]
                            ), ext_config.satisfied_size(cell)

                    )for cell in cells)
                ):
                    satisfied = False
                    reduced_config: Dict[Cell,
                                         List[Tuple[ExtBType, int, int]]] = dict()
                    for idx, (unsatisfied_config,
                              satisfied_config,
                              edge_weight) in enumerate(
                            zip(unsatisfied_configs,
                                satisfied_configs,
                                edge_weights.values())
                    ):
                        # cell_config: Dict[ExtBType, int] = defaultdict(lambda: 0)
                        for unsatisfied_num, satisfied_num, ext_btype in \
                                zip(unsatisfied_config,
                                    satisfied_config,
                                    edge_weight.keys()):
                            num = unsatisfied_num + satisfied_num
                            ab_p, ba_p = ext_btype.is_positive(ext_pred)
                            if num > 0 and ab_p:
                                satisfied = True
                            # cell_config[ext_btype] += num
                        reduced_config[cells[idx]] = tuple(zip(
                            edge_weight.keys(),
                            unsatisfied_config,
                            satisfied_config
                        ))

                    if satisfied:
                        logger.debug('satisfied reduce config: %s, %s',
                                     unsatisfied_configs, satisfied_configs)
                        weight = init_weight
                        reduced_k = [
                            ext_config.unsatisfied_size(cell) for cell in cells
                        ]
                        for idx, (cell, cell_config) in enumerate(reduced_config.items()):
                            unsatisfied_config = []
                            satisfied_config = []
                            for ext_btype, unsatisfied_num, satisfied_num in cell_config:
                                unsatisfied_config.append(unsatisfied_num)
                                satisfied_config.append(satisfied_num)
                                num = unsatisfied_num + satisfied_num
                                weight = weight * \
                                    np.power(
                                        edge_weights[cell][ext_btype], num)
                                if ext_btype.is_positive(ext_pred)[1]:
                                    reduced_k[idx] -= unsatisfied_num
                            weight = weight * \
                                MultinomialCoefficients.coef(
                                    tuple(unsatisfied_config)
                                ) * \
                                MultinomialCoefficients.coef(
                                    tuple(satisfied_config)
                                )
                        assert all(k >= 0 for k in reduced_k)
                        reduced_wfomc = self.existential_weights[frozenset(
                            zip(cells, [ext_config.size(cell)
                                        for cell in cells], reduced_k)
                        )]
                        if self.context.contain_cardinality_constraint():
                            d = np.sum(np.dot(
                                self.context.top_weights, accum_weight * weight * reduced_wfomc
                            ))
                            d = d.real
                        else:
                            d = np.sum(
                                accum_weight * weight * reduced_wfomc
                            )
                        if np.abs(d) > 1e-10:
                            dist.append(d)
                            reduced_configs.append(reduced_config)
                            weights.append(weight)
            logger.debug('========= reduce config with weight: ========')
            for config, weight in zip(reduced_configs, dist):
                logger.debug('%s: %s', config, weight)
            # sample
            sampled_idx = random.choices(
                range(len(dist)),
                weights=dist, k=1
            )[0]
            sampled_config = reduced_configs[sampled_idx]
            logger.debug('sampled config: %s', sampled_config)
            accum_weight *= weights[sampled_idx]
            step_sampled_evidences = set()
            for cell, cell_config in sampled_config.items():
                unsatisfied_indices = ext_config.unsatisfied(cell)
                satisfied_indices = ext_config.satisfied(cell)
                # NOTE: still need a shuffle here?
                # random.shuffle(unsatisfied_indices)
                # random.shuffle(satisfied_indices)
                idx = 0
                for ext_btype, unsatisfied_num, satisfied_num in cell_config:
                    for _ in range(unsatisfied_num):
                        evidences = ext_btype.get_evidences()
                        step_sampled_evidences.update(
                            self._replace_consts(
                                e, {a: self.domain[unsatisfied_index],
                                    b: self.domain[unsatisfied_indices[idx]]}
                            ) for e in evidences
                        )
                        if ext_btype.is_positive(ext_pred)[1]:
                            ext_config.satisfies(unsatisfied_indices[idx])
                            logger.debug('index %s satisfies the existential quantifier',
                                         unsatisfied_indices[idx])
                        idx += 1
                idx = 0
                for ext_btype, unsatisfied_num, satisfied_num in cell_config:
                    for _ in range(satisfied_num):
                        evidences = ext_btype.get_evidences()
                        step_sampled_evidences.update(
                            self._replace_consts(
                                e, {a: self.domain[unsatisfied_index],
                                    b: self.domain[satisfied_indices[idx]]}
                            ) for e in evidences
                        )
                        idx += 1
            logger.debug('sampled evidences at this step: %s',
                         step_sampled_evidences)
            sampled_evidences.update(step_sampled_evidences)
            logger.debug('existential config:\n%s', ext_config)
        return sampled_evidences

    def sample_on_config_with_ext(self, config) -> Set[Atom]:
        random.shuffle(self.domain)
        cell_assignment, cell_weight = self.wfomc.assign_cell(
            self.cell_graph, config
        )
        sampled_atoms = self._remove_aux_atoms(
            self._get_unary_atoms(cell_assignment)
        )
        logger.debug('initial unary atoms: %s', sampled_atoms)
        existential_evidences = self._sample_ext_evidences(cell_assignment)
        logger.debug('sampled existential quantified literals: %s',
                     existential_evidences)
        sampled_atoms = sampled_atoms.union(
            self._sample_binary_atoms(
                cell_assignment,
                cell_weight,
                existential_evidences
            )
        )
        return self._remove_aux_atoms(sampled_atoms)

    def _compute_wmc_prod(
        self, cell_assignment: List[Cell],
        pair_evidences: Dict[Tuple[Const, Const], FrozenSet[Lit]] = None
    ) -> List[np.ndarray]:
        wmc_prod = [np.ones(
            [self.context.weight_dims], dtype=self.context.dtype
        )]
        n_elements = len(cell_assignment)
        # compute from back to front
        for i in range(n_elements - 1, -1, -1):
            for j in range(n_elements - 1, max(i, 1), -1):
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
            sampled_atoms = sampled_atoms.union(set(
                lit.atom for lit in positive_lits
            ))
        return sampled_atoms

    def _sample_binary_atoms(self, cell_assignment: List[Cell],
                             cell_weight: np.ndarray,
                             binary_evidences: FrozenSet[Lit] = None) -> Set[Atom]:
        # compute wmc_prod
        pair_evidences = defaultdict(list)
        if binary_evidences is not None:
            for evidence in binary_evidences:
                # NOTE: we always deal with the index of domain elements here!
                pair_index = tuple(self.domain.index(c)
                                   for c in evidence.atom.args)
                assert len(pair_index) == 2
                if pair_index[0] < pair_index[1]:
                    evidence = Lit(evidence.pred()(a, b), evidence.positive)
                else:
                    pair_index = (pair_index[1], pair_index[0])
                    evidence = Lit(evidence.pred()(b, a), evidence.positive)
                pair_evidences[pair_index].append(evidence)
        wmc_prod = self._compute_wmc_prod(cell_assignment, pair_evidences)
        q = np.ones([self.context.weight_dims], dtype=self.context.dtype)
        idx = 0
        sampled_atoms = set()
        for i, cell_1 in enumerate(cell_assignment):
            for j, cell_2 in enumerate(cell_assignment):
                if i >= j:
                    continue
                logger.debug('Sample the atom consisting of %s(%s) and %s(%s)',
                             i, self.domain[i], j, self.domain[j])
                # compute distribution
                btypes_with_weight = self.cell_graph.get_btypes(
                    (cell_1, cell_2), frozenset(pair_evidences[(i, j)])
                )
                # compute the sampling distribution
                dist = []
                raw_atoms = []
                r_hat = []
                for btype, btype_weight in btypes_with_weight:
                    if self.context.contain_cardinality_constraint():
                        gamma_w = np.sum(
                            np.dot(
                                self.context.top_weights,
                                cell_weight * q * btype_weight * wmc_prod[idx]
                            )
                        )
                        # NOTE: must be real number
                        gamma_w = gamma_w.real
                    else:
                        gamma_w = np.sum(
                            cell_weight * q * btype_weight * wmc_prod[idx]
                        )
                    if np.abs(gamma_w) > 1e-10:
                        dist.append(gamma_w)
                        raw_atoms.append(
                            [lit.atom for lit in btype if lit.positive]
                        )
                        r_hat.append(btype_weight)
                logger.debug('Distribution of each B-type:')
                for d, v in zip(dist, raw_atoms):
                    logger.debug('%s %s', v, d)
                # sample
                sampled_idx = random.choices(
                    range(len(dist)),
                    weights=dist, k=1
                )[0]
                sampled_raw_atoms = raw_atoms[sampled_idx]
                sampled_prob = r_hat[sampled_idx]
                # replace to real domain elements
                sampled_atoms_replaced = set(
                    self._replace_consts(
                        atom,
                        {a: self.domain[i], b: self.domain[j]}
                    ) for atom in sampled_raw_atoms
                )
                sampled_atoms = sampled_atoms.union(sampled_atoms_replaced)
                # update q
                q *= sampled_prob
                # move forward
                idx += 1
                logger.debug(
                    'sampled atoms at this step: %s', sampled_atoms_replaced
                )
                logger.debug('updated q: %s', q)
        return sampled_atoms

    def sample_on_config(self, config):
        logger.debug('sample on cell configuration %s', config)
        if self.context.contain_existential_quantifier():
            return self._sample_ext_on_config(config)
        # shuffle domain elements
        random.shuffle(self.domain)
        # for tree axiom
        A = None
        force_edges = None
        with Timer() as t:
            cell_assignment, cell_weight = self.wfomc.assign_cell(
                self.cell_graph, config
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
            # compute wmc_prod
            # wmc_prod = self._compute_wmc_prod(
            #     cell_assignment, force_edges
            # )
            self.t_assigning += t.elapsed

        with Timer() as t:
            sampled_atoms = sampled_atoms.union(
                self._sample_binary_atoms(
                    cell_assignment, cell_weight
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
            self.t_sampling_worlds += t.elapsed
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

    def _get_config_weights(self):
        configs = []
        weights = []

        cells = self.context.cell_graph.get_cells()
        for partition in multinomial(len(cells), self.domain_size):
            coef = MultinomialCoefficients.coef(partition)
            config = dict(zip(cells, partition))
            config_result = self.wfomc.get_config_result(config)
            if self.context.contain_cardinality_constraint():
                # NOTE: the weight must be real numbers
                config_weight = np.sum(
                    np.dot(self.context.top_weights, config_result.weight)
                )
                weight = config_weight.real if config_weight.real > 0 else 0
            else:
                weight = np.sum(config_result.weight)
            configs.append(config)
            weights.append(coef * weight)
        return configs, weights

    def sample(self, k=1):
        samples = []
        sampled_configs = random.choices(
            self.configs, weights=self.weights, k=k)
        self.t_assigning = 0
        self.t_sampling = 0
        self.t_sampling_worlds = 0
        for sampled_config in sampled_configs:
            if self.context.contain_existential_quantifier():
                samples.append(self.sample_on_config_with_ext(sampled_config))
            else:
                samples.append(self.sample_on_config(sampled_config))
        logger.info('elapsed time for assigning cell type: %s',
                    self.t_assigning)
        logger.info('elapsed time for sampling possible worlds: %s',
                    self.t_sampling_worlds)
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
    parser.add_argument('--debug', action='store_true', default=False)
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
