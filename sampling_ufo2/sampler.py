import random
import numpy as np
import argparse
import os
import logzero
import logging
import pickle

from logzero import logger
from typing import Tuple
from contexttimer import Timer

from sampling_ufo2.fol.syntax import a, b
from sampling_ufo2.utils import MultinomialCoefficients, TreeSumContext
from sampling_ufo2.cell_graph import CellGraph
from sampling_ufo2.context import Context
from sampling_ufo2.parser import parse_mln_constraint
from sampling_ufo2.wfomc.wfomc import get_config_weight, assign_cell


class Sampler(object):
    def __init__(self, mln, tree_constraint, cardinality_constraint):
        with Timer() as t:
            self.context = Context(mln, tree_constraint,
                                   cardinality_constraint)
            self.cell_graph = CellGraph(self.context)
            self.n_cells = len(self.cell_graph.cells)
            self.cell_graph.show()
            self.domain_size = len(self.context.domain)
            # for tree axiom
            self.config, self.weights = self._get_config_weights()
            logger.info('Compute U-type configuration weight: %s', t.elapsed)
        wfomc = np.sum(self.weights)
        assert np.abs(wfomc) > 1e-5, 'Input sentence is unsatisfiable'
        logger.info('wfomc:%s', wfomc)
        logger.debug(list(zip(self.config, self.weights)))
        assert len([i for i in self.weights if i < 0]) == 0, "negative weight"
        self.domain = list(self.context.domain)
        logger.debug('domain: %s', self.domain)

        # for measuring performance
        self.t_sampling = 0
        self.t_assigning = 0
        self.t_sampling_worlds = 0

    def _compute_wmc_prod(self, cell_assignment, force_edges=None):
        logger.debug('cell assignment: %s', cell_assignment)
        wmc_prod = []
        # compute from back to front
        for i in range(self.domain_size - 1, -1, -1):
            for j in range(self.domain_size - 1, i, -1):
                cell_1 = cell_assignment[i]
                cell_2 = cell_assignment[j]
                if not wmc_prod:
                    wmc_prod.append(np.ones(
                        [self.context.weight_dims], dtype=self.context.dtype))
                    continue
                if self.context.contain_tree_constraint():
                    if (i, j) in force_edges:
                        edge_weight = self.cell_graph.get_edge_weight(
                            frozenset((cell_1, cell_2)))[0]
                    else:
                        edge_weight = self.cell_graph.get_edge_weight(
                            frozenset((cell_1, cell_2)))[1]
                else:
                    edge_weight = self.cell_graph.get_edge_weight(frozenset(
                        (cell_1, cell_2)))
                prod = wmc_prod[0] * edge_weight
                wmc_prod.insert(0, prod)
        return wmc_prod

    def _sample_on_config(self, config):
        logger.debug('sample from cell configuration %s', config)
        random.shuffle(self.domain)
        sampled_atoms = set()
        A = None
        force_edges = None
        tree_sum_context = None
        with Timer() as t:
            # shuffle domain elements
            cell_assignment, cw = assign_cell(self.cell_graph, config)
            for idx, cell in enumerate(cell_assignment):
                evidences = cell.get_evidences(self.domain[idx])
                positive_lits = filter(lambda lit: lit.positive, evidences)
                sampled_atoms = sampled_atoms.union(set(
                    lit.atom for lit in positive_lits
                ))
            logger.debug('initial atoms: %s', sampled_atoms)

            if self.context.contain_tree_constraint():
                config_weight = get_config_weight(
                    self.context, self.cell_graph, config)
                A = config_weight.A
                force_edges = config_weight.force_edges
                tree_sum_context = TreeSumContext(A, force_edges)
            # compute wmc_prod
            wmc_prod = self._compute_wmc_prod(
                cell_assignment, force_edges)
            # logger.debug('wmc prod: %s', wmc_prod)
            # logger.debug('cw: %s', cw)
            self.t_assigning += t.elapsed

        with Timer() as t:
            q = np.ones([self.context.weight_dims], dtype=self.context.dtype)
            idx = 0
            for i, cell_1 in enumerate(cell_assignment):
                for j, cell_2 in enumerate(cell_assignment):
                    if i >= j:
                        continue
                    logger.debug('Sample the atom consisting of %s(%s) and %s(%s)',
                                 i, self.domain[i], j, self.domain[j])
                    # compute distribution
                    sampler = self.cell_graph.samplers[frozenset(
                        (cell_1, cell_2))]
                    dist = []
                    raw_atoms = []
                    r_hat = []
                    if self.context.contain_tree_constraint():
                        sampler_p, sampler_n = sampler
                        len_p = len(sampler_p.dist)
                        len_n = len(sampler_n.dist)
                        connect = []
                        disconnected_ts, connected_ts = tree_sum_context.try_connect(
                            (i, j))
                        logger.debug('Disconnect tree sum: %s, Connect tree sum: %s',
                                     disconnected_ts, connected_ts)
                        # if (i, j) is already contracted, (i, j) must be connected
                        if tree_sum_context.is_contracted((i, j)):
                            valid_dist_indices = len_p
                        else:
                            valid_dist_indices = len_p + len_n
                        for gamma_idx in range(valid_dist_indices):
                            if gamma_idx < len_p:
                                gamma_dist = sampler_p.dist[gamma_idx]
                                tmp_ts = connected_ts
                                atom = sampler_p.decode(
                                    sampler_p.codes[gamma_idx])
                                c = True
                            else:
                                if tree_sum_context.is_contracted((i, j)):
                                    continue
                                gamma_dist = sampler_n.dist[gamma_idx - len_p]
                                tmp_ts = disconnected_ts
                                atom = sampler_n.decode(
                                    sampler_n.codes[gamma_idx - len_p])
                                c = False
                            if self.context.contain_cardinality_constraint():
                                gamma_w = np.sum(
                                    np.dot(
                                        self.context.top_weights,
                                        (cw * q * gamma_dist *
                                         wmc_prod[idx] * tmp_ts)
                                    )
                                )
                                # NOTE: must be real number
                                gamma_w = gamma_w.real
                            else:
                                gamma_w = np.sum(
                                    cw * q * gamma_dist * wmc_prod[idx] * tmp_ts)
                            if gamma_w != 0:
                                dist.append(gamma_w)
                                raw_atoms.append(atom)
                                r_hat.append(gamma_dist)
                                connect.append(c)
                    else:
                        for gamma_idx, gamma_dist in enumerate(sampler.dist):
                            atoms = sampler.decode(sampler.codes[gamma_idx])
                            if self.context.contain_cardinality_constraint():
                                gamma_w = np.sum(
                                    np.dot(
                                        self.context.top_weights,
                                        (cw * q * gamma_dist * wmc_prod[idx])
                                    )
                                )
                                # NOTE: must be real number
                                gamma_w = gamma_w.real
                            else:
                                gamma_w = np.sum(
                                    cw * q * gamma_dist * wmc_prod[idx])
                            if gamma_w != 0:
                                dist.append(gamma_w)
                                raw_atoms.append(atoms)
                                r_hat.append(gamma_dist)
                    logger.debug('Distribution of each B-type:')
                    for d, v in zip(dist, raw_atoms):
                        logger.debug('%s %s', v, d)
                    # sample
                    sampled_idx = random.choices(
                        range(len(dist)),
                        weights=dist, k=1
                    )[0]
                    if self.context.contain_tree_constraint():
                        if connect[sampled_idx]:
                            tree_sum_context.connect((i, j))
                        else:
                            tree_sum_context.disconnect((i, j))
                    sampled_raw_atoms = raw_atoms[sampled_idx]
                    sampled_prob = r_hat[sampled_idx]
                    # replace to real elements
                    sampled_atoms_replaced = self._replace_consts(
                        sampled_raw_atoms,
                        {a: self.domain[i], b: self.domain[j]}
                    )
                    sampled_atoms = sampled_atoms.union(sampled_atoms_replaced)
                    # update q
                    q *= sampled_prob
                    # move forward
                    idx += 1
                    logger.debug(
                        'sampled atoms at this step: %s', sampled_atoms_replaced
                    )
                    # logger.debug(
                    #     'sampled atoms: %s', sampled_atoms
                    # )
                    logger.debug('updated q: %s', q)
            self.t_sampling_worlds += t.elapsed
        assert idx == len(wmc_prod)
        return self._remove_aux_atoms(sampled_atoms)

    def _remove_aux_atoms(self, atoms):
        preds = self.context.mln.preds()
        return set(
            filter(lambda atom: atom.pred in preds, atoms)
        )

    def _replace_consts(self, atoms, replacement):
        def replace(atom, replacement):
            args = [replacement.get(a) for a in atom.args]
            return atom.pred(*args)
        replaced_atoms = set(
            replace(atom, replacement) for atom in atoms
        )
        return replaced_atoms

    def _get_config_weights(self):
        config = []
        weights = []
        multinomial_coefficients = MultinomialCoefficients(
            self.domain_size, self.n_cells)

        for partition in multinomial_coefficients:
            coef = multinomial_coefficients.coef(partition)
            config_weight = get_config_weight(
                self.context, self.cell_graph, partition).weight
            if self.context.contain_cardinality_constraint():
                # NOTE: must be real numbers
                config_weight = np.sum(
                    np.dot(self.context.top_weights, config_weight))
                weight = config_weight.real if config_weight.real > 0 else 0
            else:
                weight = np.sum(config_weight)

            config.append(partition)
            weights.append(coef * weight)
        return config, weights

    def sample(self, k=1):
        samples = []
        sampled_configs = random.choices(
            self.config, weights=self.weights, k=k)
        self.t_assigning = 0
        self.t_sampling = 0
        self.t_sampling_worlds = 0
        for sampled_config in sampled_configs:
            samples.append(self._sample_on_config(sampled_config))
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
    logger.debug('Samples:')
    for s in samples:
        logger.debug(sorted(str(i) for i in s))
