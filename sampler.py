import random
import argparse
import os
import logzero
import logging

from copy import deepcopy
from pracmln import MLN
from logzero import logger
from copy import deepcopy
from contexttimer import Timer

from context import Context
from cell_graph import CellGraph
from sympy import ntheory
from wfomc import product_wmc
from atom import Atom


class Sampler(object):
    def __init__(self, mln):
        self.mln = deepcopy(mln)
        self.context = Context(self.mln)
        self.cell_graph = CellGraph(self.context)
        self.config, self.weights = self._get_config_weights()
        logger.info('wfomc:%s', sum(self.weights))
        # logger.info(self.context.preds)
        # logger.info(self.cell_graph.cells)
        # logger.info(list(zip(self.config, self.weights)))
        # t = 0
        # for i, c in enumerate(self.config):
        #     for idx in [1, 3, 5, 7]:
        #         if c[idx] == 1:
        #             t += (self.weights[i] / 2)
        #         elif c[idx] == 2:
        #             t += self.weights[i]
        # logger.info(t)
        assert len(self.mln.domains) == 1, "only support one domain for now"
        self.domain = list(self.context.mln.domains.values())[0]
        logger.debug('domain: %s', self.domain)
        # for measuring performance
        self.t_sampling = 0
        self.t_assigning = 0
        self.t_sampling_worlds = 0

    def _compute_wmc_prod(self, cell_assignment):
        logger.debug('cell assignment: %s', cell_assignment)
        wmc_prod = []
        # compute from back to front
        for i in range(len(self.domain) - 1, -1, -1):
            for j in range(len(self.domain) - 1, i, -1):
                cell_1 = cell_assignment[self.domain[i]]
                cell_2 = cell_assignment[self.domain[j]]
                if not wmc_prod:
                    wmc_prod.insert(0, [1] * self.context.w_dim)
                    continue
                prod = []
                for k in range(self.context.w_dim):
                    prod.append(
                        wmc_prod[0][k] * self.cell_graph.edge_weight[k][frozenset((cell_1, cell_2))]
                    )
                wmc_prod.insert(0, prod)
        return wmc_prod

    def _sample_from_config(self, config):
        with Timer() as t:
            logger.debug('sample from cell configuration %s', config)
            sampled_vars = set()
            cell_assignment = {}
            # assign each element a cell type
            random.shuffle(self.domain)
            cw = [1] * self.context.w_dim
            idx = 0
            for i, n in enumerate(config):
                for j in range(n):
                    cell_assignment[self.domain[idx]] = self.cell_graph.cells[i]
                    # include the atoms of the cell type
                    sampled_vars = sampled_vars.union(
                        self.cell_graph.cells[i].decode(
                            self.domain[idx], include_negative=False
                        )
                    )
                    idx += 1
                # compute cw
                for k in range(self.context.w_dim):
                    cw[k] *= (self.cell_graph.cells[i].inherent_weight[k] ** n)
            # compute wmc_prod
            wmc_prod = self._compute_wmc_prod(cell_assignment)
            logger.debug('wmc prod: %s', wmc_prod)
            logger.debug('cw: %s', cw)
            self.t_assigning += t.elapsed

        with Timer() as t:
            logger.debug('initial atoms: %s', sampled_vars)
            q = [1] * self.context.w_dim
            idx = 0
            for i, x_i in enumerate(self.domain):
                for j, x_j in enumerate(self.domain):
                    if i >= j:
                        continue
                    logger.debug('sample the atom consisting of %s and %s', x_i, x_j)
                    sampler = None
                    cell_1 = cell_assignment[x_i]
                    cell_2 = cell_assignment[x_j]
                    sampler = self.cell_graph.samplers[frozenset((cell_1, cell_2))]
                    dist = []
                    for gamma_idx in range(len(sampler.dist)):
                        gamma_w = 0
                        for k in range(self.context.w_dim):
                            gamma_w += (
                                cw[k] * q[k] * sampler.dist[gamma_idx][k] * wmc_prod[idx][k]
                            )
                        dist.append(gamma_w.real)
                    # dist = [d[0].real for d in sampler.dist]
                    logger.debug('unknown vars: %s', sampler.unknown_vars)
                    logger.debug('dist: ')
                    for d, c in zip(dist, [sampler.decode(c) for c in sampler.codes]):
                        logger.debug('%s %s', c, d)
                    logger.debug('weight: %s', cw[0] * sum(dist) * q[0] * wmc_prod[idx][0])
                    # sample
                    sampled_code = random.choices(sampler.codes, weights=dist, k=1)[0]
                    sampled_vars_raw = sampler.decode(sampled_code)
                    # replace to real elements
                    sampled_vars_replaced = self._replace_consts(
                        sampled_vars_raw,
                        {'a': x_i, 'b': x_j}
                    )
                    sampled_vars = sampled_vars.union(sampled_vars_replaced)
                    # update q
                    for k in range(self.context.w_dim):
                        q[k] *= sampler.dist[sampled_code][k]
                    # move forward
                    idx += 1
                    logger.debug(
                        'sampled atoms: %s', sampled_vars_replaced
                    )
                    logger.debug('updated q: %s', q)
            self.t_sampling_worlds += t.elapsed
        assert idx == len(wmc_prod)
        return sampled_vars

    def _replace_consts(self, vars, assign):
        replaced_vars = [
            Atom.from_var(v).replace(assign).to_var() for v in vars
        ]
        return replaced_vars

    def sample(self, k=1):
        samples = []
        with Timer() as t:
            sampled_configs = random.choices(self.config, weights=self.weights, k=k)
            logger.info('elapsed time for sampling cell configurations: %s', t.elapsed)
        self.t_assigning = 0
        self.t_sampling = 0
        self.t_sampling_worlds = 0
        for sampled_config in sampled_configs:
            samples.append(self._sample_from_config(sampled_config))
        self.t_sampling = self.t_assigning + self.t_sampling_worlds
        logger.info('total sampling time: %s', self.t_sampling)
        logger.info('elapsed time for assigning cell type: %s', self.t_assigning)
        logger.info('elapsed time for sampling possible worlds: %s', self.t_sampling_worlds)
        return samples

    def _get_config_weights(self):
        config = []
        weights = []
        domain_size = len(list(self.mln.domains.values())[0])
        n_cells = len(self.cell_graph.cells)
        iterator = ntheory.multinomial.multinomial_coefficients_iterator(
            n_cells, domain_size
        )
        for partition, coef in iterator:
            config_wfomc = 0
            for k in range(self.context.w_dim):
                config_wfomc += product_wmc(self.cell_graph, partition, k)
            config_wfomc *= coef
            config.append(partition)
            weights.append(config_wfomc.real)
        return config, weights


def parse_args():
    parser = argparse.ArgumentParser(
        description='WFOMC for MLN',
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

    mln = MLN.load(args.input, grammar='StandardGrammar')
    with Timer() as t:
        sampler = Sampler(mln)
    logger.info('elapsed time for initializing sampler: %s', t.elapsed)
    samples = sampler.sample(args.n_samples)
    logger.info('Samples:')
    for s in samples:
        logger.info(sorted(s))
