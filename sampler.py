import random
import argparse
import os
import logzero
import logging

from copy import deepcopy
from pracmln import MLN
from logzero import logger
from copy import deepcopy

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
        assert len(self.mln.domains) == 1, "only support one domain for now"
        self.domain = list(self.context.mln.domains.values())[0]
        logger.debug('domain: %s', self.domain)

    def _compute_wmc_prod(self, cell_assignment):
        logger.debug('cell assignment: %s', cell_assignment)
        wmc_prod = []
        wmc_prod.insert(0, [1] * self.context.w_dim)
        # compute from back to front
        for i in range(len(self.domain) - 1, -1, -1):
            for j in range(len(self.domain) - 1, i, -1):
                prod = []
                for k in range(self.context.w_dim):
                    cell_1 = cell_assignment[self.domain[i]]
                    cell_2 = cell_assignment[self.domain[j]]
                    if cell_1 == cell_2:
                        prod.append(wmc_prod[0][k] * cell_1.s[k])
                    else:
                        prod.append(
                            wmc_prod[0][k] * self.cell_graph.r[k][frozenset((cell_1, cell_2))]
                        )
                wmc_prod.insert(0, prod)
        return wmc_prod

    def _sample_with_config(self, config):
        logger.debug('sample from cell configuration %s', config)
        sampled_vars = set()
        cell_assignment = {}
        domain = set(self.domain)
        cw = [1] * self.context.w_dim
        for i, n in enumerate(config):
            elements_sampled = set(random.sample(domain, n))
            for element in elements_sampled:
                cell_assignment[element] = self.cell_graph.cells[i]
                # include the atoms of the cell type
                sampled_vars = sampled_vars.union(
                    self.cell_graph.cells[i].decode(
                        element, include_negative=False
                    )
                )
            domain = domain - elements_sampled
            # compute cw
            for k in range(self.context.w_dim):
                cw[k] *= (self.cell_graph.cells[i].w[k] ** n)
        # compute wmc_prod
        wmc_prod = self._compute_wmc_prod(cell_assignment)
        logger.debug('wmc prod: %s', wmc_prod)
        logger.debug('cw: %s', cw)

        logger.debug('initial atoms: %s', sampled_vars)
        q = [1] * self.context.w_dim
        idx = 0
        for i, x_i in enumerate(self.domain):
            for j, x_j in enumerate(self.domain):
                if i >= j:
                    continue
                logger.debug('sample the atom consisting of %s and %s', x_i, x_j)
                sampler = None
                if cell_assignment[x_i] == cell_assignment[x_j]:
                    cell = cell_assignment[x_i]
                    sampler = cell.s_sampler
                elif cell_assignment[x_i] != cell_assignment[x_j]:
                    cell_1 = cell_assignment[x_i]
                    cell_2 = cell_assignment[x_j]
                    sampler = self.cell_graph.r_samplers[frozenset((cell_1, cell_2))]
                dist = []
                for gamma_idx in range(len(sampler.dist[0])):
                    gamma_w = 0
                    for k in range(self.context.w_dim):
                        gamma_w += (
                            cw[k] * q[k] * sampler.dist[k][gamma_idx] * wmc_prod[idx][k]
                        )
                    dist.append(gamma_w.real)
                # sample
                sampled_vars_raw = sampler.decode(
                    random.choices(sampler.codes, dist, k=1)[0]
                )
                # replace to real elements
                sampled_vars_replaced = self._replace_consts(
                    sampled_vars_raw,
                    {'a': x_i, 'b': x_j}
                )
                sampled_vars = sampled_vars.union(sampled_vars_replaced)
                # update q
                for k in range(self.context.w_dim):
                    for var in sampler.unknown_vars:
                        if var in sampled_vars_raw:
                            q[k] *= sampler.wmc.var_weights[var][k]
                        else:
                            q[k] *= sampler.wmc.var_weights[var.negate()][k]
                # move forward
                idx += 1
                logger.debug(
                    'sampled atoms: %s', sampled_vars_replaced
                )
                logger.debug('updated q: %s', q)
        return sampled_vars

        # for cell_1 in self.cell_graph.cells:
        #     n_1 = len(cell_domain[cell_1])
        #     if n_1 == 0:
        #         continue
        #     # sample by s
        #     s_samples = cell_1.s_sampler.sample(
        #         n_1 * (n_1 - 1)
        #     )
        #     cnt = 0
        #     for c_1 in cell_domain[cell_1]:
        #         sampled_vars = sampled_vars.union(
        #             cell_1.decode(c_1, include_negative=False)
        #         )
        #         for c_2 in cell_domain[cell_1]:
        #             if c_1 == c_2:
        #                 sampled_vars = sampled_vars.union(
        #                     set(self._replace_consts(
        #                         cell_1.w_sampler.sample(1)[0],
        #                         {'c': c_1}
        #                     ))
        #                 )
        #                 continue
        #             sampled_vars = sampled_vars.union(
        #                 set(self._replace_consts(
        #                     s_samples[cnt],
        #                     {'a': c_1, 'b': c_2}
        #                 ))
        #             )
        #             cnt += 1
        #     assert cnt == len(s_samples)

        #     # sample by r
        #     for cell_2 in self.cell_graph.cells:
        #         cnt = 0
        #         n_2 = len(cell_domain[cell_2])
        #         if cell_1 == cell_2 or n_2 == 0:
        #             continue
        #         r_samples = self.cell_graph.r_samplers[cell_1][cell_2].sample(
        #             n_1 * n_2
        #         )
        #         for c_1 in cell_domain[cell_1]:
        #             for c_2 in cell_domain[cell_2]:
        #                 sampled_vars = sampled_vars.union(
        #                     set(self._replace_consts(
        #                         r_samples[cnt],
        #                         {'a': c_1, 'b': c_2}
        #                     ))
        #                 )
        #                 cnt += 1
        #         assert cnt == len(r_samples)
        # logger.debug(sampled_vars)
        # return sampled_vars

    def _replace_consts(self, vars, assign):
        replaced_vars = [
            Atom.from_var(v).replace(assign).to_var() for v in vars
        ]
        return replaced_vars

    def sample(self, k=1):
        samples = []
        sampled_configs = random.choices(self.config, weights=self.weights, k=k)
        for sampled_config in sampled_configs:
            samples.append(self._sample_with_config(sampled_config))
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
                config_wfomc += (coef * product_wmc(self.cell_graph, partition, k))
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
    sampler = Sampler(mln)
    samples = sampler.sample(args.n_samples)
    logger.info('Samples:')
    for s in samples:
        logger.info(sorted(s))
