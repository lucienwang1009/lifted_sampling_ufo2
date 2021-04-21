import random
import argparse
import os
import logzero
import logging

from copy import deepcopy
from pracmln import MLN
from logzero import logger

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
        self.domain = set(*self.context.mln.domains.values())

    def _sample_with_config(self, config):
        sampled_vars = set()
        cell_domain = {}
        domain = self.domain
        for i, n in enumerate(config):
            cell_domain[self.cell_graph.cells[i]] = set(random.sample(domain, n))
            domain = domain - cell_domain[self.cell_graph.cells[i]]
        logger.debug(cell_domain)
        for cell_1 in self.cell_graph.cells:
            n_1 = len(cell_domain[cell_1])
            if n_1 == 0:
                continue
            # sample by s
            s_samples = cell_1.s_sampler.sample(
                n_1 * (n_1 - 1)
            )
            cnt = 0
            for c_1 in cell_domain[cell_1]:
                sampled_vars = sampled_vars.union(
                    cell_1.decode(c_1, include_negative=False)
                )
                for c_2 in cell_domain[cell_1]:
                    if c_1 == c_2:
                        sampled_vars = sampled_vars.union(
                            set(self._replace_consts(
                                cell_1.w_sampler.sample(1)[0],
                                {'c': c_1}
                            ))
                        )
                        continue
                    sampled_vars = sampled_vars.union(
                        set(self._replace_consts(
                            s_samples[cnt],
                            {'a': c_1, 'b': c_2}
                        ))
                    )
                    cnt += 1
            assert cnt == len(s_samples)

            # sample by r
            for cell_2 in self.cell_graph.cells:
                cnt = 0
                n_2 = len(cell_domain[cell_2])
                if cell_1 == cell_2 or n_2 == 0:
                    continue
                r_samples = self.cell_graph.r_samplers[cell_1][cell_2].sample(
                    n_1 * n_2
                )
                for c_1 in cell_domain[cell_1]:
                    for c_2 in cell_domain[cell_2]:
                        sampled_vars = sampled_vars.union(
                            set(self._replace_consts(
                                r_samples[cnt],
                                {'a': c_1, 'b': c_2}
                            ))
                        )
                        cnt += 1
                assert cnt == len(r_samples)
        logger.debug(sampled_vars)
        return sampled_vars

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
            for d in range(self.context.w_dim):
                config_wfomc += (coef * product_wmc(self.cell_graph, partition))
            config_wfomc *= coef
            config.append(partition)
            weights.append(config_wfomc)
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
        logger.info(s)
