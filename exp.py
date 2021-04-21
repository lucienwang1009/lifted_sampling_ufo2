import argparse
import math
import logging
import logzero

from copy import deepcopy
from pracmln import MLN
from nnf import Var
from collections import Counter
from logzero import logger

from sampler import Sampler
from partition_func_solver import WFOMCSolver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--marginal', '-m', type=str, nargs='+')
    parser.add_argument('--n_worlds', '-k', type=int, default=10000)
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main(args):
    mln = MLN.load(args.input, grammar='StandardGrammar')
    sampler = Sampler(mln)
    worlds = sampler.sample(args.n_worlds)
    c = Counter()
    for w in worlds:
        c.update(w)
    logger.info('counter: %s', c)
    with WFOMCSolver() as s:
        ln_Z = s.solve(mln)
        for m in args.marginal:
            v = Var(m)
            logger.info('Sampled P for %s: %s', m, c[v] / args.n_worlds)
            tmp_mln = deepcopy(mln)
            tmp_mln.formula(m, float('inf'))
            tmp_ln_Z = s.solve(tmp_mln)
            logger.info('Exact P for %s: %s', m, math.exp(tmp_ln_Z - ln_Z))


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    main(args)
