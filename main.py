import argparse
import os
import logging
import logzero
import pickle
import numpy as np

from logzero import logger
from contexttimer import Timer
from pracmln import MLN

from wfomc.cnf import CNFManager
from wfomc.compiler import DsharpCompiler

example_usage = '''Example:
python main.py -d person -p 'smokes(person);friends(person,person)' \\
    -f 'smokes(x);smokes(x) ^ friends(x,y) => smokes(y)' -s 2
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sampling from MLN',
        epilog=example_usage,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


def sampling(mln):
    # convert mln to the sentence used by WFOMC
    cnf_manager = CNFManager(mln)
    ddnnf = DsharpCompiler.compile(cnf_manager)


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
    sampling(mln)
