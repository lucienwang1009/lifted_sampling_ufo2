import argparse
import logging
import logzero


from sampling_ufo2.context import WFOMCContext
from sampling_ufo2.parser import parse_mln_constraint
from sampling_ufo2.wfomc import count_distribution


def parse_args():
    parser = argparse.ArgumentParser(
        description='WFOMC for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    mln, tree_constraint, cardinality_constraint = parse_mln_constraint(
        args.input
    )
    print(cardinality_constraint.pred2card)
    context = WFOMCContext(mln, tree_constraint, cardinality_constraint)
    count_dist = count_distribution(
        context.sentence, context.get_weight,
        context.domain, context.mln.preds(),
        context.tree_constraint, context.cardinality_constraint
    )
    print(count_dist)
