from parsimonious import Grammar
from logzero import logger

from sampling_ufo2.parser.parser import MLNVisitor, MLNConstraintVisitor
from sampling_ufo2.parser.grammars import mlnrules, mln_with_constraint_rules


def filter_comments(content):
    res = []
    lines = content.split('\n')
    for line in lines:
        comment_ind = line.find('//')
        if comment_ind == -1:
            res.append(line)
        else:
            res.append(line[:comment_ind])
    return '\n'.join(res)


def parse_mln(input_file):
    with open(input_file, 'r') as f:
        mln_content = f.read()
    mln_content = filter_comments(mln_content)
    grammar = Grammar(mlnrules)
    tree = grammar.parse(mln_content)
    mln = MLNVisitor().visit(tree)
    logger.info('Parsed MLN:\n %s', mln)
    return mln


def parse_mln_constraint(input_file):
    with open(input_file, 'r') as f:
        mln_content = f.read()
    mln_content = filter_comments(mln_content)
    grammar = Grammar(mln_with_constraint_rules)
    mln, tree_constraint, cardinality_constraint = \
        MLNConstraintVisitor().visit(grammar.parse(mln_content))
    logger.info('Parsed MLN: %s', mln)
    logger.info('Parsed tree constraint: %s', tree_constraint)
    logger.info('Parsed cardinality constraints: %s', cardinality_constraint)
    return mln, tree_constraint, cardinality_constraint
