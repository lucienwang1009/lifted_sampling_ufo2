from __future__ import annotations

import os

from collections import defaultdict
from typing import FrozenSet, List
from nnf import Or, And, dsharp, NNF
from nnf import Var as nnf_var

from .syntax import CNF, Const, DisjunctiveClause, Lit, Pred, Substitution, Var
from .syntax import x, y, z


PREDICATES = defaultdict(list)

dsharp_exe = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'external',
    'dsharp'
)


def new_predicate(arity: int, name: str) -> Pred:
    global PREDICATES
    p = Pred('{}{}'.format(name, len(PREDICATES[name])), arity)
    PREDICATES[name].append(p)
    return p


def get_predicates(name: str) -> List[Pred]:
    global CNT_PREDICATES
    return PREDICATES[name]


def to_nnf_var(lit: Lit) -> Var:
    return nnf_var(str(lit.atom)) if lit.positive else nnf_var(str(lit.atom)).negate()


def to_nnf(formula: CNF) -> NNF:
    clauses = []
    for disjuction in formula.clauses:
        clause = []
        for lit in disjuction.literals:
            clause.append(to_nnf_var(lit))
        clauses.append(Or(clause))
    cnf = And(clauses)
    nnf = dsharp.compile(cnf, executable=dsharp_exe, smooth=True)
    return nnf


def pad_vars(vars: FrozenSet[Var], arity: int) -> FrozenSet[Var]:
    if arity > 3:
        raise RuntimeError(
            "Not support arity > 3"
        )
    ret_vars = set(vars)
    default_vars = [x, y, z]
    idx = 0
    while(len(ret_vars) < arity):
        ret_vars.add(default_vars[idx])
        idx += 1
    return frozenset(list(ret_vars)[:arity])


def ground_FO2(sentence: CNF, c1: Const, c2: Const = None) -> CNF:
    variables = sentence.vars()
    if len(variables) > 2 or len(variables) < 1:
        raise RuntimeError(
            "Can only ground out FO2"
        )
    if len(variables) == 1:
        constants = [c1]
    else:
        if c2 is not None:
            constants = [c1, c2]
        else:
            constants = [c1, c1]
    substitution = Substitution(zip(variables, constants))
    gnd_formula = sentence.substitute(substitution)
    return gnd_formula


def exact_one_of(preds: List[Pred]) -> CNF:
    lits = [Lit(p(x)) for p in preds]
    # p1(x) v p2(x) v ... v pm(x)
    clauses = [DisjunctiveClause(frozenset(lits))]
    for i, l1 in enumerate(lits):
        for j, l2 in enumerate(lits):
            if i < j:
                clauses.append(
                    DisjunctiveClause(
                        frozenset([l1.negate(), l2.negate()])
                    )
                )
    return CNF(frozenset(clauses))
