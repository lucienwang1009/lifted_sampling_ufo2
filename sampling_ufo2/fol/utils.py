from __future__ import annotations

import os

from typing import List, Tuple, Dict, FrozenSet
from itertools import product
from nnf import Or, And, dsharp, NNF
from nnf import Var as nnf_var

from .syntax import Atom, Lit, DisjunctiveClause, ConjunctiveClause, Pred, Var, CNF
from .syntax import x, y, z
from .backend import SympyBackend as backend


auxiliary_pred_name = 'aux'
n_auxiliary_preds = 0

dsharp_exe = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'external',
    'dsharp'
)


def new_predicate(arity: int, name: str = auxiliary_pred_name) -> Pred:
    global n_auxiliary_preds
    p = Pred('{}{}'.format(name, n_auxiliary_preds), arity)
    n_auxiliary_preds += 1
    return p


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
