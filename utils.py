import os

from pracmln.logic.fol import Implication, Exist, Lit, Biimplication
from logzero import logger
from nnf import Var, Or, And, dsharp
from pracmln import Database

from atom import Atom


dsharp_exe = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'dsharp'
)


def to_nnf(formula):
    clauses = []
    cnf = formula.cnf()
    for disjuction in cnf.children:
        clause = []
        if isinstance(disjuction, Lit):
            lits = [disjuction]
        else:
            lits = disjuction.children
        for lit in lits:
            atom = Atom.from_literal(lit)
            var = Var(str(atom))
            if lit.negated:
                clause.append(var.negate())
            else:
                clause.append(var)
        clauses.append(Or(clause))
    cnf = And(clauses)
    nnf = dsharp.compile(cnf, executable=dsharp_exe, smooth=True)
    return nnf
