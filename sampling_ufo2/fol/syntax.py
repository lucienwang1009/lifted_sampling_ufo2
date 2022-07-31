from __future__ import annotations


from dataclasses import dataclass, field
from typing import FrozenSet, List, Tuple, Iterable
from collections import OrderedDict
from pysat.solvers import Solver

from . import backend


class Term(object):
    """
    First-order logic terms, including constants and variables
    """

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Formula(object):
    """
    Formula abstract class
    """
    symbol: backend.symbol = field(
        default=None, repr=False, init=False, hash=False, compare=False)


@dataclass(frozen=True)
class Pred:
    """
    Predicate
    """
    name: str
    arity: int

    def __post_init__(self):
        if self.arity < 0:
            raise RuntimeError("Arity must be a natural number")

    def __call__(self, *args: Term):
        # NOTE(hack): the callable obj cannot be the column of dataframe
        if len(args) == 0 or not isinstance(args[0], (Var, Const)):
            return self
        if self.arity != len(args):
            raise RuntimeError(
                "Mismatching number of arguments and predicate %s: %s != %s", str(self), self.arity, len(args))
        return Atom(pred=self, args=tuple(args))

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Var(Term):
    """
    Variable
    """
    name: str

    def substitute(self, substitution: Substitution) -> Const:
        if self in substitution:
            return substitution[self]
        return self


@dataclass(frozen=True)
class Const(Term):
    """
    Constant
    """
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Atom(Formula):
    pred: Pred
    args: Tuple[Term]

    def __post_init__(self):
        if len(self.args) != self.pred.arity:
            raise RuntimeError(
                "Number of terms does not match the predicate's arity")
        object.__setattr__(self, 'symbol', backend.get_symbol(self))

    def __str__(self):
        return '{}({})'.format(self.pred,
                               ','.join([str(arg) for arg in self.args]))

    def __repr__(self):
        return str(self)

    def vars(self) -> FrozenSet[Var]:
        return frozenset(filter(lambda x: isinstance(x, Var), self.args))

    def consts(self) -> FrozenSet[Const]:
        return frozenset(filter(lambda x: isinstance(x, Const), self.args))

    def substitute(self, substitution: Substitution) -> Atom:
        substituted_args = []
        for arg in self.args:
            substituted_args.append(arg.substitute(substitution))
        return Atom(self.pred, tuple(substituted_args))


@dataclass(frozen=True)
class Lit(Formula):
    atom: Atom
    positive: bool = True

    def __post_init__(self):
        atom_symbol = self.atom.symbol
        if not self.positive:
            atom_symbol = ~atom_symbol
        object.__setattr__(self, 'symbol', atom_symbol)

    def __str__(self):
        return '{}{}'.format(
            '' if self.positive else '!',
            self.atom
        )

    def __repr__(self):
        return str(self)

    def negate(self) -> Lit:
        return Lit(self.atom, not self.positive)

    def vars(self) -> FrozenSet[Var]:
        return self.atom.vars()

    def consts(self) -> FrozenSet[Const]:
        return self.atom.consts()

    def substitute(self, substitution: Substitution) -> Lit:
        return Lit(self.atom.substitute(substitution), self.positive)

    def pred(self) -> Pred:
        return self.atom.pred


@dataclass(frozen=True)
class Clause(Formula):
    literals: FrozenSet[Lit]

    def vars(self) -> FrozenSet[Var]:
        vs = set()
        for lit in self.literals:
            vs.update(lit.vars())
        return frozenset(vs)

    def consts(self) -> FrozenSet[Const]:
        cs = set()
        for lit in self.literals:
            cs.update(lit.consts())
        return frozenset(cs)

    def atoms(self) -> FrozenSet[Atom]:
        return frozenset(
            lit.atom for lit in self.literals
        )

    def substitute(self, substitution: Substitution) -> Clause:
        substituted_literals = set()
        for lit in self.literals:
            substituted_literals.add(lit.substitute(substitution))
        cls = type(self)
        return cls(frozenset(substituted_literals))

    def preds(self) -> FrozenSet[Pred]:
        return frozenset(
            lit.pred() for lit in self.literals
        )


@dataclass(frozen=True)
class ConjunctiveClause(Clause):

    def __post_init__(self):
        lit_symbols = [lit.symbol for lit in self.literals]
        object.__setattr__(self, 'symbol', backend.And(*lit_symbols))

    def __str__(self):
        return ' ^ '.join(
            [str(lit) for lit in self.literals]
        )

    def negate(self) -> DisjunctiveClause:
        return DisjunctiveClause(
            frozenset(lit.negate() for lit in self.literals)
        )

    def __invert__(self) -> DisjunctiveClause:
        return self.negate()


@dataclass(frozen=True)
class DisjunctiveClause(Clause):

    def __post_init__(self):
        lit_symbols = [lit.symbol for lit in self.literals]
        object.__setattr__(self, 'symbol', backend.Or(*lit_symbols))

    def __str__(self):
        return ' v '.join(
            [str(lit) for lit in self.literals]
        )

    def negate(self) -> ConjunctiveClause:
        return ConjunctiveClause(
            frozenset(lit.negate() for lit in self.literals)
        )

    def __invert__(self):
        return self.negate()

    def __and__(self, other):
        pass

    def __or__(self, other):
        return DisjunctiveClause(self.literals.union(other.literals))


@dataclass(frozen=True)
class CNF():
    clauses: FrozenSet[DisjunctiveClause]

    def __post_init__(self):
        clause_symbols = [clause.symbol for clause in self.clauses]
        object.__setattr__(self, 'symbol', backend.And(*clause_symbols))

    def vars(self) -> FrozenSet[Var]:
        return self._gather('vars')

    def consts(self) -> FrozenSet[Const]:
        return self._gather('consts')

    def atoms(self) -> FrozenSet[Atom]:
        return self._gather('atoms')

    def substitute(self, substitution: Substitution) -> Clause:
        clauses = [clause.substitute(substitution) for clause in self.clauses]
        return CNF(frozenset(clauses))

    def preds(self) -> FrozenSet[Pred]:
        return self._gather('preds')

    def Not(self) -> CNF:
        new_symbol = backend.Not(self.symbol)
        return backend.to_cnf(new_symbol)

    def Equate(self, other: CNF) -> CNF:
        new_symbol = backend.Equivalent(self.symbol, other.symbol)
        return backend.to_cnf(new_symbol)

    def Or(self, other: CNF) -> CNF:
        new_symbol = backend.Or(self.symbol, other.symbol)
        return backend.to_cnf(new_symbol)

    def And(self, other: CNF) -> CNF:
        new_symbol = backend.And(self.symbol, other.symbol)
        return backend.to_cnf(new_symbol)

    def _gather(self, func_name, **func_kwargs):
        res = set()
        for clause in self.clauses:
            res.update(getattr(clause, func_name)(**func_kwargs))
        return frozenset(res)

    @classmethod
    def from_formula(cls, formula: Formula) -> CNF:
        if isinstance(formula, Atom):
            return cls.from_atom(formula)
        elif isinstance(formula, Lit):
            return cls.from_lit(formula)
        elif isinstance(formula, DisjunctiveClause):
            return cls.from_clause(formula)
        elif isinstance(formula, cls):
            return formula
        else:
            raise RuntimeError(
                'Unsupported type: %s', type(formula)
            )

    @classmethod
    def from_pred(cls, pred: Pred, variables: List[Term]) -> CNF:
        assert pred.arity <= len(variables)
        return cls.from_atom(pred(*variables[:pred.arity]))

    @classmethod
    def from_atom(cls, atom: Atom) -> CNF:
        return cls.from_lit(Lit(atom))

    @classmethod
    def from_lit(cls, lit: Lit) -> CNF:
        return cls.from_clause(DisjunctiveClause(
            frozenset([lit])
        ))

    @classmethod
    def from_clause(cls, clause: DisjunctiveClause) -> CNF:
        return cls(frozenset([clause]))

    def __str__(self):
        return ' ^ '.join(
            ['({})'.format(str(clause)) for clause in self.clauses]
        )

    def __repr__(self):
        return str(self)

    def _encode_Dimacs(self):
        decode = dict(enumerate(self.atoms(), start=1))
        encode = {v: k for k, v in decode.items()}

        clauses = [
            [encode[lit.atom] if lit.positive else -encode[lit.atom]
             for lit in clause.literals]
            for clause in self.clauses
        ]
        return clauses, decode

    def _solver_for(self):
        clauses, decode = self._encode_Dimacs()
        solver = Solver(bootstrap_with=clauses)
        return solver, decode

    def models(self) -> Iterable[FrozenSet[Lit]]:
        """
        Yield all models of the formula

        :rtype Iterable[FrozenSet[Lit]]: models
        """
        solver, decode = self._solver_for()
        with solver:
            if not solver.solve():
                return
            for model in solver.enum_models():
                yield frozenset(
                    [Lit(decode[abs(num)], num > 0) for num in model]
                )


@dataclass(frozen=True)
class Exist():
    quantified_vars: FrozenSet[Var]

    def __str__(self):
        return 'Exist {}'.format(
            ','.join(str(v) for v in self.quantified_vars)
        )

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class QuantifiedFormula(Formula):
    cnf: CNF
    exist: Exist = None

    def __post_init__(self):
        object.__setattr__(self, 'symbol', self.cnf.symbol)

    def vars(self) -> FrozenSet[Var]:
        return self.cnf.vars()

    def ext_uni_vars(self) -> Tuple[FrozenSet[Var], FrozenSet[Var]]:
        all_vars = self.vars()
        if self.exist is None:
            ext_vars = None
        else:
            ext_vars = self.exist.quantified_vars
        return (ext_vars, all_vars - ext_vars)

    def consts(self) -> FrozenSet[Const]:
        return self.cnf.consts()

    def atoms(self) -> FrozenSet[Atom]:
        return self.cnf.atoms()

    def substitute(self, substitution: Substitution) -> Clause:
        return QuantifiedFormula(self.cnf.substitute(substitution))

    def preds(self) -> FrozenSet[Pred]:
        return self.cnf.preds()

    def is_exist(self) -> bool:
        return self.exist is not None

    def Equate(self, other: QuantifiedFormula) -> QuantifiedFormula:
        raise RuntimeError(
            'Not support conjunct two formula'
        )

    @property
    def clauses(self) -> FrozenSet[DisjunctiveClause]:
        return self.cnf.clauses

    @classmethod
    def from_pred(cls, pred: Pred, variables: List[Term]) -> QuantifiedFormula:
        return cls(CNF.from_pred(pred, variables))

    @classmethod
    def from_atom(cls, atom: Atom) -> QuantifiedFormula:
        return cls(CNF.from_atom(atom))

    @classmethod
    def from_lit(cls, lit: Lit) -> QuantifiedFormula:
        return cls(CNF.from_lit(lit))

    def __str__(self):
        return '{}{}'.format(
            str(self.exist) + ' ' if self.exist else '',
            self.cnf
        )

    def __repr__(self):
        return str(self)


def AndQuantifiedFormula(*formulas: List[QuantifiedFormula]) -> QuantifiedFormula:
    for formula in formulas:
        if formula.is_exist():
            raise RuntimeError(
                'Cannot conjunct existentially quantified formula'
            )
    return QuantifiedFormula(AndCNF(*[f.cnf for f in formulas]))


def AndCNF(*cnfs: List[CNF]) -> CNF:
    symbols = [cnf.symbol for cnf in cnfs]
    and_symbols = backend.And(*symbols)
    cnf = backend.to_cnf(and_symbols)
    return cnf


class Substitution(OrderedDict):
    pass


tautology = CNF(frozenset([ConjunctiveClause(frozenset([]))]))
x, y, z = Var('x'), Var('y'), Var('z')
a, b, c = Const('a'), Const('b'), Const('c')
