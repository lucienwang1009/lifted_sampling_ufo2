from __future__ import annotations

import typing
from sympy import Symbol
from sympy.logic.boolalg import Boolean, Or, And, Equivalent, to_cnf, Not
from typing import Dict

if typing.TYPE_CHECKING:
    from .syntax import Atom, CNF, Formula


class SyntaxBackend(object):
    pass


class SympyBackend(SyntaxBackend):
    atom2sym: Dict[Atom, Symbol] = dict()
    sym2atom: Dict[Symbol, Atom] = dict()

    symbol = Boolean

    @staticmethod
    def get_symbol(atom: Atom) -> Symbol:
        if atom in SympyBackend.atom2sym:
            return SympyBackend.atom2sym.get(atom)
        s = Symbol(str(atom))
        SympyBackend.atom2sym[atom] = s
        SympyBackend.sym2atom[s] = atom
        return s

    @staticmethod
    def get_atom(symbol: Symbol) -> Atom:
        if symbol not in SympyBackend.sym2atom:
            raise RuntimeError(
                "Symbol %s not found", symbol
            )
        return SympyBackend.sym2atom.get(symbol)

    @staticmethod
    def Equivalent(*args):
        return Equivalent(*args)

    @staticmethod
    def And(*args):
        return And(*args)

    @staticmethod
    def Or(*args):
        return Or(*args)

    @staticmethod
    def Not(*args):
        return Not(*args)

    @staticmethod
    def to_cnf(symbol: Boolean) -> CNF:
        from .syntax import Atom, Lit, DisjunctiveClause, CNF

        def to_internal(symbol: Boolean) -> Formula:
            if symbol.is_Atom:
                return SympyBackend.get_atom(symbol)
            elif symbol.is_Not:
                return Lit(to_internal(symbol.args[0]), False)
            elif isinstance(symbol, Or):
                args = [to_internal(arg) for arg in symbol.args]
                lits = [Lit(arg) if isinstance(arg, Atom)
                        else arg for arg in args]
                return DisjunctiveClause(frozenset(lits))
            elif isinstance(symbol, And):
                args = [to_internal(arg) for arg in symbol.args]
                clauses = []
                for arg in args:
                    if isinstance(arg, Atom):
                        clauses.append(DisjunctiveClause(
                            frozenset([Lit(arg)])))
                    elif isinstance(arg, Lit):
                        clauses.append(DisjunctiveClause(frozenset([arg])))
                    else:
                        clauses.append(arg)
                return CNF(frozenset(clauses))

        cnf = to_cnf(symbol)
        return CNF.from_formula(to_internal(cnf))
