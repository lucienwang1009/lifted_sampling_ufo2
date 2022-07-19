"""
Logic symbols manipulations
"""
from __future__ import annotations

import typing
from sympy import Symbol
from sympy.logic import boolalg
from sympy.logic import simplify_logic
from typing import Dict

if typing.TYPE_CHECKING:
    from .syntax import Atom, CNF, Formula


atom2sym: Dict[Atom, Symbol] = dict()
sym2atom: Dict[Symbol, Atom] = dict()

symbol = boolalg.Boolean

def get_symbol(atom: Atom) -> Symbol:
    if atom in atom2sym:
        return atom2sym.get(atom)
    s = Symbol(str(atom))
    atom2sym[atom] = s
    sym2atom[s] = atom
    return s

def get_atom(symbol: Symbol) -> Atom:
    if symbol not in sym2atom:
        raise RuntimeError(
            "Symbol %s not found", symbol
        )
    return sym2atom.get(symbol)

def Equivalent(*args):
    return boolalg.Equivalent(*args)

def And(*args):
    return boolalg.And(*args)

def Or(*args):
    return boolalg.Or(*args)

def Not(*args):
    return boolalg.Not(*args)

def to_cnf(symbol: boolalg.Boolean, simplify=True) -> CNF:
    from .syntax import Atom, Lit, DisjunctiveClause, CNF

    def to_internal(symbol: boolalg.Boolean) -> Formula:
        if symbol.is_Atom:
            return get_atom(symbol)
        elif symbol.is_Not:
            return Lit(to_internal(symbol.args[0]), False)
        elif isinstance(symbol, boolalg.Or):
            args = [to_internal(arg) for arg in symbol.args]
            lits = [Lit(arg) if isinstance(arg, Atom)
                    else arg for arg in args]
            return DisjunctiveClause(frozenset(lits))
        elif isinstance(symbol, boolalg.And):
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

    if simplify:
        symbol = simplify_logic(symbol)
    cnf = boolalg.to_cnf(symbol)
    return CNF.from_formula(to_internal(cnf))
