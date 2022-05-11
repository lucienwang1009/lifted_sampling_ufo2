import numpy as np

from logzero import logger
from typing import Dict, Tuple, Any, Set, Callable

from sampling_ufo2.network.mln import MLN
from sampling_ufo2.network.constraint import TreeConstraint, ArborescenceConstraint, CardinalityConstraint
from sampling_ufo2.fol.syntax import Atom, Pred, CNF, Substitution, Const, Lit, Var, AndCNF
from sampling_ufo2.fol.syntax import a, b, c
from sampling_ufo2.fol.utils import new_predicate


class Context(object):
    def __init__(self, mln: MLN, tree_constraint: TreeConstraint,
                 cardinality_constraint: CardinalityConstraint):
        self.mln: MLN = mln
        self.tree_constraint: TreeConstraint = tree_constraint
        self.cardinality_constraint: CardinalityConstraint = cardinality_constraint
        self.top_weights: np.ndarray
        self.reverse_dft_coef: np.ndarray
        if self.cardinality_constraint is not None:
            self.mln, self.top_weights, self.reverse_dft_coef = self.cardinality_constraint.dft(
                self.mln)
            logger.debug(
                'Top weights: %s, reverse dft coeffient: %s',
                self.top_weights, self.reverse_dft_coef
            )
            logger.debug(
                'Complex MLN:\n %s',
                self.mln
            )
        # if it's a complex MLN
        self.dtype = self.mln.dtype
        if self.mln.weight(0).shape:
            self.weight_dims = self.mln.weight(0).shape[0]
        else:
            self.weight_dims = 1

        self.domain: Set[Const] = self.mln.domain
        self.formula: CNF
        self.weights: Dict[Pred, np.ndarray]
        self.formula, self.weights = self.get_sentence()
        logger.debug('formula for WFOMC: %s', self.formula)
        logger.debug('weights for WFOMC:')
        for pred, w in self.weights.items():
            logger.debug('%s: %s', pred, w)

        self.preds: Tuple[Pred] = tuple(self.formula.preds())
        self.vars: Tuple[Var] = tuple(self.formula.vars())

        # ground formula
        gnd_formula_ab1 = self.ground_formula(a, b)
        gnd_formula_ab2 = self.ground_formula(b, a)
        self.gnd_formula_ab: CNF = AndCNF(
            gnd_formula_ab1, gnd_formula_ab2)
        self.gnd_formula_cc: CNF = self.ground_formula(c, c)

        # deal with tree axiom
        if self.tree_constraint is not None:
            if self.contain_tree_constraint():
                tree_lit = Lit(self.tree_constraint.pred(a, b))
                self.gnd_formula_ab_p: CNF = AndCNF(
                    gnd_formula_ab1, gnd_formula_ab2, CNF.from_lit(
                        tree_lit)
                )
                self.gnd_formula_ab_n: CNF = AndCNF(
                    gnd_formula_ab1, gnd_formula_ab2, CNF.from_lit(
                        tree_lit.negate())
                )
            elif self.contain_arborescence_constraint():
                tree_lit1 = Lit(self.tree_constraint.pred(a, b))
                tree_lit2 = Lit(self.tree_constraint.pred(a, b), False)
                tree_lit3 = Lit(self.tree_constraint.pred(b, a))
                tree_lit4 = Lit(self.tree_constraint.pred(b, a), False)
                # WMC(\phi(a,b) ^ R(a, b) ^ !R(b, a))
                self.gnd_formula_ab_p: CNF = AndCNF(
                    gnd_formula_ab1, gnd_formula_ab2, CNF.from_lit(
                        tree_lit1), CNF.from_lit(tree_lit4)
                )
                # WMC(\phi(a,b) ^ !R(a, b) ^ !R(b, a))
                self.gnd_formula_ab_n: CNF = AndCNF(
                    gnd_formula_ab1, gnd_formula_ab2, CNF.from_lit(
                        tree_lit2), CNF.from_lit(tree_lit4)
                )
                # WMC(\phi(a,b) ^ !R(a, b) ^ R(b, a))
                self.gnd_formula_ba_p: CNF = AndCNF(
                    gnd_formula_ab1, gnd_formula_ab2, CNF.from_lit(
                        tree_lit3), CNF.from_lit(tree_lit2)
                )
                # WMC(\phi(a,b) ^ (R(a, b) v R(b, a)))
                self.gnd_formula_ab_or: CNF = AndCNF(
                    gnd_formula_ab1, gnd_formula_ab2, CNF.from_lit(
                        tree_lit1).Or(CNF.from_lit(tree_lit3))
                )
            else:
                raise RuntimeError(
                    "Unknown tree constraint: %s", type(self.tree_constraint)
                )

    def contain_tree_constraint(self) -> bool:
        return not isinstance(self.tree_constraint, ArborescenceConstraint) \
            and isinstance(self.tree_constraint, TreeConstraint)

    def contain_arborescence_constraint(self) -> bool:
        return isinstance(self.tree_constraint, ArborescenceConstraint)

    def contain_cardinality_constraint(self) -> bool:
        return isinstance(self.cardinality_constraint, CardinalityConstraint)

    def ground_formula(self, c1: Const, c2: Const) -> CNF:
        substitution = Substitution(zip(self.vars, [c1, c2]))
        gnd_formula = self.formula.substitute(substitution)
        return gnd_formula

    def get_weight_fn(self) -> Callable[[Pred], Tuple[np.ndarray]]:
        def get_weight(pred: Pred) -> Tuple[np.ndarray]:
            default = np.array(1.0, dtype=self.dtype)
            if pred in self.weights:
                return self.weights[pred]
            return (default, default)
        return get_weight

    def get_sentence(self) -> Tuple[CNF, Dict[Pred, np.ndarray]]:
        weights = dict()
        sentence = []
        for idx, (formula, weight) in enumerate(self.mln):
            if self.mln.is_hard(idx):
                sentence.append(formula.cnf)
            else:
                variables = tuple(formula.vars())
                aux_pred = new_predicate(len(variables))
                # set weight for aux predicate
                weights[aux_pred] = (np.array(np.exp(weight), dtype=self.dtype),
                                     np.array(1.0, dtype=self.dtype))
                aux_atom = aux_pred(*variables)
                sentence.append(formula.cnf.Equate(
                    CNF.from_atom(aux_atom)
                ))
            if formula.is_exist():
                ext_vars = formula.exist.quantified_vars
                uni_vars = formula.vars() - ext_vars
                skolem_pred = new_predicate(len(uni_vars), 'skolem')
                skolem_atom = skolem_pred(*uni_vars)
                if self.mln.is_hard(idx):
                    # replace the hard formula
                    sentence.pop()
                    sentence.append(
                        CNF.from_atom(skolem_atom).Or(formula.cnf.Not())
                    )
                else:
                    sentence.append(
                        CNF.from_atom(skolem_atom).Or(
                            CNF.from_atom(aux_atom).Not()
                        )
                    )
                weights[skolem_pred] = (np.array(1.0, dtype=self.dtype),
                                        np.array(-1.0, dtype=self.dtype))
        return AndCNF(*sentence), weights
