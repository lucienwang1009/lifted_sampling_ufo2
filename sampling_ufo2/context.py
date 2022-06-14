import numpy as np

from logzero import logger
from typing import Callable, Dict, Set, Tuple, List
from dataclasses import dataclass, field
from itertools import product

from sampling_ufo2.network.mln import MLN
from sampling_ufo2.network.constraint import TreeConstraint, CardinalityConstraint
from sampling_ufo2.fol.syntax import AndCNF, CNF, Const, Lit, Pred, QuantifiedFormula, \
    Atom
from sampling_ufo2.fol.syntax import a, b
from sampling_ufo2.fol.utils import new_predicate
from sampling_ufo2.cell_graph import CellGraph


@dataclass(frozen=True)
class ExtBType(object):
    code_ab: Tuple[bool]
    code_ba: Tuple[bool]
    preds: Tuple[Pred]
    uni_var_indices: Tuple[int]
    atoms: Tuple[Tuple[Atom]] = field(
        default=None, repr=False, init=False, hash=False, compare=False)

    def __post_init__(self):
        atoms = list()
        for p, idx in zip(self.preds, self.uni_var_indices):
            if idx == 0:
                args_ab = [a, b]
                args_ba = [b, a]
            elif idx == 1:
                args_ab = [b, a]
                args_ba = [a, b]
            else:
                raise RuntimeError(
                    'Index of universial quantified variable is wrong: {}'.format(
                        idx)
                )
            atoms.append((
                p(*args_ab),
                p(*args_ba)
            ))
        object.__setattr__(self, 'atoms', tuple(atoms))

    def get_conditional_formula(self) -> CNF:
        evidences = set()
        for ab, ba, (atom_ab, atom_ba) in zip(self.code_ab, self.code_ba, self.atoms):
            if (ab):
                evidences.add(CNF.from_lit(Lit(atom_ab)))
            else:
                evidences.add(CNF.from_lit(Lit(atom_ab, False)))
            if (ba):
                evidences.add(CNF.from_lit(Lit(atom_ba)))
            else:
                evidences.add(CNF.from_lit(Lit(atom_ba, False)))
        return AndCNF(*evidences)

    def is_positive(self, pred: Pred) -> bool:
        idx = self.preds.index(pred)
        return self.code_ab[idx], self.code_ba[idx]

    def __str__(self):
        conditional_formula = self.get_conditional_formula()
        return str(conditional_formula)

    def __repr__(self):
        return str(self)


class Context(object):
    def __init__(self, mln: MLN, tree_constraint: TreeConstraint,
                 cardinality_constraint: CardinalityConstraint):
        self.mln: MLN = mln
        self.tree_constraint: TreeConstraint = tree_constraint
        self.cardinality_constraint: CardinalityConstraint = cardinality_constraint
        self.top_weights: np.ndarray
        self.reverse_dft_coef: np.ndarray
        if self.cardinality_constraint is not None:
            self.mln, self.top_weights, self.reverse_dft_coef = \
                self.cardinality_constraint.dft(self.mln)
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
        self.sentence: CNF
        self.weights: Dict[Pred, np.ndarray] = dict()
        self.tseitin_to_formula: Dict[Pred, QuantifiedFormula] = dict()

        # For existantial quantifiers
        # existential quantified predicates,
        # they should be some converted tseitin predicates
        self.ext_preds: List[Pred] = list()
        self.skolem_preds: List[Pred] = list()  # skolem predicates
        self._skolemized_sentence: CNF  # skolemized sentence
        self._skolem_to_tseitin: Dict[Pred, Pred] = dict()
        self._tseitin_to_skolem: Dict[Pred, Pred] = dict()
        self._uni_var_indices: List[int] = list()

        # Convert to FO sentence for WFOMC
        self.build_sentence()
        logger.debug('sentence for WFOMC: %s', self.sentence)
        logger.debug('weights for WFOMC:')
        for pred, w in self.weights.items():
            logger.debug('%s: %s', pred, w)

        # deal with tree axiom
        self.conditional_formulas: List[CNF] = list()
        if self.tree_constraint is not None:
            if self.contain_tree_constraint():
                tree_lit = Lit(self.tree_constraint.pred(a, b))
                self.tree_condition_ab_p: CNF = CNF.from_lit(tree_lit)
                self.tree_condition_ab_n: CNF = CNF.from_lit(tree_lit.negate())
                self.conditional_formulas.extend([
                    self.tree_condition_ab_p,
                    self.tree_condition_ab_n
                ])
            else:
                raise RuntimeError(
                    "Unknown tree constraint: %s", type(self.tree_constraint)
                )

        if self.contain_existential_quantifier():
            logger.info('skolemized sentence for WFOMC: %s',
                        self._skolemized_sentence)
            self.ext_btypes: List[ExtBType] = self._get_ext_btypes()
            for ext_btype in self.ext_btypes:
                self.conditional_formulas.append(
                    ext_btype.get_conditional_formula()
                )
            logger.info('Build cell graph for skolemized sentence')
            self.skolem_cell_graph = CellGraph(
                self._skolemized_sentence,
                self.get_weight_fn(),
            )

        logger.info('Build cell graph')
        self.cell_graph = CellGraph(
            self.sentence, self.get_weight_fn(), self.conditional_formulas)

    def _get_ext_btypes(self) -> List[ExtBType]:
        ext_btypes = list()
        n_ext_preds = len(self.ext_preds)
        for i in product(*([[True, False]] * n_ext_preds)):
            for j in product(*([[True, False]] * n_ext_preds)):
                ext_btypes.append(
                    ExtBType(i, j, tuple(self.ext_preds),
                             tuple(self._uni_var_indices))
                )
        return ext_btypes

    def contain_tree_constraint(self) -> bool:
        return isinstance(self.tree_constraint, TreeConstraint)

    def contain_cardinality_constraint(self) -> bool:
        return isinstance(self.cardinality_constraint, CardinalityConstraint)

    def contain_existential_quantifier(self) -> bool:
        return len(self.ext_preds) > 0

    def get_weight_fn(self, weights: Dict[Pred, np.ndarray] = None) -> Callable[[Pred], Tuple[np.ndarray]]:
        if weights is None:
            weights = self.weights

        def get_weight(pred: Pred) -> Tuple[np.ndarray]:
            default = np.array([1.0], dtype=self.dtype)
            if pred in weights:
                return weights[pred]
            return (default, default)
        return get_weight

    def build_sentence(self):
        sentence = []
        skolemized_sentence = []
        for idx, (formula, weight) in enumerate(self.mln):
            variables = tuple(formula.vars())
            aux_pred = new_predicate(len(variables))
            aux_atom = aux_pred(*variables)
            aux_cnf = CNF.from_atom(aux_atom)
            tseitin_equation = formula.cnf.Equate(aux_cnf)
            sentence.append(tseitin_equation)
            skolemized_sentence.append(tseitin_equation)
            self.tseitin_to_formula[aux_atom] = formula
            if formula.is_exist():
                assert self.mln.is_hard(idx)
                ext_vars, uni_vars = formula.ext_uni_vars()
                assert len(ext_vars) == 1 and len(uni_vars) == 1

                skolem_pred = new_predicate(len(uni_vars), 'skolem')
                skolem_atom = skolem_pred(*uni_vars)
                skolemized_sentence.append(
                    CNF.from_atom(skolem_atom).Or(aux_cnf.Not())
                )
                self.weights[skolem_pred] = (np.array([1.0], dtype=self.dtype),
                                             np.array([-1.0], dtype=self.dtype))

                uni_var = list(uni_vars)[0]
                if uni_var == aux_atom.args[0]:
                    self._uni_var_indices.append(0)
                else:
                    self._uni_var_indices.append(1)
                self.ext_preds.append(aux_pred)
                self.skolem_preds.append(skolem_pred)
                self._skolem_to_tseitin[skolem_pred] = aux_pred
                self._tseitin_to_skolem[aux_pred] = skolem_pred
            else:
                if self.mln.is_hard(idx):
                    sentence.append(aux_cnf)
                    skolemized_sentence.append(aux_cnf)
                else:
                    # set weight for aux predicate
                    self.weights[aux_pred] = (np.array(np.exp(weight), dtype=self.dtype),
                                              np.array([1.0], dtype=self.dtype))
        self.sentence = AndCNF(*sentence)
        self._skolemized_sentence = AndCNF(
            *skolemized_sentence)
