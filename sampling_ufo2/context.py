import math

from logzero import logger
from typing import Callable, Dict, Set, Tuple, List, FrozenSet
from dataclasses import dataclass, field
from itertools import product

from sampling_ufo2.network.mln import MLN
from sampling_ufo2.network.constraint import TreeConstraint, CardinalityConstraint
from sampling_ufo2.fol.syntax import AndCNF, CNF, Const, Lit, Pred, Atom, DisjunctiveClause, a, b, x
from sampling_ufo2.fol.utils import new_predicate, exact_one_of
from sampling_ufo2.utils import AUXILIARY_PRED_NAME, TSEITIN_PRED_NAME, \
    SKOLEM_PRED_NAME, EVIDOM_PRED_NAME, Rational


@dataclass(frozen=True)
class EBType(object):
    code_ab: Tuple[bool]
    code_ba: Tuple[bool]
    preds: Tuple[Pred]
    uni_var_indices: Tuple[int]
    atoms: Tuple[Tuple[Atom]] = field(
        default=None, repr=False, init=False, hash=False, compare=False
    )

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

    def get_evidences(self) -> FrozenSet[Lit]:
        evidences = set()
        for ab, ba, (atom_ab, atom_ba) in zip(self.code_ab, self.code_ba, self.atoms):
            if (ab):
                evidences.add(Lit(atom_ab))
            else:
                evidences.add(Lit(atom_ab, False))
            if (ba):
                evidences.add(Lit(atom_ba))
            else:
                evidences.add(Lit(atom_ba, False))
        return frozenset(evidences)

    def is_positive(self, pred: Pred) -> bool:
        idx = self.preds.index(pred)
        return self.code_ab[idx], self.code_ba[idx]

    def __str__(self):
        evidences = self.get_evidences()
        return str(evidences)

    def __repr__(self):
        return str(self)


class WFOMCContext(object):
    def __init__(self, mln: MLN, tree_constraint: TreeConstraint,
                 cardinality_constraint: CardinalityConstraint):
        self.mln: MLN = mln
        self.tree_constraint: TreeConstraint = tree_constraint
        self.cardinality_constraint: CardinalityConstraint = cardinality_constraint

        self.domain: Set[Const] = self.mln.domain
        self.sentence: CNF
        self.weights: Dict[Pred, Rational] = dict()

        # For existantial quantifiers
        # existential quantified predicates,
        # they should be some converted tseitin predicates
        self.ext_preds: List[Pred] = list()
        self.uni_var_indices: List[int] = list()
        self.tseitin_preds: List[Pred] = list()
        self.tseitin_to_extpred: Dict[Pred, Pred] = dict()
        self.tseitin_to_skolem: Dict[Pred, Pred] = dict()
        self.domain_preds: List[Pred] = list()
        self.domain_to_evidence_preds: Dict[Pred, FrozenSet[Pred]] = dict()
        self.eutype_preds: List[Pred] = list()
        self.skolemized_sentence: CNF  # skolemized sentence
        # self._skolem_preds: List[Pred] = list()  # skolem predicates
        # self._skolem_to_tseitin: Dict[Pred, Pred] = dict()

        # Convert to FO sentence for WFOMC
        self.build_sentence()
        logger.info('sentence for WFOMC: %s', self.sentence)
        logger.info('weights for WFOMC:')
        for pred, w in self.weights.items():
            logger.info('%s: %s', pred, w)

        # deal with tree axiom
        if self.tree_constraint is not None:
            if self.contain_tree_constraint():
                tree_lit = Lit(self.tree_constraint.pred(a, b))
                self.tree_p_evidence = frozenset([tree_lit])
                self.tree_n_evidence = frozenset([tree_lit.negate()])
            else:
                raise RuntimeError(
                    "Unknown tree constraint: %s", type(self.tree_constraint)
                )

        # deal with existential quantifiers
        if self.contain_existential_quantifier():
            tseitin_clauses = [
                DisjunctiveClause(frozenset([Lit(p(x))])) for p in self.tseitin_preds
            ]
            evidence_sentence = []
            for flag in product(*([[True, False]] * len(tseitin_clauses))):
                domain_pred = new_predicate(1, EVIDOM_PRED_NAME)
                if any(flag):
                    evidences = [clause for idx, clause in enumerate(
                        tseitin_clauses) if flag[idx]]
                    # evidom(x) => evi1(x) ^ evi2(x) ^ ... ^ evim(x)
                    evidence_sentence.append(
                        CNF.from_lit(Lit(domain_pred(x), False)).Or(
                            CNF(frozenset(evidences))
                        )
                    )
                    evidence_preds = frozenset(
                        pred for idx, pred in enumerate(self.tseitin_preds) if flag[idx]
                    )
                else:
                    evidence_preds = frozenset()
                self.domain_preds.append(domain_pred)
                self.domain_to_evidence_preds[domain_pred] = evidence_preds
            self.skolemized_sentence = self.skolemized_sentence.And(
                AndCNF(*evidence_sentence)
            ).And(exact_one_of(self.domain_preds))

            logger.info('skolemized sentence for WFOMC: %s',
                        self.skolemized_sentence)

            self.ebtypes: List[EBType] = self._get_ebtypes()

    def _get_ebtypes(self) -> List[EBType]:
        ebtypes = list()
        n_ext_preds = len(self.ext_preds)
        for i in product(*([[True, False]] * n_ext_preds)):
            for j in product(*([[True, False]] * n_ext_preds)):
                ebtypes.append(
                    EBType(i, j, tuple(self.ext_preds),
                           tuple(self.uni_var_indices))
                )
        return ebtypes

    def contain_tree_constraint(self) -> bool:
        return isinstance(self.tree_constraint, TreeConstraint)

    def contain_cardinality_constraint(self) -> bool:
        return isinstance(self.cardinality_constraint, CardinalityConstraint)

    def contain_existential_quantifier(self) -> bool:
        return len(self.ext_preds) > 0

    def get_weight_fn(self, weights: Dict[Pred, Rational] = None) \
            -> Callable[[Pred], Tuple[Rational, Rational]]:
        if weights is None:
            weights = self.weights

        def get_weight(pred: Pred) -> Tuple[Rational, Rational]:
            default = Rational(1, 1)
            if pred in weights:
                return weights[pred]
            return (default, default)
        return get_weight

    def get_weight(self, pred: Pred) -> Tuple[Rational, Rational]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def build_sentence(self):
        sentence = []
        skolemized_sentence = []
        for idx, (formula, weight) in enumerate(self.mln):
            if formula.is_exist():
                variables = tuple(formula.vars())
                aux_pred = new_predicate(len(variables), AUXILIARY_PRED_NAME)
                aux_atom = aux_pred(*variables)
                aux_cnf = CNF.from_atom(aux_atom)
                aux_equation = formula.cnf.Equate(aux_cnf)
                sentence.append(aux_equation)
                skolemized_sentence.append(aux_equation)

                # self.aux_to_formula[aux_atom] = formula
                # NOTE: presume the existential quantified formula is hard
                assert self.mln.is_hard(idx)
                ext_vars, uni_vars = formula.ext_uni_vars()
                # NOTE: only support \forall x \exist y: R(x,y)
                assert len(ext_vars) == 1 and len(uni_vars) == 1
                uni_var = list(uni_vars)[0]
                if uni_var == aux_atom.args[0]:
                    self.uni_var_indices.append(0)
                else:
                    self.uni_var_indices.append(1)
                self.ext_preds.append(aux_pred)

                # \forall x: T(x) <=> \exist y: R(x,y)
                # \forall x\forall y: \neg R(x,y) v T(x)
                # \forall x\exist y: \neg T(x) v R(x,y)
                tseitin_pred = new_predicate(len(uni_vars), TSEITIN_PRED_NAME)
                tseitin_atom = tseitin_pred(*uni_vars)
                aux_lit = Lit(aux_atom)
                tseitin_lit = Lit(tseitin_atom)
                tseitin_equation = CNF.from_clause(
                    DisjunctiveClause(frozenset(
                        [aux_lit.negate(), tseitin_lit]
                    ))
                )
                skolemized_sentence.append(tseitin_equation)
                # NOTE: adding tseitin
                # skolemized_sentence.append(CNF.from_lit(tseitin_lit))
                self.tseitin_preds.append(tseitin_pred)
                self.tseitin_to_extpred[tseitin_pred] = aux_pred

                # skolemization: S(x) v (T(x) ^ \neg R(x,y))
                # (S(x) v T(x)) ^ (S(x) v \neg R(x,y))
                skolem_pred = new_predicate(len(uni_vars), SKOLEM_PRED_NAME)
                skolem_atom = skolem_pred(*uni_vars)
                skolem_lit = Lit(skolem_atom)
                skolemized_sentence.append(
                    CNF(
                        frozenset([
                            DisjunctiveClause(
                                frozenset([skolem_lit, tseitin_lit])
                            ),
                            DisjunctiveClause(
                                frozenset([skolem_lit, aux_lit.negate()])
                            )
                        ])
                    )
                )
                self.weights[skolem_pred] = (Rational(1, 1), Rational(-1, 1))
                # self._skolem_preds.append(skolem_pred)
                # self._skolem_to_tseitin[skolem_pred] = tseitin_pred
                self.tseitin_to_skolem[tseitin_pred] = skolem_pred
            else:
                if self.mln.is_hard(idx):
                    sentence.append(formula.cnf)
                    skolemized_sentence.append(formula.cnf)
                else:
                    # set weight for aux predicate
                    variables = tuple(formula.vars())
                    aux_pred = new_predicate(
                        len(variables), AUXILIARY_PRED_NAME)
                    aux_atom = aux_pred(*variables)
                    aux_cnf = CNF.from_atom(aux_atom)
                    aux_equation = formula.cnf.Equate(aux_cnf)
                    sentence.append(aux_equation)
                    skolemized_sentence.append(aux_equation)
                    self.weights[aux_pred] = (
                        Rational(math.exp(weight), 1), Rational(1, 1)
                    )
        self.sentence = AndCNF(*sentence)
        if len(skolemized_sentence) > 0:
            self.skolemized_sentence = AndCNF(
                *skolemized_sentence)
