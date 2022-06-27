from parsimonious.grammar import Grammar
from parsimonious.nodes import Node, NodeVisitor
from logzero import logger
from typing import Set

from sampling_ufo2.parser.grammars import mln_with_constraint_rules
from sampling_ufo2.fol.syntax import Var, Const, Atom, Pred, Lit, DisjunctiveClause, CNF, Exist, QuantifiedFormula
from sampling_ufo2.network import MLN
from sampling_ufo2.network import TreeConstraint, CardinalityConstraint


class MLNParseException(RuntimeError):
    pass


class MLNVisitor(NodeVisitor):

    unwrapped_exceptions = (MLNParseException)

    def __init__(self):
        super().__init__()
        # NOTE: support only one domain for now
        self.domain: Set[str]
        self.domain_name: str
        self.predicate_definition: Set[Pred] = set()

    def visit_mln(self, node, visited_children):
        _, _, items = visited_children
        weights = []
        formulas = []
        for item in items:
            _, weighted_formula, _ = item
            if weighted_formula is None:
                # found predicate definition
                continue
            weight, formula = weighted_formula
            weights.append(weight)
            formulas.append(formula)
        try:
            mln = MLN(formulas, weights, self.domain,
                      self.predicate_definition)
        except RuntimeError as e:
            raise MLNParseException(e)
        return mln

    def visit_domain(self, node, visited_children):
        name, _, _, _, spec_or_num = visited_children
        spec_or_num = spec_or_num[0][0]
        if isinstance(spec_or_num, float):
            spec_or_num = set('C{}'.format(i)
                              for i in range(1, int(spec_or_num) + 1))
        spec_or_num = set(map(lambda x: Const(str(x)), spec_or_num))
        logger.debug('Domain spec: %s', spec_or_num)
        self.domain = spec_or_num
        self.domain_name = name

    def visit_domain_spec(self, node, visited_children):
        return set(self._visit_list(*visited_children[1:3]))

    def visit_domain_slice(self, node, visited_children):
        """
        domain_slice = "{" num sep "..." sep num "}"
        """
        _, start, _, _, _, end, _ = visited_children
        return set(range(int(start), int(end) + 1))

    def visit_weighted_formula(self, node, visited_children):
        weight, formula, hard = visited_children
        atoms = formula.atoms()
        if len(atoms) == 1:
            atom = next(iter(atoms))
            if atom.vars() == frozenset([Var(self.domain_name)]) and \
                    len(atom.consts()) == 0:
                self.predicate_definition.add(atom.pred)
                logger.debug('Found predicate definiton: %s', atom)
                return None

        if self._match_nothing(weight):
            weight = float('inf')
        else:
            weight = weight[0][0]
        if self._match_nothing(hard) and weight == float('inf'):
            raise MLNParseException(
                'Formula should be either weighted or hard: %s', formula)
        logger.debug('Weighted formula: %s, %s', weight, formula)
        return weight, formula

    def visit_formula(self, node, visited_children):
        exist, cnf = visited_children
        if self._match_nothing(exist):
            formula = QuantifiedFormula(cnf)
        else:
            exist = exist[0]
            for v in exist.quantified_vars:
                if v not in cnf.vars():
                    raise MLNParseException(
                        'Quantified variable not in formula: %s, %s',
                        v, cnf
                    )
            formula = QuantifiedFormula(cnf, exist)
        logger.debug('Quantified Formula: %s', formula)
        return formula

    def visit_exist(self, node, visited_children):
        _, _, terms, _ = visited_children
        if len(terms) < 1:
            raise MLNParseException(
                'At least one variable is quantified'
            )
        return Exist(frozenset(Var(v) for v in terms))

    def visit_cnf(self, node, visited_children):
        clauses = self._visit_list(*visited_children[1:3])
        cnf = CNF(frozenset(clauses))
        logger.debug("CNF: %s", cnf)
        return cnf

    def visit_clause(self, node, visited_children):
        lits = self._visit_list(*visited_children[1:3])
        clause = DisjunctiveClause(frozenset(lits))
        logger.debug("Clause: %s", clause)
        return clause

    def visit_literal(self, node, visited_children):
        neg, atom = visited_children
        lit = Lit(atom, self._match_nothing(neg))
        logger.debug("Literal: %s", lit)
        return lit

    def visit_atom(self, node, visited_children):
        pred, _, terms, _ = visited_children
        fol_terms = []
        for t in terms:
            if t in self.domain:
                fol_terms.append(
                    Const(t)
                )
            else:
                fol_terms.append(
                    Var(t)
                )
        atom = Atom(
            Pred(pred, len(terms)),
            tuple(fol_terms)
        )
        logger.debug('Atom: %s', atom)
        return atom

    def visit_terms(self, node, visited_children):
        terms = self._visit_list(*visited_children)
        logger.debug('Terms: %s', terms)
        return terms

    def visit_one_line(self, node, visited_children):
        return node.text

    def visit_sym(self, node, visited_children):
        return node.text

    def visit_num(self, node, visited_children):
        return float(node.text)

    def generic_visit(self, node, visited_children):
        return visited_children or node

    def _visit_list(self, head, remaining):
        res = []
        # empty list
        if self._match_nothing(head):
            return []
        # first item with "?"
        elif isinstance(head, list):
            res = head
        else:
            res = [head]
        for sep_item in remaining:
            _, item = sep_item
            res.append(item)
        return res

    def _match_nothing(self, obj):
        return isinstance(obj, Node) and obj.text == ""


class MLNConstraintVisitor(MLNVisitor):
    def __init__(self):
        super().__init__()

    def visit_mln_with_constraint(self, node, visited_children):
        mln, _, constraints, _ = visited_children
        tree, ccs = constraints
        tree_constraint = None
        cardinality_constraint = None
        if tree is None and ccs is None:
            return mln, None, None
        preds = mln.preds()
        preds = dict((pred.name, pred) for pred in preds)
        if tree is not None:
            if len(tree) == 1 and tree[0] in preds:
                tree_constraint = TreeConstraint(preds[tree[0]])
            else:
                raise MLNParseException(
                    "Tree constraint %s is not correct", tree
                )
        if ccs is not None:
            assert isinstance(ccs, list)
            pred2card = set()
            for pred, card in ccs:
                if pred in preds and card.is_integer():
                    pred2card.add((preds[pred], int(card)))
                else:
                    raise MLNParseException(
                        "Cardinality constraint |%s|=%s is wrong", pred, card
                    )
            cardinality_constraint = CardinalityConstraint(
                frozenset(pred2card)
            )
        return mln, tree_constraint, cardinality_constraint

    def visit_constraints(self, node, visited_children):
        tree, _, ccs = visited_children
        if self._match_nothing(tree):
            tree = None
        else:
            tree = tree[0][0]
        if self._match_nothing(ccs):
            ccs = None
        else:
            ccs = ccs[0]
        return (tree, ccs)

    def visit_tree(self, node, visited_children):
        _, pred, _ = visited_children
        return pred

    def visit_ccs(self, node, visited_children):
        ccs = self._visit_list(*visited_children)
        return ccs

    def visit_cc(self, node, visited_children):
        _, pred, *_, card = visited_children
        return pred, card


if __name__ == '__main__':
    grammar = Grammar(mln_with_constraint_rules)
    tree = grammar.parse("""
    person = 10

    smokes(person)
    friends(person, person)

    1.2 friends(x,y) ^ smokes(x) v smokes(y)
    !friends(x, y) v friends(y,x).

    Tree[friends]
    |smokes| = 6
    """)
    mln, tree_constraint, cardinality_constraint = MLNConstraintVisitor().visit(tree)

    logger.info(mln)
    if tree_constraint is not None:
        logger.info(tree_constraint)
    if cardinality_constraint is not None:
        logger.info(cardinality_constraint)
