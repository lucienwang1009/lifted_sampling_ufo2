import cmath

from pracmln import MLN
from pracmln.mln.mlnpreds import Predicate
from pracmln.logic.fol import Implication, Exist, Lit, Biimplication
from logzero import logger

from atom import Atom


aux_pred_prefix = 'aux'


class Context(object):
    def __init__(self, mln):
        self.mln = mln
        # the length of weight vector
        self.w_dim = 0
        self._convert_weight()

        self.formula, self.weights = self._to_wfomc_sentence()
        self.preds = self.mln.predicates

        gnd_formula_ab1 = self.ground_formula('a', 'b')
        gnd_formula_ab2 = self.ground_formula('b', 'a')
        self.mln.formula(
            '(' + str(gnd_formula_ab1) + ') ^ (' + str(gnd_formula_ab2) + ')'
        )
        self.gnd_formula_ab = self.mln.formulas[-1]
        self.gnd_formula_cc = self.ground_formula('c', 'c')
        # cache evidences
        self.same_gnd_atoms = [
            Atom(predname=p.name, args=(['c'] * len(p.argdoms)))
            for p in self.preds
        ]
        # logger.debug('same assignment evidences: %s', self.same_gnd_evidences)
        self.left_gnd_atoms = [
            Atom(predname=p.name, args=(['a'] * len(p.argdoms)))
            for p in self.preds
        ]
        # logger.debug('left assignment evidences: %s', self.left_gnd_evidences)
        self.right_gnd_atoms = [
            Atom(predname=p.name, args=(['b'] * len(p.argdoms)))
            for p in self.preds
        ]

    def _convert_weight(self):
        for f in self.mln.formulas:
            weights = f.weight.split(',')
            if self.w_dim == 0:
                self.w_dim = len(weights)
            elif self.w_dim != len(weights):
                raise RuntimeError('Different lengths of weight vector')
            f.weight = list(map(complex, weights))

    def ground_formula(self, c1, c2):
        varnames = list(self.formula.vardoms().keys())
        gnd_formula = self.formula.ground(
            None, {varnames[0]: c1, varnames[1]: c2}, partial=True
        )
        return gnd_formula

    def get_weight(self, predname, index):
        if predname in self.weights:
            return self.weights[predname][index]
        return (1, 1)

    def _to_wfomc_sentence(self):
        weights = {}
        sentence_strs = []
        for index, formula in enumerate(self.mln.formulas):
            aux_predname = '{}{}'.format(aux_pred_prefix, index)
            # set weight for aux predicate
            weights[aux_predname] = list(map(
                lambda x: (cmath.exp(x), 1), formula.weight
            ))
            # add predicate to mln
            self.mln.predicate(Predicate(aux_predname,
                                         list(formula.vardoms().values())))
            vars = formula.vardoms().keys()
            sentence_strs.append(
                '({}({}) <=> ({}))'.format(
                    aux_predname,
                    ','.join(vars),
                    str(formula)
                )
            )
        logger.debug('sentence for wfomc: %s', ' ^ '.join(sentence_strs))
        self.mln.formula(' ^ '.join(sentence_strs))
        return self.mln.formulas[-1], weights
