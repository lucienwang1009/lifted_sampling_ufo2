import math

from pracmln import MLN
from pracmln.mln.mlnpreds import Predicate
from pracmln.logic.fol import Implication, Exist, Lit, Biimplication
from logzero import logger


aux_pred_prefix = 'aux'


class Context(object):
    def __init__(self, mln):
        self.mln = mln
        self.formula, self.weights = self._to_wfomc_sentence()
        self.preds = self.mln.predicates

    def ground_formula(self, c1, c2):
        varnames = list(self.formula.vardoms().keys())
        gnd_formula = self.formula.ground(
            None, {varnames[0]: c1, varnames[1]: c2}, partial=True
        )
        return gnd_formula

    def get_weight(self, predname):
        if predname in self.weights:
            return self.weights[predname]
        return (1, 1)

    def _to_wfomc_sentence(self):
        weights = {}
        sentence_strs = []
        for index, formula in enumerate(self.mln.formulas):
            aux_predname = '{}{}'.format(aux_pred_prefix, index)
            # set weight for aux predicate
            weights[aux_predname] = (math.exp(float(formula.weight)), 1)
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
