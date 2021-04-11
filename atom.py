import re

from pracmln.logic.fol import Lit
from nnf import Var

class Atom(object):
    '''
    pracmln doesn't support Atom, so we implement it here
    '''
    def __init__(self, predname, args):
        self.predname = predname
        self.args = args

    def replace(self, assign):
        self.args = [assign[arg] for arg in self.args]
        return self

    def to_var(self):
        return Var(str(self))

    def ground(self, consts):
        pass

    @staticmethod
    def from_var(var):
        predname, args = re.findall(r'([^\(]*)\(([^\)]*)\)', var.name)[0]
        return Atom(predname, args.split(','))

    @staticmethod
    def from_literal(lit):
        return Atom(lit.predname, lit.args)

    def to_literal(self, negated=False):
        return Lit(negated, self.predname, self.vardoms)

    def __str__(self):
        return '{}({})'.format(self.predname, ','.join(self.args))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.predname == other.predname \
            and self.args == other.args

    def __hash__(self):
        return hash(self.__str__())
