import random

from copy import deepcopy
from nnf import Var, amc
from logzero import logger

from utils import to_nnf
from atom import Atom

class WMC(object):
    def __init__(self, gnd_formula, context):
        self.gnd_formula = gnd_formula
        self.context = context
        self.nnf = to_nnf(gnd_formula)
        self.var_weights = self._get_var_weights()

    def _get_var_weights(self):
        weights = {}
        for varname in self.nnf.vars():
            var = Var(varname)
            weights[var] = []
            weights[var.negate()] = []
            for d in range(self.context.w_dim):
                w = self.context.get_weight(Atom.from_var(var).predname, d)
                weights[var].append(w[0])
                weights[var.negate()].append(w[1])
        return weights

    def wmc(self, evidences=None, index=0):
        def weights_fn(var):
            if var in evidences:
                # NOTE: avoid duplicate multiplications
                return 1
            elif var.negate() in evidences:
                return 0
            return self.var_weights[var][index]
        return amc.WMC(self.nnf, weights_fn)


class WMCSampler(object):
    def __init__(self, wmc, evidences, context):
        self.wmc = wmc
        self.evidences = evidences
        self.context = context
        self.unknown_vars = self._get_unknown_vars()
        self.codes, self.dist = self._get_distribution()

    def _get_unknown_vars(self):
        unknown_vars = []
        for varname in self.wmc.nnf.vars():
            var = Var(varname)
            if var not in self.evidences \
                    and ~var not in self.evidences:
                unknown_vars.append(var)
        return unknown_vars

    def _get_distribution(self):
        codes = []
        dist = [[]] * self.context.w_dim
        for code in range(0, 2 ** len(self.unknown_vars) - 1):
            evidences = deepcopy(self.evidences)
            for i in range(len(self.unknown_vars)):
                if (code & (1 << i)):
                    evidences.add(self.unknown_vars[i])
                else:
                    evidences.add(self.unknown_vars[i].negate())
            codes.append(code)
            for k in range(self.context.w_dim):
                dist[k].append(self.wmc.wmc(evidences))
        return codes, dist

    def decode(self, code):
        decoded_vars = []
        for i in range(len(self.unknown_vars)):
            if (code & (1 << i)):
                decoded_vars.append(self.unknown_vars[i])
        return decoded_vars

    def sample(self, k=1):
        samples = [[]] * k
        if len(self.unknown_vars) == 0:
            return samples
        sample_codes = random.choices(self.codes, self.dist, k=k)
        for i, code in enumerate(sample_codes):
            samples[i].extend(self._decode(code))
        return samples
