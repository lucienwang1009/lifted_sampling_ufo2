import os
import decimal
import pickle
from logzero import logger
import logzero
import logging
from pyunigen import Sampler as unigen_sampler
from sampling_ufo2.fol.syntax import QuantifiedFormula, CNF, Const, \
    Substitution, AndCNF, OrCNF
from sampling_ufo2.context import WFOMCContext
from sampling_ufo2.parser import parse_mln_constraint
from pysat.card import CardEnc, EncType
from contexttimer import Timer
from typing import List
from copy import deepcopy
import argparse


logzero.loglevel(logging.INFO)


class RetVal:
    def __init__(self, origVars, origCls, vars, totalCount, div):
        self.origVars = origVars
        self.origCls = origCls
        self.vars = vars
        self.totalCount = totalCount
        self.div = div


class Converter:
    def __init__(self, precision, verbose=False):
        self.precision = precision
        self.verbose = verbose
        self.samplSet = {}

    def pushVar(self, variable, cnfClauses):
        cnfLen = len(cnfClauses)
        for i in range(cnfLen):
            cnfClauses[i].append(variable)
        return cnfClauses

    def getCNF(self, variable, binStr, sign, origVars):
        cnfClauses = []
        binLen = len(binStr)
        cnfClauses.append([binLen+1+origVars])
        for i in range(binLen):
            newVar = binLen-i+origVars
            if sign is False:
                newVar = -1*(binLen-i+origVars)
            if binStr[binLen-i-1] == '0':
                cnfClauses.append([newVar])
            else:
                cnfClauses = self.pushVar(newVar, cnfClauses)
        self.pushVar(variable, cnfClauses)
        return cnfClauses

    def encodeCNF(self, variable, kWeight, iWeight, origvars, cls, div):
        if iWeight == 1 and kWeight == 1:
            return "", origvars, cls, div+1

        if iWeight == 0:
            if kWeight == 0:
                lines = "-%d 0\n" % variable
                return lines, origvars, cls+1, div
            else:
                lines = "%d 0\n" % variable
                return lines, origvars, cls+1, div

        self.samplSet[origvars+1] = 1
        binStr = str(bin(int(kWeight)))[2:-1]
        binLen = len(binStr)
        for i in range(iWeight-binLen-1):
            binStr = '0'+binStr
        for i in range(iWeight-1):
            self.samplSet[origvars+i+2] = 1
        complementStr = ''
        for i in range(len(binStr)):
            if binStr[i] == '0':
                complementStr += '1'
            else:
                complementStr += '0'
        origCNFClauses = self.getCNF(-variable, binStr, True, origvars)

        writeLines = ''
        for i in range(len(origCNFClauses)):
            cls += 1
            for j in range(len(origCNFClauses[i])):
                writeLines += str(origCNFClauses[i][j])+' '
            writeLines += '0\n'

        -variable
        cnfClauses = self.getCNF(variable, complementStr, False, origvars)
        for i in range(len(cnfClauses)):
            if cnfClauses[i] in origCNFClauses:
                continue
            cls += 1
            for j in range(len(cnfClauses[i])):
                writeLines += str(cnfClauses[i][j])+' '
            writeLines += '0\n'

        vars = origvars+iWeight
        return writeLines, vars, cls, div+iWeight

    # return the number of bits needed to represent the weight (2nd value returned)
    # along with the weight:bits ratio
    def parseWeight(self, initWeight):
        if type(initWeight) == float or type(initWeight) == str:
            initWeight = decimal.Decimal(initWeight)

        assert type(
            initWeight) == decimal.Decimal, "You must pass a float, string or a Decimal"

        assert self.precision > 1, "Precision must be at least 2"
        assert initWeight >= decimal.Decimal(
            0.0), "Weight must not be below 0.0"
        assert initWeight <= decimal.Decimal(
            1.0), "Weight must not be above 1.0"

        if self.verbose:
            print("Query for weight %s" % (initWeight))

        weight = initWeight*pow(2, self.precision)
        weight = weight.quantize(decimal.Decimal("1"))
        # for CEIL, but double the error, set:
        # weight = weight.quantize(decimal.Decimal("1"), rounding=decimal.ROUND_CEILING)
        weight = int(weight)
        prec = self.precision
        if self.verbose:
            print("weight %3.5f prec %3d" % (weight, prec))

        while weight % 2 == 0 and prec > 0:
            weight = weight/2
            prec -= 1

            if self.verbose:
                print("weight %3.5f prec %3d" % (weight, prec))

        if self.verbose:
            print("for %f returning: weight %3.5f prec %3d" %
                  (initWeight, weight, prec))

        return weight, prec

    #  The code is straightforward chain formula implementation
    def transform(self, lines, outputFile):
        origCNFLines = ''
        vars = 0
        cls = 0
        div = 0
        maxvar = 0
        foundCInd = False
        foundHeader = False
        for line in lines:
            if len(line) == 0:
                print("ERROR: The CNF contains an empty line.")
                print("ERROR: Empty lines are NOT part of the DIMACS specification")
                print("ERROR: Remove the empty line so we can parse the CNF")
                exit(-1)

            if line.strip()[:2] == 'p ':
                fields = line.strip().split()
                vars = int(fields[2])
                cls = int(fields[3])
                foundHeader = True
                continue

            # parse independent set
            if line[:5] == "c ind":
                foundCInd = True
                for var in line.strip().split()[2:]:
                    if var == "0":
                        break
                    self.samplSet[int(var)] = 1
                continue

            if line.strip()[0] == 'c':
                origCNFLines += str(line)
                continue

            if not foundHeader:
                print(
                    "ERROR: The 'p cnf VARS CLAUSES' header must be at the top of the CNF!")
                exit(-1)

            # an actual clause
            if line.strip()[0].isdigit() or line.strip()[0] == '-':
                for lit in line.split():
                    maxvar = max(abs(int(lit)), maxvar)
                origCNFLines += str(line)

            # NOTE: we are skipping all the other types of things in the CNF
            #       for example, the weights
            continue

        if maxvar > vars:
            print("ERROR: CNF contains var %d but header says we only have %d vars" % (
                maxvar, vars))
            exit(-1)

        print("Header says vars: %d  maximum var used: %d" % (vars, maxvar))

        if not foundHeader:
            print("ERROR: No header 'p cnf VARS CLAUSES' found in the CNF!")
            exit(-1)

        # if "c ind" was not found, then all variables are in the sampling set
        if not foundCInd:
            for i in range(1, vars+1):
                self.samplSet[i] = 1

        # weight parsing and CNF generation
        origWeight = {}
        transformCNFLines = ''
        for line in lines:
            if line.strip()[:2] == 'w ':
                fields = line.strip()[2:].split()
                var = int(fields[0])
                val = decimal.Decimal(fields[1])
                if val == decimal.Decimal(1):
                    print("c Skipping line due to val is 1 ", line.strip())
                    continue

                if var < 0:
                    print("c Skipping line due to literal <0 ", line.strip())
                    continue

                # already has been declared, error
                if var in origWeight:
                    print("ERROR: Variable %d has TWO weights declared" % var)
                    print("ERROR: Please ONLY declare each variable's weight ONCE")
                    exit(-1)

                if var not in self.samplSet:
                    print(
                        "ERROR: Variable %d has a weight but is not part of the sampling set" % var)
                    print(
                        "ERROR: Either remove the 'c ind' line or add this variable to it")
                    exit(-1)

                origWeight[var] = val
                self.samplSet[var] = 1
                kWeight, iWeight = self.parseWeight(val)

                if self.verbose:
                    representedW = decimal.Decimal(
                        kWeight)/decimal.Decimal(2**iWeight)
                    # print("kweight: %5d iweight: %5d" % (kWeight, iWeight))
                    print("var: %5d orig-weight: %s kweight: %5d iweight: %5d represented-weight: %s"
                          % (var, val, kWeight, iWeight, representedW))

                # we have to encode to CNF the translation
                eLines, vars, cls, div = self.encodeCNF(
                    var, kWeight, iWeight, vars, cls, div)
                transformCNFLines += eLines

        # with open(outputFile, 'w') as f:
        #     f.write('p cnf '+str(vars)+' '+str(cls)+' \n')
        #     f.write('c ind ')
        #     for k in self.samplSet:
        #         f.write("%d " % k)
        #     f.write("0\n")

        #     f.write(origCNFLines)
        #     f.write(transformCNFLines)

        # return RetVal(origVars, origCls, vars, cls, div), origCNFLines + transformCNFLines
        return origCNFLines + transformCNFLines


def ground_on_domain(sentence: QuantifiedFormula,
                     domain: List[Const],
                     constraints=None) -> CNF:
    num_vars = len(sentence.vars())
    assert num_vars <= 2, "Only support FO2"
    formulas = []
    if sentence.is_exist():
        ext_vars, uni_vars = sentence.ext_uni_vars()
        assert len(ext_vars) == 1 and len(uni_vars) == 1
        ext_var = list(ext_vars)[0]
        uni_var = list(uni_vars)[0]
        for c1 in domain:
            ext_formulas = []
            for c2 in domain:
                ext_formulas.append(
                    sentence.cnf.substitute(
                        Substitution(zip([uni_var, ext_var],
                                         [c1, c2]))
                    )
                )
            formulas.append(OrCNF(*ext_formulas))
        return AndCNF(*formulas)
    else:
        vars = list(sentence.vars())
        if num_vars == 1:
            for c in domain:
                formulas.append(
                    sentence.cnf.substitute(
                        Substitution(zip(vars, [c]))
                    )
                )
        else:
            for c1 in domain:
                for c2 in domain:
                    formulas.append(
                        sentence.cnf.substitute(
                            Substitution(zip(vars, [c1, c2]))
                        )
                    )
        return AndCNF(*formulas)


def to_unweighted(clauses, n_vars, n_clauses):
    lines = [
        f'p cnf {n_vars} {n_clauses}',
    ]
    lines += list(' '.join(str(c)
                           for c in clause) + '\n' for clause in clauses)
    decimal.getcontext().prec = 100
    c = Converter(precision=7, verbose=False)
    clause_lines = c.transform(lines, None)
    ret = []
    top_id = 0
    for line in clause_lines.split('\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        items = list(int(i) for i in line.split(' '))
        if items[-1] == 0:
            del items[-1]
        top_id = max(top_id, max(abs(i) for i in items))
        ret.append(items)
    return ret


def unigen_sampling(mln, ccs, k):
    mln = deepcopy(mln)
    with Timer() as timer:
        formulas = []
        ext_formulas = []
        ext_indices = []
        all_hard = True
        for idx, formula in enumerate(mln.formulas):
            if mln.is_hard(idx):
                all_hard = False
            if formula.is_exist():
                ext_formulas.append(formula)
                ext_indices.append(idx)
            # formulas.append(ground_on_domain(formula, mln.domain))
        for formula in ext_formulas:
            idx = mln.formulas.index(formula)
            del mln.formulas[idx]
            del mln.weights[idx]
        context = WFOMCContext(mln, None, None)
        formulas.append(ground_on_domain(
            QuantifiedFormula(context.sentence), mln.domain
        ))
        for formula in ext_formulas:
            formulas.append(ground_on_domain(
                formula, mln.domain
            ))
        grounding_cnf = AndCNF(*formulas)
        clauses, decode = grounding_cnf.encode_Dimacs(context.get_weight)
        top_id = len(decode)
        if ccs is not None:
            logger.info('Before CC, the number of lits: {}'.format(top_id))
            constrained_preds = []
            cardinalities = []
            for pred, card in ccs.pred2card:
                constrained_preds.append(pred)
                cardinalities.append(card)
            constrained_lits = [[] for _ in range(len(constrained_preds))]
            for id, lit in decode.items():
                for idx, pred in enumerate(constrained_preds):
                    if lit.pred() == pred:
                        constrained_lits[idx].append(id)
            for lits, card in zip(constrained_lits, cardinalities):
                res = CardEnc.equals(lits, bound=card,
                                     top_id=top_id,
                                     encoding=EncType.kmtotalizer)
                # print(top_id, res.clauses)
                clauses.extend(res.clauses)
                top_id = res.nv
            logger.info('After CC, the number of lits: {}'.format(top_id))
        # logger.info(f'Gound sentence: {clauses}')
    t_grounding = timer.elapsed
    logger.info(f'Time for grounding: {timer.elapsed}')
    if not all_hard:
        clauses = to_unweighted(clauses, top_id, len(clauses))

    with Timer() as timer:
        c = unigen_sampler()
        for clause in clauses:
            c.add_clause(clause)
        cells, hashes, samples = c.sample(num=k)
    logger.info(f'Time for sampling: {timer.elapsed}')
    logger.info(f'Total time: {timer.elapsed + t_grounding}')
    ret = []
    for sample in samples:
        ret.append(list(decode[id] for id in sample if id in decode))
    return ret


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sampler for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--n_samples', '-k', type=int, required=True)
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points')
    parser.add_argument('--show_samples', '-s',
                        action='store_true', default=False)
    parser.add_argument('--debug', '-d', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
        args.show_samples = True
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')
    mln, tree_constraint, cardinality_constraint = parse_mln_constraint(
        args.input)

    samples = unigen_sampling(mln, cardinality_constraint, args.n_samples)
    save_file = os.path.join(args.output_dir, 'samples.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(samples, f)
    logger.info('Samples are saved in %s', save_file)
    if args.show_samples:
        logger.info('Samples:')
        for s in samples:
            logger.info(sorted(str(i) for i in s))
