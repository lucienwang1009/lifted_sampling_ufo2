import subprocess
import os
import re
import time

from logzero import logger
from contexttimer import Timer
from pracmln import MLN

from generator import MLNGenerator
from py4j.java_gateway import JavaGateway


class PartitionFunctionSolver(object):
    pass


class WFOMCSolver(PartitionFunctionSolver):
    def __init__(self):
        self.mln_generator = MLNGenerator()
        self.pattern = re.compile(r'exp\(([\d\.\-E]+)\)')
        self.jar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'forclift.jar')

        self.calls = 0

    def __enter__(self):
        self.process = self.start_forclift()
        time.sleep(0.5)
        return self

    def __exit__(self, type, value, traceback):
        self.stop_forclift()

    def start_forclift(self):
        command = [
            'java', '-jar', self.jar_file, '--gateway'
        ]
        return subprocess.Popen(command)

    def _call(self, mln):
        self.calls += 1
        gateway = JavaGateway()
        with self.mln_generator.generate(mln) as file_name:
            with Timer() as t:
                result = gateway.entry_point.WFOMC(file_name, True)
            logger.debug('elapsed time for WFOMC call: %s', t.elapsed)
            logger.debug('result: %s', result)
        return result

    def solve(self, mln):
        """
        Solve the partition function problem for given MLN.
        Return ln(Z) where Z is the partition function.
        """
        result = self._call(mln)
        res = re.findall(self.pattern, result)
        if not res or len(res) > 1:
            raise RuntimeError('Exception while running WFOMC: {}'.format(result))
        return float(res[0])

    def stop_forclift(self):
        self.process.kill()


class ComplexWFOMCSolver(WFOMCSolver):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(r'exp\(([\d\.\-\+Ei]+)\)')
        self.jar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'forclift_complex.jar')

        self.calls = 0

    def solve(self, mln):
        """
        Solve the partition function problem for given MLN.
        Return ln(Z) where Z is the partition function.
        """
        result = self._call(mln)
        res = re.findall(self.pattern, result)
        if not res or len(res) > 1:
            raise RuntimeError('Exception while running WFOMC: {}'.format(result))
        return complex(res[0].replace('i', 'j'))


if __name__ == '__main__':
    mln = MLN.load('./models/friendsmoker.mln', grammar='StandardGrammar')
    with WFOMCSolver() as solver:
        print(solver.solve(mln))
