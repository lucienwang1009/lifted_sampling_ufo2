import os
import tempfile

from logzero import logger
from contextlib import contextmanager

from pracmln import MLN

class MLNGenerator(object):
    def __init__(self):
        super().__init__()

    @contextmanager
    def generate(self, mln):
        fd = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.mln',
            delete=False
        )
        content = ''
        for name, domain in mln.domains.items():
            content += '{} = {{{}}}{}'.format(
                name,
                ', '.join(domain),
                os.linesep
            )
        content += os.linesep
        for predicate in mln.predicates:
            content += str(predicate) + os.linesep
        content += os.linesep

        for formula in mln.formulas:
            if formula.ishard:
                content += '{}.'.format(str(formula))
            else:
                # NOTE: mln can have complex weight
                weight = formula.weight
                if isinstance(weight, complex):
                    weight_str = '{},{}'.format(
                        weight.real,
                        weight.imag
                    )
                else:
                    weight_str = weight
                content += '{} {}{}'.format(
                    weight_str,
                    str(formula),
                    os.linesep
                )
        fd.file.write(content)
        fd.close()
        try:
            yield fd.name
        finally:
            logger.debug('delete tmp file: %s', fd.name)
            os.remove(fd.name)


if __name__ == '__main__':
    mln = MLN.load('./models/friendsmoker.mln', grammar='StandardGrammar')
    generator = MLNGenerator()
    with generator.generate(mln) as file_name:
        with open(file_name, 'r') as f:
            for line in f.readlines():
                print(line.strip())
