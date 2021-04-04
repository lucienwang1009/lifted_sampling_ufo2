
from pracmln import MLN

from context import Context
from cell_graph import CellGraph


class Sampler(object):
    def __init__(self, mln):
        self.mln = mln
        self.context = Context(mln)
        self.cell_graph = CellGraph(self.context)


if __name__ == '__main__':
    mln = MLN.load('./models/friendsmoker.mln', grammar='StandardGrammar')
    sampler = Sampler(mln)
    sampler.cell_graph.show()
