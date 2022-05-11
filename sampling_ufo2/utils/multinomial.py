import functools
from typing import Tuple


class MultinomialCoefficients(object):
    def __init__(self, n: int, m: int):
        """
        Compute multinomial coefficients.

        :param n int: the sum value
        :param m int: the term number
        """
        self.n = n
        self.m = m
        self.pt = self.precompute_pascal()

    # @staticmethod
    # @jit(nopython=True)
    def precompute_pascal(self):
        pt = []
        lst = [1]
        for i in range(self.n + 1):
            pt.append(lst)
            newlist = []
            newlist.append(lst[0])
            for i in range(len(lst) - 1):
                newlist.append(lst[i] + lst[i + 1])
            newlist.append(lst[-1])
            lst = newlist
        return pt

    def __iter__(self):
        return MultinomialCoefficients._recursive(self.m, self.n)

    @staticmethod
    def _recursive(length: int, total_sum: int):
        if length == 1:
            yield (total_sum, )
        else:
            for value in range(total_sum + 1):
                for permutation in MultinomialCoefficients._recursive(length - 1, total_sum - value):
                    yield (value, ) + permutation

    @functools.lru_cache(maxsize=None)
    def coef(self, lst: Tuple[int]) -> int:
        ret = 1
        tmplist = lst
        while len(tmplist) > 1:
            ret *= self._mycomb(sum(tmplist), tmplist[-1])
            tmplist = tmplist[:-1]
        return ret

    def _mycomb(self, a, b):
        if a < b:
            return 0
        elif b == 0:
            return 1
        else:
            return self.pt[a][b]
