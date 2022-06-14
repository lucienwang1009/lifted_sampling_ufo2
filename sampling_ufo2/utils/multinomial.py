import functools

from typing import Tuple, List


def multinomial(length: int, total_sum: int) -> Tuple[int]:
    """
    Generate a list of numbers, whose size is `length` and sum is `total_sum`

    :param length int: length of the generated list
    :param total_sum int: the summation over the list
    :rtype Tuple[int]:
    """
    if length == 1:
        yield (total_sum, )
    else:
        for value in range(total_sum + 1):
            for permutation in multinomial(length - 1, total_sum - value):
                yield (value, ) + permutation


class MultinomialCoefficients(object):
    """
    Multinomial coefficients

    Usage:
    ```
    MultinomialCoefficients.precompute_pascal(n)
    ...
    MultinomialCoefficients.coef(list)
    ```


    """
    pt: List[List[int]] = None
    n: int = 0

    @staticmethod
    # @jit
    def precompute(n: int):
        """
        Pre-compute the pascal triangle.

        :param n int: the maximal total sum
        """
        pt: List[List[int]] = []
        lst: List[int] = [1]
        for i in range(n + 1):
            pt.append(lst)
            newlist = []
            newlist.append(lst[0])
            for i in range(len(lst) - 1):
                newlist.append(lst[i] + lst[i + 1])
            newlist.append(lst[-1])
            lst = newlist
        MultinomialCoefficients.pt = pt
        MultinomialCoefficients.n = n

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def coef(lst: Tuple[int]) -> int:
        """
        Compute the multinomial coefficient of `lst`
        """
        if MultinomialCoefficients.pt is None:
            raise RuntimeError(
                'Please initialize MultinomialCoefficients first by `MultinomialCoefficients.precompute(n)`'
            )
        if sum(lst) > MultinomialCoefficients.n:
            raise RuntimeError(
                'The sum %d of input is larger than precomputed maximal sum %d, '
                'please re-initialized MultinomialCoefficients using bigger n',
                sum(lst), MultinomialCoefficients.n
            )
        ret = 1
        tmplist = lst
        while len(tmplist) > 1:
            ret *= MultinomialCoefficients._mycomb(sum(tmplist), tmplist[-1])
            tmplist = tmplist[:-1]
        return ret

    def _mycomb(a, b):
        if a < b:
            return 0
        elif b == 0:
            return 1
        else:
            return MultinomialCoefficients.pt[a][b]
