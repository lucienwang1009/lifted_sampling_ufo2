import numpy as np
from typing import Tuple, List, Union

from logzero import logger
from collections import defaultdict


class ArborescenceSumContext(object):
    def __init__(self, w: np.ndarray, root: int, contracted_edges: List[Tuple[int]] = None):
        """
        Context for calculating tree sume of a undirected graph characterized by the adjacent matrix `w` with the constraint edges.

        :param w np.ndarray: the adjacent matrix, the weight (the last dimension) can be an array
        :param contracted_edges List[Tuple[int]]: the edges that must be contained in the spanning tree
        """
        super().__init__()
        self.w = w
        self.origin_w = self.w.copy()
        self.root = root
        self.contracted_edges = contracted_edges
        logger.debug('root: %s', self.root)
        logger.debug('contraction edges: %s', self.contracted_edges)
        self.impossible = False
        # no edge is from root
        if np.all(self.w[root, :, :] == 0):
            self.impossible = True
            logger.debug(
                'No edge is from root, tree sum is 0'
            )
            return
        # there is a constraint edge points to root
        for edge in self.contracted_edges:
            if edge[1] == self.root:
                self.impossible = True
                logger.debug(
                    'There exists an edge pointing to root, tree sum is 0'
                )
                return

        self.contracted_vertices = set()
        # map contracted vertex to the contracting vertex
        self.contraction_mapping = list(range(self.w.shape[0]))
        self.contraction_mapping_reverse = defaultdict(set)
        self.prod_weight = np.ones([self.w.shape[2]], dtype=self.w.dtype)
        if self.contracted_edges is not None:
            for edge in self.contracted_edges:
                if not self.contract(edge):
                    self.impossible = True
                    break

    def contract(self, edge: Tuple[int]) -> bool:
        i = self.contraction_mapping[edge[0]]
        j = self.contraction_mapping[edge[1]]
        logger.debug('Contract %s(%s) and %s(%s)', edge[0], i, edge[1], j)
        if i == j:
            logger.debug(
                'Contract failed: adding %s leads to a cycle', edge)
            return False
        if j != edge[1]:
            logger.debug(
                'Contract failed: vertex %s is contracted twice', edge[1]
            )
            return False
        self.prod_weight *= self.origin_w[edge[0], edge[1], :]
        self.w[i, :, :] += self.w[j, :, :]
        # self.w[:, i, :] += self.w[:, j, :]
        self.contracted_vertices.add(j)
        self.w = self._fill_diagonal(self.w)
        # map j to i
        self.contraction_mapping[j] = i
        self.contraction_mapping_reverse[i].add(j)
        for k in self.contraction_mapping_reverse[j]:
            self.contraction_mapping[k] = i
            self.contraction_mapping_reverse[i].add(k)
        return True

    def try_contract(self, edge: Tuple[int]) -> Union[bool, np.ndarray]:
        if self.impossible:
            return 0
        i = self.contraction_mapping[edge[0]]
        j = self.contraction_mapping[edge[1]]
        logger.debug('Try to contract %s(%s) and %s(%s)',
                     edge[0], i, edge[1], j)
        if i == j:
            # cycle
            return 0
        w = self.w.copy()
        prod_weight = self.prod_weight * w[:, i, j]
        w[:, i, :] += w[:, j, :]
        w[:, :, i] += w[:, :, j]
        w = self._fill_diagonal(w)
        return self._internal_tree_sum(w, prod_weight, self.contracted_vertices.union(set([j])))

    def _remove_vertices(self, w: np.ndarray, contracted_vertices: List[Tuple[int]]) -> np.ndarray:
        remaining_vertices = set(range(w.shape[0]))
        remaining_vertices = list(
            sorted(remaining_vertices - contracted_vertices))
        root_new_idx = remaining_vertices.index(self.root)
        logger.debug('Remaining vertices: %s', remaining_vertices)
        return w[remaining_vertices, :, :][:, remaining_vertices, :], root_new_idx

    def _fill_diagonal(self, w):
        w[range(w.shape[1]), range(w.shape[1]), :] = 0
        return w

    def _internal_tree_sum(self, w, root, prod_weight, contracted_vertices):
        w = w.copy()
        w, root_idx = self._remove_vertices(w, contracted_vertices)
        row_sum = np.sum(w, axis=0)
        w = np.eye(row_sum.shape[0])[..., None] * row_sum[None, ...] - w
        w = np.delete(
            np.delete(w, root_idx, 0), root_idx, 1
        ).transpose(2, 0, 1)
        # NOTE: np.linalg.det does not support complex256
        if w.dtype == np.complex256:
            w = w.astype(np.complex)
        return prod_weight * np.linalg.det(w)

    def tree_sum(self) -> np.ndarray:
        if self.impossible:
            ret = 0
        else:
            ret = self._internal_tree_sum(
                self.w, self.root, self.prod_weight, self.contracted_vertices)
        logger.info('Directed tree sum: %s', ret)
        return ret
