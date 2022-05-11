import numpy as np
from typing import Tuple, List, Union

from logzero import logger
from collections import defaultdict


def tree_sum(w: np.ndarray, contracted_edges: List[Tuple[int]] = None) -> np.ndarray:
    w = w + w.transpose(1, 0, 2)
    origin_w = w.copy()
    logger.debug('contraction edges: %s', contracted_edges)
    contraction_vertices = set()
    # map contracted vertex to the contracting vertex
    contraction_mapping = list(range(w.shape[0]))
    contraction_mapping_reverse = defaultdict(set)
    prod_weight = np.ones([w.shape[2]], dtype=w.dtype)
    if contracted_edges is not None:
        for edge in contracted_edges:
            v1 = contraction_mapping[edge[0]]
            v2 = contraction_mapping[edge[1]]
            logger.debug('Contract %s(%s) and %s(%s)',
                         edge[0], v1, edge[1], v2)
            if v1 == v2:
                logger.debug(
                    'Contract failed: adding {} leads to a cycle'.format(edge))
                return 0
            prod_weight *= origin_w[edge[0], edge[1], :]
            w[v1, :, :] += w[v2, :, :]
            w[:, v1, :] += w[:, v2, :]
            contraction_vertices.add(v2)
            w[range(w.shape[1]), range(w.shape[1]), :] = 0
            # map v2 to v1
            contraction_mapping[v2] = v1
            contraction_mapping_reverse[v1].add(v2)
            for k in contraction_mapping_reverse[v2]:
                contraction_mapping[k] = v1
                contraction_mapping_reverse[v1].add(k)
    remaining_vertices = set(range(w.shape[0]))
    remaining_vertices = list(remaining_vertices - contraction_vertices)
    logger.debug('Remaining vertices: %s', remaining_vertices)
    w = w[remaining_vertices, :, :][:, remaining_vertices, :]
    row_sum = np.sum(w, axis=0)
    w = np.eye(row_sum.shape[0])[..., None] * row_sum[None, ...] - w
    w = w[:-1, :-1, :].transpose(2, 0, 1)
    # NOTE: np.linalg.det does not support complex256
    if w.dtype == np.complex256:
        w = w.astype(np.complex)
    return prod_weight * np.linalg.det(w)


class TreeSumContext(object):
    def __init__(self, w: np.ndarray, contracted_edges: List[Tuple[int]] = None):
        """
        Context for calculating tree sume of a undirected graph characterized by the adjacent matrix `w` with the constraint edges.

        :param w np.ndarray: the adjacent matrix, the weight (the last dimension) can be an array
        :param contraction_edges List[Tuple[int]]: the edges that must be contained in the spanning tree
        """
        super().__init__()
        self.w = w
        self.contracted_edges = contracted_edges
        if np.any(tree_sum(self.w, self.contracted_edges)) == 0:
            raise RuntimeError(
                'Initial tree sum cannot be 0'
            )

    def is_contracted(self, edge: Tuple[int]) -> bool:
        return edge in self.contracted_edges or edge[::-1] in self.contracted_edges

    def try_connect(self, edge: Tuple[int]) -> Tuple[np.ndarray]:
        connected_ts = tree_sum(self.w, self.contracted_edges + [edge])
        if self.is_contracted(edge):
            return 0, connected_ts
        tmp = self.w[edge[0], edge[1], :]
        self.w[edge[0], edge[1], :] = 0
        disconnected_ts = tree_sum(self.w, self.contracted_edges)
        self.w[edge[0], edge[1], :] = tmp
        return disconnected_ts, connected_ts

    def connect(self, edge: Tuple[int]) -> None:
        self.w[edge[0], edge[1]] = 1
        self.contracted_edges.append(edge)

    def disconnect(self, edge: Tuple[int]) -> None:
        self.w[edge[0], edge[1]] = 0
