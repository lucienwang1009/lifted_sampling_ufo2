import random
from logzero import logger
from typing import Dict, FrozenSet, List, Set, Tuple
from collections import defaultdict
from itertools import product
from copy import deepcopy

from sampling_ufo2.utils import multinomial, RingElement
from sampling_ufo2.cell_graph import Cell
from sampling_ufo2.context import EBType
from sampling_ufo2.fol.syntax import Pred


class ExistentialContext(object):
    def __init__(self, cell_assignment: List[Cell],
                 tseitin_to_extpred: Dict[Pred, Pred]):
        self.cell_assignment = cell_assignment
        self.cell_elements: Dict[Cell, Set[int]] = defaultdict(set)
        self.cell_config: Dict[Cell, int] = defaultdict(lambda: 0)
        for idx, cell in enumerate(cell_assignment):
            self.cell_elements[cell].add(idx)
            self.cell_config[cell] += 1
        self.cells = list(self.cell_config.keys())
        self.tseitin_to_extpred = tseitin_to_extpred

        self.eu_config: Dict[Tuple[Cell, FrozenSet[Pred]],
                             int] = defaultdict(lambda: 0)
        self.eu_elements: Dict[Tuple[Cell,
                                     FrozenSet[Pred]], Set[int]] = defaultdict(set)
        for cell, num in self.cell_config.items():
            eetype = set()
            for tseitin, ext in tseitin_to_extpred.items():
                if not cell.is_positive(ext):
                    eetype.add(tseitin)
            key = (cell, frozenset(eetype))
            self.eu_config[key] = num
            self.eu_elements[key].update(self.cell_elements[cell])
        logger.debug('initial eu_config: %s', self.eu_elements)

    def all_satisfied(self) -> bool:
        return all(self.eu_config[(cell, frozenset())] == self.cell_config[cell]
                   for cell in self.cells)

    def select_eutype(self) -> Tuple[Cell, FrozenSet[Pred]]:
        assert not self.all_satisfied()
        for (cell, eetype), num in self.eu_config.items():
            if len(eetype) != 0 and num > 0:
                return cell, eetype

    def reduce_element(self, cell: Cell, eetype: FrozenSet[Pred]) -> int:
        self.cell_config[cell] -= 1
        self.eu_config[(cell, eetype)] -= 1
        element = self.eu_elements[(cell, eetype)].pop()
        return element

    def reduce_eetype(self, eetype: FrozenSet[Pred], ebtype: EBType,
                      ab_or_ba: int = 0) -> FrozenSet[Pred]:
        return eetype.difference(set(
            tseitin for tseitin in eetype
            if ebtype.is_positive(self.tseitin_to_extpred[tseitin])[ab_or_ba]
        ))

    def satisfied(self, eetype: FrozenSet[Pred], overall_eb_config: Dict[EBType, int]) \
            -> bool:
        if len(eetype) == 0:
            return True
        for ebtype, num in overall_eb_config.items():
            if num > 0:
                # ab_p
                eetype = self.reduce_eetype(eetype, ebtype)
            if len(eetype) == 0:
                return True
        return len(eetype) == 0

    def iter_eb_config(
        self,
        ebtype_weights: Dict[Cell, Dict[EBType, RingElement]]
    ):
        for raw_config in product(
                *list(multinomial(
                    len(ebtype_weights[cell]), num
                ) for (cell, _), num in self.eu_config.items())
        ):
            eb_config = defaultdict(dict)
            for i, (cell, eetype) in enumerate(self.eu_config.keys()):
                config = raw_config[i]
                for j, ebtype in enumerate(ebtype_weights[cell].keys()):
                    eb_config[(cell, eetype)][ebtype] = config[j]
            yield eb_config

    def reduce_eu_config(self, eb_config: Dict[Tuple[Cell, FrozenSet[Pred]],
                                               Tuple[int, ...]]) \
            -> FrozenSet[Tuple[Cell, FrozenSet[Pred], int]]:
        reduced_eu_config = deepcopy(self.eu_config)
        for (cell, eetype), config in eb_config.items():
            for ebtype, num in config.items():
                reduced_eetype = self.reduce_eetype(eetype, ebtype, ab_or_ba=1)
                reduced_eu_config[(cell, eetype)] -= num
                reduced_eu_config[(cell, reduced_eetype)] += num
        return frozenset(
            (*k, v) for k, v in reduced_eu_config.items() if v > 0
        )

    def sample_and_update(self, eb_config: Dict[Tuple[Cell, FrozenSet[Pred]],
                                                Tuple[int, ...]]) \
            -> Dict[EBType, List[int]]:
        ebtype_elements: Dict[EBType, List[int]] = defaultdict(list)
        updated_elements = dict()
        # sample
        for eutype, config in eb_config.items():
            idx = 0
            elements = list(self.eu_elements[eutype])
            # NOTE: we need to shuffle it again!
            random.shuffle(elements)
            updated_elements[eutype] = defaultdict(list)
            for ebtype, num in config.items():
                ebtype_elements[ebtype] += elements[idx:(idx + num)]
                reduced_eetype = self.reduce_eetype(
                    eutype[1], ebtype, ab_or_ba=1)
                sampled_elements = elements[idx:(idx + num)]
                updated_elements[eutype][reduced_eetype].extend(sampled_elements)
                idx += num

        # update
        for eutype, config in updated_elements.items():
            for reduced_eetype, elements in config.items():
                self.eu_elements[eutype].difference_update(elements)
                self.eu_elements[
                    (eutype[0], reduced_eetype)
                ].update(elements)
                num = len(elements)
                self.eu_config[eutype] -= num
                self.eu_config[(eutype[0], reduced_eetype)] += num
        logger.debug('update eu_config: %s', self.eu_elements)
        return ebtype_elements

    def __str__(self):
        s = ''
        for (cell, eetype), num in self.eu_config.items():
            s += 'Cell {}, {}: {}\n'.format(cell, list(eetype), num)
        return s

    def __repr__(self):
        return str(self)
