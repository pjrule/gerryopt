"""Vector operations for the GerryOpt DSL."""
from typing import List, TypeVar, Generic
from statistics import mean

V = TypeVar('V')


class Vec(Generic[V]):
    """A GerryOpt vector."""
    def __init__(self, vals: List[V]):
        self._vals = vals

    def min(self) -> V:
        return min(self._vals)

    def max(self) -> V:
        return max(self._vals)

    def sum(self) -> V:
        return sum(self._vals)

    def mean(self) -> float:
        return mean(self._vals)

    def count(self) -> int:
        return len(self._vals)
