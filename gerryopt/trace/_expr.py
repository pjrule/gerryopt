"""Tracing-based JIT compilation for GerryChain operations."""
from abc import abstractmethod, ABC
from typing import Any
from gerryopt.trace.types import NdArrayT, Scalar


def is_expr(expr: Any) -> bool:
    return isinstance(expr, TracedExpr)


class TracedExpr(ABC):
    """A generic traced expresssion."""
    dtype: type

    @abstractmethod
    def __init__(self, expr: 'TracedExpr'):
        pass

    @property
    def is_scalar(self):
        return isinstance(self.dtype, Scalar)

    @property
    def is_ndarray(self):
        return issubclass(self.dtype, NdArrayT)