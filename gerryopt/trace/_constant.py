"""Tracing for generic expressions."""
from gerryopt.trace._expr import TracedExpr
from gerryopt.trace.types import is_scalar, Scalar


class Constant(TracedExpr):
    """A traced constant expression."""
    val: Scalar

    def __init__(self, val: Scalar):
        if not is_scalar(type(val)):
            raise TypeError('Constants must be of type int, float, or bool.')
        self.val = val
        self.dtype = type(val)

    def __repr__(self):
        return f'Constant({self.dtype.__name__}, {self.val})'

    def __hash__(self):
        return hash(self.val)


def coerce_constants(*args):
    """Coerces scalar values (bool, int, float) to `Constant` expressions."""
    return (Constant(val) if is_scalar(type(val)) else val for val in args)


def is_constant(val) -> bool:
    return isinstance(val, Constant)