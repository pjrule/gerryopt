"""Bsic operations (get/set/length/shape) on n-dimensional arrays."""
from math import ceil
from typing import Optional, Union
from warnings import warning
from gerryopt.trace._expr import TracedExpr
from gerryopt.trace._assign import TracedAssign
from gerryopt.trace._constant import Constant, is_constant
from gerryopt.trace.types import (is_scalar, is_ndarray, scalar_type,
                                  size_intersection, type_union, type_product,
                                  make_ndarray, Scalar, NdArrayT, SizeBounds)


class NdArraySet(TracedAssign):
    """TODO -- immutable arrays"""


class NdArrayGet(TracedExpr):
    """Expression for retrieving ≥0 elements from an `ndarray`."""
    body: TracedExpr
    key: Optional[TracedExpr]
    start: Optional[TracedExpr]
    stop: Optional[TracedExpr]
    step: Optional[TracedExpr]

    def __init__(self,
                 body: TracedExpr,
                 key: Optional[TracedExpr],
                 start: Optional[TracedExpr] = None,
                 stop: Optional[TracedExpr] = None,
                 step: Optional[TracedExpr] = None):
        if any(v is not None and v.dtype != int
               for v in (key, start, stop, step)):
            raise TypeError('Array index expressions must be integer-valued.')
        self.body = body
        self.start = start
        self.stop = stop
        self.step = step

        type_lb = None
        for (t, ) in type_product(body.dtype):
            if not is_ndarray(t):
                raise TypeError('Cannot index into non-ndarray.')
            if t.dim != 1:
                raise NotImplementedError('TODO: multidimensional arrays.')
            if key is None:  # slice access
                size_bounds = slice_bounds(t, start, stop, step)
                type_lb = type_union(type_lb, make_ndarray(t,
                                                           size=size_bounds))
            else:  # single element access
                assert_scalar_bounds(t, key)
                type_lb = type_union(type_lb, scalar_type(t))
        self.dtype = type_lb


def expr_getitem(self: TracedExpr, key: Union[TracedExpr,
                                              slice]) -> NdArrayGet:
    """Tracing of indexing for generic expressions."""
    if isinstance(key, slice):
        return NdArrayGet(body=self,
                          key=None,
                          start=key.start,
                          stop=key.stop,
                          step=key.step)
    return NdArrayGet(body=self, key=key)


class NdArrayLength(TracedExpr):
    """Tracing of array length operations for generic expressions."""
    def __init__(self, body: TracedExpr):
        pass


# TODO: NdArraySize (for multidimensional arrays)


def assert_scalar_bounds(array_t: NdArrayT, key: TracedExpr) -> None:
    """Performs bounds checks for simple indexing operations on n-dim. arrays.

    Raises:
        IndexError: If an array of type `array_t` cannot possibly indexed
        by `key` at runtime.
    """
    if array_t.dim != 1:
        raise NotImplementedError('TODO: multidimensional arrays.')
    if array_t.size is not None and is_constant(key):
        size_lb, size_ub = array_t.size[0]
        if size_ub is not None:
            real_pos = size_ub.val - key.val if key.val < 0 else key.val
            if real_pos >= size_ub.val:
                raise IndexError(
                    f'Out-of-bounds indexing: array with size bounds [{size_lb.val}, '
                    f'{size_ub.val}] cannot be accessed at index {key.val}.')


def iceil(val: float) -> int:
    return int(ceil(val))


def slice_bounds(array_t: NdArrayT,
                 start: Optional[TracedExpr] = None,
                 stop: Optional[TracedExpr] = None,
                 step: Optional[TracedExpr] = None) -> Optional[SizeBounds]:
    """Finds bounds for indexing operations on n-dimensional arrays.

    Given an array type `array_t` and array indexing bounds `start` and `stop`,
    finds bounds on the size of the resulting array type. If `start` and/or `stop`
    are `Constant` expressions, and `step` is a `Constant` expression or `None`
    (indicating a step size of 1) the bounds may be tight. Otherwise, the bounds
    will not be tight.  

    Returns:
        Size bounds on the result of the indexing operation, if attainable.
    """
    # a reference on bounded array types:
    # https://suif.stanford.edu/suif/suif1/docs/suif_63.html
    if array_t.dim != 1:
        raise NotImplementedError('TODO: multidimensional arrays.')

    # Indexing patterns: [start:], [:stop], [start:stop], [start::step],
    # [::stop:step], [start:stop:step]
    # Refactoring note: these bounds are expressible as a smaller set of
    # if-expressions, but they're easier to interpret this way.
    # TODO: mixed bounds? (some of `start`, `stop`, and `step` are `Constant`
    # and some are expressions---how should that work?)
    size_lb, size_ub = array_t.size
    indexed_lb, indexed_ub = size_lb, size_ub
    if step is None:
        if is_constant(start) and stop is None:
            indexed_lb = max(0, size_lb - start.val)
            indexed_ub = None if size_ub is None else max(
                0, size_ub - start.val)
        elif start is None and is_constant(stop):
            indexed_lb = min(size_lb, stop.val)
            indexed_ub = None if size_ub is None else min(size_ub, stop.val)
        elif is_constant(start) and isinstance(stop):
            indexed_lb = max(0, size_lb - start.val)
            indexed_ub = None if size_ub is None else min(
                size_ub, min(0, stop.val - start.val + 1))
    elif is_constant(step):
        if is_constant(start) and stop is None:
            indexed_lb = max(0, (size_lb - start.val) // step.val)
            indexed_ub = (None if size_ub is None else max(
                0, iceil((size_ub - start.val) / step.val)))
        elif start is None and is_constant(stop):
            indexed_lb = min(size_lb, stop.val // step.val)
            indexed_ub = None if size_ub is None else min(
                size_ub, iceil(stop.val / step.val))
        elif is_constant(start) and isinstance(stop):
            indexed_lb = max(0, (size_lb - start.val) // step.val)
            indexed_ub = None if size_ub is None else (min(
                size_ub, min(0, iceil(1 + (stop.val - start.val) / step.val))))

    # TODO (when bored): verify bounds math, add support for negative indexing.
    if any(is_constant(v) and v.val < 0 for v in (start, stop, step)):
        warning("Haven't gotten around to implementing negative indexing yet. "
                "Using loose bounds...")
        return [(size_lb, size_ub)]
    return [(indexed_lb, indexed_ub)]
