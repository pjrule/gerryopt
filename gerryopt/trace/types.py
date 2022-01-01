"""Basic types and type utility functions for tracing."""
from typing import (Union, Any, Iterable, Optional, List, Tuple, get_args,
                    get_origin)
from itertools import product
from gerryopt.trace._expr import TracedExpr


SCALAR_TYPES = [int, float, bool]
Scalar = Union[int, float, bool]
Val = Union[TracedExpr, Scalar]


def type_product(*args: type) -> Iterable:
    """Generates the Cartesian product of types."""
    unrolled = [get_args(t) if get_origin(t) is Union else (t, ) for t in args]
    return product(*unrolled)


def type_union(*args: type) -> Optional[type]:
    """Finds the union of types, eliminating `None` where possible."""
    union_t = None
    for t in args:
        if union_t is None:
            union_t = t
        elif t is not None:
            union_t = Union[union_t, t]
    return union_t


NdArrayT = type('ndarray', (), {})
SizeBounds = List[Tuple[int, Optional[int]]]


def make_ndarray(scalar_type: type,
                 dim: int = 1,
                 size: Optional[SizeBounds] = None) -> NdArrayT:
    """Makes a specialized ndarray type with optional size bounds."""
    if dim != 1:
        raise NotImplementedError('TODO: multidimensional arrays.')

    attrs = {'dim': dim, 'size': size}
    if size is not None:
        size_label = '_'.join(
            f'{"∞" if lb is None else lb}_{"∞" if ub is None else ub}'
            for lb, ub in size)
        return type(f'ndarray__{dim}__{size_label}', (NdArrayT, ), attrs)
    elif any(lb is None for lb, _ in size):
        raise TypeError('Cannot have infinite lower bound on ndarray size.')
    return type(f'ndarray__{dim}', (NdArrayT, ), attrs)


def size_intersection(a: NdArrayT, b: NdArrayT) -> Optional[SizeBounds]:
    """Finds intersection (if possible) between sizes of ndarray types.

    Raises:
        TypeError: If an intersection cannot be found for a dimension.
    """
    if a.dim != b.dim:
        raise NotImplementedError(
            'TODO: broadcasting for multidimensional arrays.')

    a_bounds = getattr(a, 'size')
    b_bounds = getattr(b, 'size')
    if a_bounds is None or b_bounds is None:
        return None
    inter_bounds = []
    for (a_lb, a_ub), (b_lb, b_ub), in zip(a_bounds, b_bounds):
        inter_lb = max(a_lb, b_lb)
        inter_ub = None if (a_ub is None or b_ub is None) else min(a_ub, b_ub)
        if inter_ub is not None and inter_lb > inter_ub:
            raise TypeError(
                f'Cannot find size intersection of bounds [{a_lb}, {a_ub}] '
                f'and [{b_lb}, {b_ub}].')
        else:
            inter_bounds.append((inter_lb, inter_ub))
    return inter_bounds


def is_scalar(dtype: Any) -> bool:
    """Determines if a type is scalar."""
    return dtype in get_args(Scalar)


def is_ndarray(dtype: Any) -> bool:
    """Determines if a type is a (specialized) ndarray."""
    return issubclass(dtype, NdArrayT)


def scalar_type(dtype: Any) -> type:
    """Coerces ndarray types to scalar types (identity on scalar types).

    Raises:
        TypeError: if `dtype` is not a scalar or ndarray type.
    """
    if is_scalar(dtype):
        return dtype
    if issubclass(dtype, NdArrayT):
        return dtype.scalar_type
    raise TypeError(f'Cannot coerce {dtype} to scalar type.')