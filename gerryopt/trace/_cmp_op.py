"""Tracing for comparison expressions."""
from itertools import product
from typing import Union
from gerryopt.trace._expr import TracedExpr
from gerryopt.trace._constant import Constant, coerce_constants
from gerryopt.trace.opcodes import (CmpOpcode, CMP_OPCODE_TO_REPR,
                                    CMP_OPCODE_TO_METHOD_NAME)
from gerryopt.trace.types import (is_scalar, is_ndarray, is_possibly_ndarray,
                                  scalar_type, size_intersection, type_union,
                                  type_product, make_ndarray, binary_broadcast,
                                  Scalar)

Val = Union[TracedExpr, Scalar]


class CmpOp(TracedExpr):
    """A binary operation expression."""
    left: TracedExpr
    right: TracedExpr
    op: CmpOpcode

    def __init__(self, left: Val, right: Val, op: CmpOpcode):
        self.left, self.right = coerce_constants(left, right)
        self.op = op
        self.dtype = None
        for (lhs, rhs) in type_product(self.left.dtype, self.right.dtype):
            self.dtype = type_union(self.dtype,
                                    binary_broadcast(bool, lhs, rhs))

    def __repr__(self):
        opcode_repr = CMP_OPCODE_TO_REPR[self.op]
        return f'CmpOp({opcode_repr}, {self.left}, {self.right})'


# Dynamically inject comparison tracing into generic expressions.
for op, name in CMP_OPCODE_TO_METHOD_NAME.items():
    setattr(TracedExpr,
            f'__{name}__',
            lambda self, other, _op=op: CmpOp(self, other, _op))
