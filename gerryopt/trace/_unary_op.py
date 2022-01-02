"""Tracing for unary operation expressions."""
from itertools import product
from gerryopt.trace._expr import TracedExpr
from gerryopt.trace._constant import Constant
from gerryopt.trace.opcodes import (UnaryOpcode, UNARY_OPCODE_TO_REPR,
                                    UNARY_OPCODE_TO_METHOD_NAME)
from gerryopt.trace.types import (is_scalar, is_ndarray, scalar_type,
                                  size_intersection, type_union, type_product,
                                  make_ndarray, Scalar)


class UnaryOp(TracedExpr):
    """A unary operation expression."""
    operand: TracedExpr
    op: UnaryOpcode

    OP_TYPES = {
        (UnaryOpcode.UADD, float): float,
        (UnaryOpcode.USUB, float): float,
        # Invert not supported on floats
        (UnaryOpcode.NOT, float): bool,
        (UnaryOpcode.UADD, int): int,
        (UnaryOpcode.USUB, int): int,
        (UnaryOpcode.INVERT, int): int,
        (UnaryOpcode.NOT, int): bool,
        (UnaryOpcode.UADD, bool): int,
        (UnaryOpcode.USUB, bool): int,
        (UnaryOpcode.INVERT, bool): int,
        (UnaryOpcode.NOT, bool): bool,
    }

    def __init__(self, operand: TracedExpr, op: UnaryOpcode):
        self.operand = operand
        self.op = op

        type_lb = None
        for (t, ) in type_product(operand.dtype):
            t_scalar = scalar_type(t)
            try:
                expr_type = UnaryOp.OP_TYPES[(op, t_scalar)]
            except KeyError:
                raise TypeError(
                    f'Unary operation {op} not supported for type {t}.')

            if is_ndarray(t):
                type_lb = type_union(
                    type_lb, make_ndarray(expr_type, size=t.dtype.size))
            else:
                type_lb = type_union(expr_type, type_lb)
        self.dtype = type_lb

    def __repr___(self):
        return f'UnaryOp({UNARY_OPCODE_TO_REPR[self.op]}, {self.operand})'


# Dynamically inject unary operation tracing into generic expressions.
for op, name in UNARY_OPCODE_TO_METHOD_NAME.items():
    setattr(TracedExpr, f'__{name}__', lambda self, _op=op: UnaryOp(self, _op))


# Special case: the `not` operator cannot be overloaded---intercepting
# its behavior requires an AST rewrite. Users can either use an AST
# rewriter or just use the syntax `<expr>.not_()` instead of `not <expr>`.
def _expr_not(self: TracedExpr) -> UnaryOp:
    """Boolean NOT of a traced expression."""
    return UnaryOp(self, UnaryOpcode.NOT)


TracedExpr.not_ = _expr_not
