"""Tracing for function calls."""
from typing import Iterable
from gerryopt.trace._expr import TracedExpr
from gerryopt.trace.opcodes import BasisOpcode


class CallExpr(TracedExpr):
    """A generic function call expression."""
    args: Iterable[TracedExpr]


class BasisCall(CallExpr):
    """A function call expression for functions in the standard basis."""
    op: BasisOpcode

    def __init__(self, dtype: type, op: BasisOpcode,
                 args: Iterable[TracedExpr]):
        self.dtype = dtype
        self.op = op
        self.args = args


class UserDefinedCall(CallExpr):
    """A function call expression for user-defined functions."""
    function_id: str  # TODO: pointer to function object instead?

    def __init__(self, dtype: type, function_id: str,
                 args: Iterable[TracedExpr]):
        self.dtype = dtype
        self.function_id = function_id
        self.args = args
