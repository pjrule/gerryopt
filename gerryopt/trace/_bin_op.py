"""Tracing for binary operation expressions."""
from enum import Enum
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import (Union, Any, Iterable, Optional, List, Tuple, get_args,
                    get_origin)
from itertools import product
from gerryopt.trace._expr import TracedExpr
from gerryopt.trace._constant import Constant
from gerryopt.trace.opcodes import BinOpcode, BIN_OPCODE_TO_METHOD_NAME
from gerryopt.trace.types import (Val, is_scalar, is_ndarray, size_intersection, type_union, type_product, make_ndarray)


class BinOp(TracedExpr):
    left: TracedExpr
    right: TracedExpr
    op: BinOpcode

    REAL_OPCODES = {
        BinOpcode.ADD, BinOpcode.SUB, BinOpcode.MUL, BinOpcode.DIV,
        BinOpcode.FLOOR_DIV, BinOpcode.MOD, BinOpcode.POW
    }
    BIT_OPCODES = {
        BinOpcode.L_SHIFT, BinOpcode.R_SHIFT, BinOpcode.BIT_OR,
        BinOpcode.BIT_XOR, BinOpcode.BIT_AND
    }

    def __init__(self, left: Val, right: Val, op: BinOpcode):
        if is_scalar(type(left)):
            left = Constant(left)
        if is_scalar(type(right)):
            right = Constant(right)

        self.left = left
        self.right = right
        self.op = op

        type_lb = None
        for (lhs, rhs) in type_product(left.dtype, right.dtype):
            lhs_scalar = scalar_type(lhs)
            rhs_scalar = scalar_type(rhs)
            if op in BinOp.REAL_OPCODES:
                # In general, we have:
                #   {float, int, bool} * float -> float
                #   float * {float, int, bool} -> float
                #   {int, bool} * {int, bool} -> int
                # There are a few exceptions to mirror NumPy semantics.
                #   * Subtraction of boolean ndarrays is not permitted.
                #   * DIV: {float, int, bool} * {float, int, bool} -> float
                #   * FLOOR_DIV, MOD, POW: {bool, int} * {bool, int} -> int
                if (is_ndarray(lhs) or is_ndarray(rhs)) and op == BinOpcode.SUB:
                    raise TypeError(
                        'Subtraction of boolean ndarrays is not permitted.')

                # Determine elementwise type.
                op_scalar = None
                if op == BinOpcode.DIV or lhs_scalar == float or rhs_scalar == float:
                    op_scalar = float
                else:
                    op_scalar = int
            elif op in BinOp.BIT_OPCODES:
                # {int, bool} * int -> int
                # int * {int, bool} -> int
                # bool * bool -> bool
                # (except for bool * bool -> int for lshift, rshift)
                # Bitwise operations are not supported on floats.
                if lhs_scalar == float or rhs_scalar == float:
                    raise TypeError(
                        'Bitwise operations are not supported on floats.')
                if (lhs_scalar == int or rhs_scalar == int or op in (BinOpcode.L_SHIFT, BinOpcode.R_SHIFT)):
                    op_scalar = int
                else:
                    op_scalar = bool
            elif op == BinOp.MAT_MUL:
                # {int, bool, float} * float -> float
                # float * {int, bool}  -> float
                # {int, bool} * int -> int
                # bool * int -> int
                # bool * bool -> bool

                # check (approximate) compatibility.
                size_intersection(lhs.dtype.size, rhs.dtype.size)  
                if lhs_scalar == float or rhs_scalar == float:
                    op_scalar = float
                elif lhs_scalar == int or rhs_scalar == int:
                    op_scalar = int
                else:
                    op_scalar = bool
                type_lb = type_union(type_lb, op_scalar)
                continue # normal broadcasting rules don't apply
            else:
                raise ValueError(f'Unknown binary operation {op}.')

            # Apply broadcasting rules (non-matrix multiplication operations).
            if is_ndarray(lhs) and is_ndarray(rhs):
                # TODO: handle >1-dimensional arrays.
                inter_size = size_intersection(lhs.dtype.size, rhs.dtype.size)
                type_lb = type_union(type_lb,
                                     make_ndarray(op_scalar, size=inter_size))
            elif is_ndarray(lhs):
                type_lb = type_union(
                    type_lb, make_ndarray(op_scalar, type=lhs.dtype.size))
            elif is_ndarray(rhs):
                type_lb = type_union(
                    type_lb, make_ndarray(op_scalar, type=rhs.dtype.size))
            else:
                type_lb = type_union(type_lb, op_scalar)
        self.dtype = type_lb

    def __repr__(self):
        return f'BinOp[{self.op}, {self.left}, {self.right}]'


# Dynamically inject binary operation tracing into generic expressions.
for opcode, name in BIN_OPCODE_TO_METHOD_NAME.items():
    setattr(TracedExpr, f'__{name}__', lambda self, other: BinOp(self, other, opcode))
    setattr(TracedExpr, f'__r{name}__', lambda self, other: BinOp(other, self, opcode))

def _expr_pow(self, other: Val, modulo=None) -> BinOp:
    if modulo is not None:
        raise NotImplementedError('Modulo not supported.')
    return BinOp(self, other, BinOpcode.POW)

def _expr_rpow(self, other: Val, modulo=None) -> BinOp:
    if modulo is not None:
        raise NotImplementedError('Modulo not supported.')
    return BinOp(other, self, BinOpcode.POW) 

TracedExpr.__pow__ = _expr_pow
TracedExpr.__rpow__ = _expr_rpow