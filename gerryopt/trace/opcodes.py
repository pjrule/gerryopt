"""Opcodes for basic operations."""
import operator
from enum import Enum

# Opcode definitions.
BinOpcode = Enum(
    'BinOpcode',
    'ADD SUB MUL DIV FLOOR_DIV MOD POW L_SHIFT R_SHIFT BIT_OR BIT_XOR BIT_AND MAT_MUL'
)
UnaryOpcode = Enum('UnaryOpcode', 'UADD USUB INVERT NOT')
BoolOpcode = Enum('BoolOpcode', 'AND OR')
CmpOpcode = Enum('CmpOpcode', 'EQ NOT_EQ LT LTE GT GTE')

# Opcodes to operator functions.
BIN_OPCODE_TO_FN = {
    BinOpcode.ADD: operator.add,
    BinOpcode.SUB: operator.sub,
    BinOpcode.MUL: operator.mul,
    BinOpcode.DIV: operator.truediv,
    BinOpcode.FLOOR_DIV: operator.floordiv,
    BinOpcode.MOD: operator.mod,
    BinOpcode.POW: operator.pow,
    BinOpcode.L_SHIFT: operator.lshift,
    BinOpcode.R_SHIFT: operator.rshift,
    BinOpcode.BIT_OR: operator.or_,
    BinOpcode.BIT_XOR: operator.xor,
    BinOpcode.BIT_AND: operator.and_,
    BinOpcode.MAT_MUL: operator.matmul
}
UNARY_OPCODE_TO_FN = {
    UnaryOpcode.UADD: operator.pos,
    UnaryOpcode.USUB: operator.neg,
    UnaryOpcode.INVERT: operator.inv,
    UnaryOpcode.NOT: operator.not_
}
OPCODE_TO_FN = {**BIN_OPCODE_TO_FN, **UNARY_OPCODE_TO_FN}

# Opcodes to abbreviated "magic method" names, where applicable (e.g. __add__)
BIN_OPCODE_TO_METHOD_NAME = {
    BinOpcode.ADD: 'add',
    BinOpcode.SUB: 'sub',
    BinOpcode.MUL: 'mul',
    BinOpcode.DIV: 'truediv',
    BinOpcode.FLOOR_DIV: 'floordiv',
    BinOpcode.MOD: 'mod',
    BinOpcode.POW: 'pow',
    BinOpcode.L_SHIFT: 'lshift',
    BinOpcode.R_SHIFT: 'rshift',
    BinOpcode.BIT_OR: 'or',
    BinOpcode.BIT_XOR: 'xor',
    BinOpcode.BIT_AND: 'and',
    BinOpcode.MAT_MUL: 'matmul'
}
OPCODE_TO_METHOD_NAME = BIN_OPCODE_TO_METHOD_NAME

# Opcodes to human-readable operator representations.
BIN_OPCODE_TO_REPR = {
    BinOpcode.ADD: '+',
    BinOpcode.SUB: '-',
    BinOpcode.MUL: '*',
    BinOpcode.DIV: '/',
    BinOpcode.FLOOR_DIV: '//',
    BinOpcode.MOD: '%',
    BinOpcode.POW: '**',
    BinOpcode.L_SHIFT: '<<',
    BinOpcode.R_SHIFT: '>>',
    BinOpcode.BIT_OR: '|',
    BinOpcode.BIT_XOR: '^',
    BinOpcode.BIT_AND: '&',
    BinOpcode.MAT_MUL: '@'
}
UNARY_OPCODE_TO_REPR = {
    UnaryOpcode.UADD: '+',
    UnaryOpcode.USUB: '-',
    UnaryOpcode.INVERT: '~',
    UnaryOpcode.NOT: 'not '
}
OPCODE_TO_REPR = {**BIN_OPCODE_TO_REPR, **UNARY_OPCODE_TO_REPR}
