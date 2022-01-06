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

ARRAY_INIT_OPS = 'NDARRAY_ZEROS NDARRAY_ONES NDARRAY_ARANGE NDARRAY_LINSPACE'
REDUCE_OPS = 'MIN MAX SUM MEAN MEDIAN MODE PERCENTILE'
BasisOpcode = Enum('BasisOpcode', ' '.join([ARRAY_INIT_OPS, REDUCE_OPS]))

# Opcodes to operator functions, where applicable.
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
CMP_OPCODE_TO_FN = {
    CmpOpcode.EQ: operator.eq,
    CmpOpcode.NOT_EQ: operator.ne,
    CmpOpcode.LT: operator.lt,
    CmpOpcode.LTE: operator.le,
    CmpOpcode.GT: operator.gt,
    CmpOpcode.GTE: operator.ge
}

OPCODE_TO_FN = {**BIN_OPCODE_TO_FN, **UNARY_OPCODE_TO_FN, **CMP_OPCODE_TO_FN}

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
UNARY_OPCODE_TO_METHOD_NAME = {
    UnaryOpcode.UADD: 'pos',
    UnaryOpcode.USUB: 'neg',
    UnaryOpcode.INVERT: 'invert',
    # `not` is intrinsic (needs AST rewrite)
}
CMP_OPCODE_TO_METHOD_NAME = {
    CmpOpcode.EQ: 'eq',
    CmpOpcode.NOT_EQ: 'ne',
    CmpOpcode.LT: 'lt',
    CmpOpcode.LTE: 'le',
    CmpOpcode.GT: 'gt',
    CmpOpcode.GTE: 'ge'
}
OPCODE_TO_METHOD_NAME = {
    **BIN_OPCODE_TO_METHOD_NAME,
    **UNARY_OPCODE_TO_METHOD_NAME,
    **CMP_OPCODE_TO_METHOD_NAME
}

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
    UnaryOpcode.NOT: 'not'
}
CMP_OPCODE_TO_REPR = {
    CmpOpcode.EQ: '==',
    CmpOpcode.NOT_EQ: '!=',
    CmpOpcode.LT: '<',
    CmpOpcode.LTE: '<=',
    CmpOpcode.GT: '>',
    CmpOpcode.GTE: '>='
}
OPCODE_TO_REPR = {
    **BIN_OPCODE_TO_REPR,
    **UNARY_OPCODE_TO_REPR,
    **CMP_OPCODE_TO_REPR
}
