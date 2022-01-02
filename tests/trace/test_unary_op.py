"""Tests for traced unary operations."""
import pytest
import operator
from typing import Callable, Generator, Tuple
from gerryopt.trace import Constant, UnaryOp
from gerryopt.trace.opcodes import (UnaryOpcode, UNARY_OPCODE_TO_FN,
                                    UNARY_OPCODE_TO_REPR)
from gerryopt.trace.types import SCALAR_TYPES, type_product, type_union

# `not` cannot be overloaded.
OVERLOADED_OPS = [UnaryOpcode.UADD, UnaryOpcode.USUB, UnaryOpcode.INVERT]


def valid_scalar_unary_ops(
) -> Generator[Tuple[UnaryOpcode, type, type], None, None]:
    """Generates all combinations of valid unary operations on scalars.

    Yields:
        3-tuples of form (opcode, operand type, result type).
    """
    for opcode in OVERLOADED_OPS:
        operator_fn = UNARY_OPCODE_TO_FN[opcode]
        for operand_type in SCALAR_TYPES:
            try:
                result_type = type(operator_fn(operand_type(1)))
                yield opcode, operand_type, result_type
            except TypeError:
                pass


def invalid_scalar_unary_ops(
) -> Generator[Tuple[UnaryOpcode, type], None, None]:
    """Generates all combinations of invalid unary operations on scalars.

    Yields:
        2-tuples of form (opcode, operand type).
    """
    for opcode in OVERLOADED_OPS:
        operator_fn = UNARY_OPCODE_TO_FN[opcode]
        for operand_type in SCALAR_TYPES:
            try:
                type(operator_fn(operand_type(1)))
            except TypeError:
                yield opcode, operand_type


def unary_op_case_id(case):
    opcode, operand_type = case[:2]
    case_name = ' '.join([
        operand_type.__name__, UNARY_OPCODE_TO_REPR[opcode],
        operand_type.__name__
    ])
    if len(case) == 3:
        case_name += f' -> {case[2].__name__}'
    return case_name


@pytest.mark.parametrize('typed_op',
                         valid_scalar_unary_ops(),
                         ids=unary_op_case_id)
def test_unary_ops_valid_constant(typed_op):
    opcode, operand_type, expected_type = typed_op
    operand = Constant(operand_type(1))
    operator_fn = UNARY_OPCODE_TO_FN[opcode]

    result = operator_fn(operand)
    assert isinstance(result, UnaryOp)
    assert result.operand is operand
    assert result.op == opcode
    assert result.dtype == expected_type


@pytest.mark.parametrize('typed_op',
                         invalid_scalar_unary_ops(),
                         ids=unary_op_case_id)
def test_unary_ops_invalid_constant(typed_op):
    opcode, operand_type = typed_op
    operand = Constant(operand_type(1))
    operator_fn = UNARY_OPCODE_TO_FN[opcode]

    with pytest.raises(TypeError):
        operator_fn(operand)


@pytest.mark.parametrize('operand_type', SCALAR_TYPES)
def test_unary_not_valid_constant(operand_type):
    operand = Constant(operand_type(1))
    result = operand.not_()

    assert isinstance(result, UnaryOp)
    assert result.operand is operand
    assert result.op == UnaryOpcode.NOT
    assert result.dtype == bool
