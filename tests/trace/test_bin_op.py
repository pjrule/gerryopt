"""Tests for traced binary operations."""
import pytest
import operator
from typing import Callable, Generator, Tuple
from gerryopt.trace import Constant, BinOp
from gerryopt.trace.opcodes import (BinOpcode, BIN_OPCODE_TO_FN,
                                    BIN_OPCODE_TO_REPR)
from gerryopt.trace.types import SCALAR_TYPES, type_product, type_union

SCALAR_BIN_OPS = BinOp.REAL_OPCODES | BinOp.BIT_OPCODES


def valid_scalar_bin_ops(
) -> Generator[Tuple[BinOpcode, type, type, type], None, None]:
    """Generates all combinations of valid binary operations on scalars.

    Yields:
        4-tuples of form (opcode, lhs type, rhs type, result type).
    """
    for opcode in SCALAR_BIN_OPS:
        operator_fn = BIN_OPCODE_TO_FN[opcode]
        for lhs_type in SCALAR_TYPES:
            for rhs_type in SCALAR_TYPES:
                try:
                    result_type = type(operator_fn(lhs_type(1), rhs_type(1)))
                    yield opcode, lhs_type, rhs_type, result_type
                except TypeError:
                    pass


def invalid_scalar_bin_ops(
) -> Generator[Tuple[BinOpcode, type, type], None, None]:
    """Generates all combinations of invalid binary operations on scalars.

    Yields:
        3-tuples of form (operator function, lhs type, rhs type).
    """
    for opcode in SCALAR_BIN_OPS:
        operator_fn = BIN_OPCODE_TO_FN[opcode]
        for lhs_type in SCALAR_TYPES:
            for rhs_type in SCALAR_TYPES:
                try:
                    type(operator_fn(lhs_type(1), rhs_type(1)))
                except TypeError:
                    yield opcode, lhs_type, rhs_type


def bin_op_case_id(case):
    opcode, lhs_type, rhs_type = case[:3]
    case_name = ' '.join(
        [lhs_type.__name__, BIN_OPCODE_TO_REPR[opcode], rhs_type.__name__])
    if len(case) == 4:
        case_name += f' -> {case[3].__name__}'
    return case_name


@pytest.mark.parametrize('typed_op',
                         valid_scalar_bin_ops(),
                         ids=bin_op_case_id)
def test_bin_ops_valid_two_constants(typed_op):
    opcode, lhs_type, rhs_type, expected_type = typed_op
    lhs = Constant(lhs_type(1))
    rhs = Constant(rhs_type(1))
    operator_fn = BIN_OPCODE_TO_FN[opcode]

    result = operator_fn(lhs, rhs)
    assert isinstance(result, BinOp)
    assert result.left is lhs
    assert result.right is rhs
    assert result.op == opcode
    assert result.dtype == expected_type


@pytest.mark.parametrize('typed_op',
                         invalid_scalar_bin_ops(),
                         ids=bin_op_case_id)
def test_bin_ops_invalid_two_constants(typed_op):
    opcode, lhs_type, rhs_type = typed_op
    lhs = Constant(lhs_type(1))
    rhs = Constant(rhs_type(1))
    operator_fn = BIN_OPCODE_TO_FN[opcode]

    with pytest.raises(TypeError):
        operator_fn(lhs, rhs)


@pytest.mark.parametrize('typed_op',
                         valid_scalar_bin_ops(),
                         ids=bin_op_case_id)
def test_bin_ops_valid_constant_lhs_primitive_rhs(typed_op):
    opcode, lhs_type, rhs_type, expected_type = typed_op
    lhs = Constant(lhs_type(1))
    rhs = rhs_type(1)
    operator_fn = BIN_OPCODE_TO_FN[opcode]

    result = operator_fn(lhs, rhs)
    assert isinstance(result, BinOp)
    assert result.left is lhs
    assert isinstance(result.right, Constant)
    assert result.right.val == rhs
    assert result.right.dtype == rhs_type
    assert result.op == opcode
    assert result.dtype == expected_type


@pytest.mark.parametrize('typed_op',
                         invalid_scalar_bin_ops(),
                         ids=bin_op_case_id)
def test_bin_ops_invalid_constant_lhs_primitive_rhs(typed_op):
    opcode, lhs_type, rhs_type = typed_op
    lhs = Constant(lhs_type(1))
    rhs = rhs_type(1)
    operator_fn = BIN_OPCODE_TO_FN[opcode]

    with pytest.raises(TypeError):
        operator_fn(lhs, rhs)


@pytest.mark.parametrize('typed_op',
                         valid_scalar_bin_ops(),
                         ids=bin_op_case_id)
def test_bin_ops_valid_primitive_lhs_constant_rhs(typed_op):
    opcode, lhs_type, rhs_type, expected_type = typed_op
    lhs = lhs_type(1)
    rhs = Constant(rhs_type(1))
    operator_fn = BIN_OPCODE_TO_FN[opcode]

    result = operator_fn(lhs, rhs)
    assert isinstance(result, BinOp)
    assert isinstance(result.left, Constant)
    assert result.left.val == lhs
    assert result.left.dtype == lhs_type
    assert result.right is rhs
    assert result.op == opcode
    assert result.dtype == expected_type


@pytest.mark.parametrize('typed_op',
                         invalid_scalar_bin_ops(),
                         ids=bin_op_case_id)
def test_bin_ops_invalid_primitive_lhs_constant_rhs(typed_op):
    opcode, lhs_type, rhs_type = typed_op
    lhs = lhs_type(1)
    rhs = Constant(rhs_type(1))
    operator_fn = BIN_OPCODE_TO_FN[opcode]

    with pytest.raises(TypeError):
        operator_fn(lhs, rhs)
