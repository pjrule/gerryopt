"""Tests for traced comparison operations."""
import pytest
from gerryopt.trace import Constant, CmpOp
from gerryopt.trace.opcodes import (CmpOpcode, CMP_OPCODE_TO_FN,
                                    CMP_OPCODE_TO_REPR)
from gerryopt.trace.types import SCALAR_TYPES, type_product, type_union


@pytest.mark.parametrize('lhs_type', SCALAR_TYPES)
@pytest.mark.parametrize('opcode',
                         CmpOpcode,
                         ids=lambda op: CMP_OPCODE_TO_REPR[op])
@pytest.mark.parametrize('rhs_type', SCALAR_TYPES)
def test_cmp_ops_two_constants(lhs_type, opcode, rhs_type):
    lhs = Constant(lhs_type(1))
    rhs = Constant(rhs_type(1))
    operator_fn = CMP_OPCODE_TO_FN[opcode]

    result = operator_fn(lhs, rhs)
    assert isinstance(result, CmpOp)
    assert result.left is lhs
    assert result.right is rhs
    assert result.op == opcode
    assert result.dtype == bool


@pytest.mark.parametrize('lhs_type', SCALAR_TYPES)
@pytest.mark.parametrize('opcode',
                         CmpOpcode,
                         ids=lambda op: CMP_OPCODE_TO_REPR[op])
@pytest.mark.parametrize('rhs_type', SCALAR_TYPES)
def test_cmp_ops_constant_lhs_primitive_rhs(lhs_type, opcode, rhs_type):
    lhs = Constant(lhs_type(1))
    rhs = Constant(rhs_type(1))
    operator_fn = CMP_OPCODE_TO_FN[opcode]

    result = operator_fn(lhs, rhs)
    assert isinstance(result, CmpOp)
    assert result.left is lhs
    assert isinstance(result.right, Constant)
    assert result.right.val == rhs
    assert result.right.dtype == rhs_type
    assert result.op == opcode
    assert result.dtype == bool


@pytest.mark.parametrize('lhs_type', SCALAR_TYPES)
@pytest.mark.parametrize('opcode',
                         CmpOpcode,
                         ids=lambda op: CMP_OPCODE_TO_REPR[op])
@pytest.mark.parametrize('rhs_type', SCALAR_TYPES)
def test_cmp_ops_primitive_lhs_constant_rhs(lhs_type, opcode, rhs_type):
    lhs = Constant(lhs_type(1))
    rhs = Constant(rhs_type(1))
    operator_fn = CMP_OPCODE_TO_FN[opcode]

    result = operator_fn(lhs, rhs)
    assert isinstance(result, CmpOp)
    assert isinstance(result.left, Constant)
    assert result.left.val == lhs
    assert result.left.dtype == lhs_type
    assert result.right is rhs
    assert result.op == opcode
    assert result.dtype == bool
