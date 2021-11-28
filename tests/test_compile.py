import ast
import inspect
import pytest
import networkx as nx
from copy import copy
from textwrap import dedent
from inspect import getclosurevars as get_ctx
from typing import Callable, Union, get_args
from gerryopt.compile import (
    BoolOp, BoolOpcode, IfExpr, LoadedNamesVisitor, UndefinedVar, ctx_union,
    defined_type_product, is_vec, preprocess_ast, scalar_type,
    type_graph_column, tally_columns, CompileError, to_ast, load_function_ast,
    type_union, type_updater_columns, preprocess_ast, DSLValidationVisitor,
    AssignmentNormalizer, find_names, always_returns, type_and_transform_expr,
    Constant, Name, UnaryOp, UnaryOpcode, is_truthy)
from gerryopt.vector import Vec
from gerrychain.updaters import Tally, cut_edges
from gerrychain.grid import create_grid_graph
from itertools import product

SIMPLE_PRIMITIVE_TYPES = [
    int,
    bool,
    float,
]

PRIMITIVE_TYPE_UNIONS = [
    Union[int, bool],
    Union[int, float],
    Union[bool, float],
    Union[int, bool, float],
]

SIMPLE_VEC_TYPES = [Vec[int], Vec[bool], Vec[float]]

VEC_TYPE_UNIONS = [
    Union[Vec[int], Vec[bool]],
    Union[Vec[int], Vec[float]],
    Union[Vec[bool], Vec[float]],
    Union[Vec[int], Vec[bool], Vec[float]],
]

VEC_TYPES = SIMPLE_VEC_TYPES + VEC_TYPE_UNIONS

PRIMITIVE_TYPES = SIMPLE_PRIMITIVE_TYPES + PRIMITIVE_TYPE_UNIONS

TRUTHY_TYPES = PRIMITIVE_TYPES

SIMPLE_TYPES = SIMPLE_PRIMITIVE_TYPES + SIMPLE_VEC_TYPES

TYPE_UNIONS = PRIMITIVE_TYPE_UNIONS + VEC_TYPE_UNIONS


@pytest.fixture
def grid_with_attrs():
    graph = create_grid_graph((6, 6), False)
    nx.set_node_attributes(graph, 1.0, 'area')
    nx.set_node_attributes(graph, True, 'flag')
    nx.set_node_attributes(graph, '', 'string')
    return graph


def always_accept(partition):
    return True


def always_accept_with_store(partition, store):
    return True


def fn_to_ast(fn: Callable):
    """Helper for generating ASTs of test functions."""
    return ast.parse(dedent(inspect.getsource(fn))).body[0]


def ast_equal(a: ast.AST, b: ast.AST):
    """Compares two ASTs of two function."""
    # TODO: comparing ASTs is apparently a rather subtle business.
    # For now, we use the hack of comparing AST dumps, but this may
    # cause false negatives in some cases.
    # (see https://stackoverflow.com/q/3312989)
    if isinstance(a, ast.Module):
        assert len(a.body) == len(b.body) == 1
        assert isinstance(a.body[0], ast.FunctionDef)
        assert isinstance(b.body[0], ast.FunctionDef)
        a = copy(a)
        b = copy(b)
        a.body[0].name = ''
        b.body[0].name = ''
        return ast.dump(a.body[0]) == ast.dump(b.body[0])
    elif isinstance(a, ast.FunctionDef):
        a = copy(a)
        b = copy(b)
        a.name = ''
        b.name = ''
        return ast.dump(a) == ast.dump(b)
    else:
        raise ValueError('Cannot compare ASTs.')


def test_type_graph_column_int(grid_with_attrs):
    return type_graph_column(grid_with_attrs, 'population') == int


def test_type_graph_column_float(grid_with_attrs):
    return type_graph_column(grid_with_attrs, 'area') == float


def test_type_graph_column_multi_type(grid_with_attrs):
    grid_with_attrs.nodes[(0, 0)]['area'] = 1
    with pytest.raises(TypeError):
        return type_graph_column(grid_with_attrs, 'area')


def test_tally_columns_single_fields():
    name_to_col = {'tally1': 'a', 'tally2': 'b'}
    updaters = {name: Tally(col) for name, col in name_to_col.items()}
    assert tally_columns(updaters) == name_to_col


def test_tally_columns_single_field_repeated():
    name_to_col = {'tally1': 'a', 'tally2': 'a'}
    updaters = {name: Tally(col) for name, col in name_to_col.items()}
    assert tally_columns(updaters) == name_to_col


def test_tally_columns_multi_field():
    with pytest.raises(ValueError):
        tally_columns({'tally': Tally(['a', 'b'])})


def test_tally_columns_non_tally():
    with pytest.raises(ValueError):
        tally_columns({'tally': Tally('a'), 'cut_edges': cut_edges})


def test_to_ast_always_accept(grid_with_attrs):
    # TODO: check more invariants.
    to_ast(always_accept, 'accept', grid_with_attrs, {})


def test_to_ast_always_accept_with_store(grid_with_attrs):
    # TODO: check more invariants.
    to_ast(always_accept_with_store, 'accept', grid_with_attrs, {})


def test_to_ast_non_accept_non_constraint(grid_with_attrs):
    with pytest.raises(CompileError):
        to_ast(always_accept, 'updater', grid_with_attrs, {})


def test_type_updaters_primitive_tallies(grid_with_attrs):
    updaters = {col: Tally(col) for col in ('area', 'population', 'flag')}
    assert type_updater_columns(grid_with_attrs, updaters) == {
        'area': float,
        'population': int,
        'flag': bool
    }


def test_to_ast_non_primitive_tally(grid_with_attrs):
    with pytest.raises(CompileError):
        type_updater_columns(grid_with_attrs, {'string': Tally('string')})


def test_load_function_ast_non_function_module(grid_with_attrs):
    with pytest.raises(CompileError):
        load_function_ast(nx)


def test_load_function_ast_always_accept(grid_with_attrs):
    fn_ast = load_function_ast(always_accept)
    assert isinstance(fn_ast, ast.FunctionDef)
    assert fn_ast.name == 'always_accept'


def test_load_function_ast_always_accept_with_store(grid_with_attrs):
    fn_ast = load_function_ast(always_accept_with_store)
    assert isinstance(fn_ast, ast.FunctionDef)
    assert fn_ast.name == 'always_accept_with_store'


def test_load_function_ast_wrong_arguments(grid_with_attrs):
    def bad_accept(a, b, c):
        return True

    with pytest.raises(CompileError):
        load_function_ast(bad_accept)


def test_load_function_ast_store_only(grid_with_attrs):
    def bad_accept(store):
        return True

    with pytest.raises(CompileError):
        load_function_ast(bad_accept)


def test_dsl_validation_always_accept():
    DSLValidationVisitor().visit(fn_to_ast(always_accept))


def test_dsl_validation_if_and_assign():
    def test_fn(partition):
        x = 1
        x += 2
        if partition.assignment[0] == x:
            return True
        elif partition.assignment[1] == x:
            return False
        else:
            return True

    DSLValidationVisitor().visit(fn_to_ast(test_fn))


def test_dsl_validation_for():
    def test_fn(partition):
        x = 1
        x += 2
        for ii in range(x):
            x += 1
        return True

    with pytest.raises(CompileError):
        DSLValidationVisitor().visit(fn_to_ast(test_fn))


def test_dsl_validation_list_expr():
    def test_fn(partition):
        x = [1, 2, 3]
        return True

    with pytest.raises(CompileError):
        DSLValidationVisitor().visit(fn_to_ast(test_fn))


def test_assignment_normalizer_augmented_assignment():
    def test_fn(x):
        x += 2

    def test_fn_normalized(x):
        x = x + 2

    expected_ast = fn_to_ast(test_fn_normalized)
    actual_ast = AssignmentNormalizer().visit(fn_to_ast(test_fn))
    assert ast_equal(expected_ast, actual_ast)


def test_assignment_normalizer_annotated_assignment():
    def test_fn():
        x: int = 2

    def test_fn_normalized():
        x = 2

    expected_ast = fn_to_ast(test_fn_normalized)
    actual_ast = AssignmentNormalizer().visit(fn_to_ast(test_fn))
    assert ast_equal(expected_ast, actual_ast)


def test_assignment_normalizer_multiple_target_assignment():
    def test_fn(x, y):
        x, y = y, x

    with pytest.raises(CompileError):
        AssignmentNormalizer().visit(fn_to_ast(test_fn))


def test_loaded_names_visitor():
    def test_fn(x, y, z, q, r):
        if x == 2:
            return y
        elif z == 3:
            return 2 + x
        return q

    visitor = LoadedNamesVisitor()
    visitor.visit(fn_to_ast(test_fn))
    assert visitor.loaded == {'x', 'y', 'z', 'q'}


def test_find_names_simple_assignments():
    def test_fn(x):
        y = 1
        z = 2
        return x + y + z

    locals, _ = find_names(fn_to_ast(test_fn), get_ctx(test_fn))
    assert locals == {'x', 'y', 'z'}


def test_find_names_simple_unbound():
    def test_fn():
        return x

    with pytest.raises(CompileError):
        find_names(fn_to_ast(test_fn), get_ctx(test_fn))


def test_find_names_elif_good():
    def test_fn(x):
        if x == 1:
            y = 1
        elif x == 2:
            y = 2
        elif x == 3:
            y = 3
        else:
            y = 4
        return y + 1

    locals, _ = find_names(fn_to_ast(test_fn), get_ctx(test_fn))
    assert locals == {'x', 'y'}


def test_find_names_if_bad():
    def test_fn(x):
        if x == 1:
            y = 1
        # y is only defined in one branch of the if statement,
        # so this may not necessarily work. In our DSL, we do
        # not allow such a scenario.
        return y + 1

    with pytest.raises(CompileError):
        find_names(fn_to_ast(test_fn), get_ctx(test_fn))


def test_find_names_globals_nonlocals_good():
    y = 1

    def test_fn(x):
        z = x + y  # uses nonlocal
        return always_accept  # uses global

    locals, closure_vars = find_names(fn_to_ast(test_fn), get_ctx(test_fn))
    assert locals == {'x', 'z'}
    assert closure_vars == {'y', 'always_accept'}


def test_find_names_globals_nonlocals_assignment_shadowing():
    y = 1

    def test_fn(x):
        y = 2
        z = x + y  # uses local that shadows nonlocal
        return always_accept  # uses global

    locals, closure_vars = find_names(fn_to_ast(test_fn), get_ctx(test_fn))
    assert locals == {'x', 'y', 'z'}
    assert closure_vars == {'always_accept'}


def test_find_names_globals_nonlocals_arg_shadowing():
    y = 1

    def test_fn(x, always_accept):
        z = x + y  # uses nonlocal
        return always_accept  # uses local that shadows global

    locals, closure_vars = find_names(fn_to_ast(test_fn), get_ctx(test_fn))
    assert locals == {'x', 'z', 'always_accept'}
    assert closure_vars == {'y'}


def test_find_names_nonlocals_bad_ref_before_set():
    y = 1

    def test_fn():
        # This raises an UnboundLocalError, which we can
        # pick up with static analysis in this case.
        if y == 1:
            y += 2

    with pytest.raises(CompileError):
        find_names(fn_to_ast(test_fn), get_ctx(test_fn))


def test_find_names_unbound_in_if_test():
    def test_fn():
        if x == 1:
            return 1
        return 2

    with pytest.raises(CompileError):
        find_names(fn_to_ast(test_fn), get_ctx(test_fn))


def test_find_names_unsupported_statement():
    def test_fn():
        x = 0
        for i in range(10):
            x += i
        return x

    with pytest.raises(CompileError):
        find_names(fn_to_ast(test_fn), get_ctx(test_fn))


def test_preprocess_ast_nonlocal_substitution_primitives():
    x = 1
    y = 2.0
    z = True

    def test_fn(a):
        if z:
            return x + y
        return a

    def test_fn_normalized(a):
        if True:
            return 1 + 2.0
        return a

    expected_ast = fn_to_ast(test_fn_normalized)
    actual_ast = preprocess_ast(fn_to_ast(test_fn), get_ctx(test_fn))
    assert ast_equal(expected_ast, actual_ast)


def test_preprocess_ast_nonlocal_substitution_non_primitive():
    x = 'a'

    def test_fn():
        return x

    with pytest.raises(CompileError):
        preprocess_ast(fn_to_ast(test_fn), get_ctx(test_fn))


def test_always_returns_no_branches_return():
    def test_fn():
        return 1

    assert always_returns(fn_to_ast(test_fn))


def test_always_returns_no_branches_return():
    def test_fn():
        return 1

    assert always_returns(fn_to_ast(test_fn).body)


def test_always_returns_no_branches_no_return():
    def test_fn():
        pass

    assert not always_returns(fn_to_ast(test_fn).body)


def test_always_returns_one_level_return():
    def test_fn(x):
        if x:
            return 1
        else:
            return 2

    assert always_returns(fn_to_ast(test_fn).body)


def test_always_returns_one_level_asymmetric_return():
    def test_fn(x):
        if x:
            return 1
        return 2

    assert always_returns(fn_to_ast(test_fn).body)


def test_always_returns_one_level_no_return():
    def test_fn(x):
        if x:
            return 1

    assert not always_returns(fn_to_ast(test_fn).body)


def test_always_returns_two_level_return():
    def test_fn(x, y):
        if x:
            if y:
                return 1
            else:
                return 2
        else:
            if y:
                return 3
            else:
                return 4

    assert always_returns(fn_to_ast(test_fn).body)


def test_always_returns_two_level_asymmetric_return():
    def test_fn(x, y):
        if x:
            if y:
                return 1
            return 2
        else:
            if not y:
                return 4
        return 3

    assert always_returns(fn_to_ast(test_fn).body)


def test_always_returns_two_level_asymmetric_no_return():
    def test_fn(x, y):
        if x:
            if y:
                return 1
            return 2
        else:
            if not y:
                return 4

    assert not always_returns(fn_to_ast(test_fn).body)


@pytest.mark.parametrize('val', [1, True, 1.0])
def test_type_and_transform_expr_constant_primitive(val):
    constant_ast = ast.Constant(value=val)
    constant_type, transformed_ast = type_and_transform_expr(constant_ast)
    assert constant_type == type(val)
    assert transformed_ast == Constant(val)


def test_type_and_transform_expr_constant_non_primitive():
    constant_ast = ast.Constant(value='abc')
    with pytest.raises(CompileError):
        type_and_transform_expr(constant_ast)


def test_type_and_transform_expr_name_in_context():
    constant_ast = ast.Name(id='x', ctx=ast.Load())
    ctx = {'x': Union[float, int]}
    constant_type, transformed_ast = type_and_transform_expr(constant_ast, ctx)
    assert constant_type == ctx['x']
    assert transformed_ast == Name('x')


def test_type_and_transform_expr_name_not_in_context():
    constant_ast = ast.Name(id='x', ctx=ast.Load())
    with pytest.raises(CompileError):
        type_and_transform_expr(constant_ast)


def test_type_and_transform_expr_name_store_context():
    constant_ast = ast.Name(id='x', ctx=ast.Store())
    ctx = {'x': Union[float, int]}
    with pytest.raises(CompileError):
        type_and_transform_expr(constant_ast, ctx)


@pytest.mark.parametrize(
    'operand_type,exp_type',
    [
        # primitive types
        (int, int),
        (bool, int),
        (float, float),

        # type unions of primitive types
        (Union[int, bool], int),
        (Union[int, float], Union[int, float]),
        (Union[bool, float], Union[int, float]),
        (Union[int, bool, float], Union[int, float]),

        # vectors of primitive types
        (Vec[int], Vec[int]),
        (Vec[bool], Vec[int]),
        (Vec[float], Vec[float]),

        # type unions of vectors of primitive types
        (Union[Vec[int], Vec[bool]], Vec[int]),
        (Union[Vec[int], Vec[float]], Union[Vec[int], Vec[float]]),
        (Union[Vec[bool], Vec[float]], Union[Vec[int], Vec[float]]),
        (Union[Vec[int], Vec[bool], Vec[float]], Union[Vec[int], Vec[float]])
    ])
@pytest.mark.parametrize('op', ['-', '+'])
def test_type_and_transform_expr_unary_op_uadd_usub(operand_type, exp_type,
                                                    op):
    if op == '-':
        ast_op = ast.USub()
        opcode = UnaryOpcode.USUB
    elif op == '+':
        ast_op = ast.UAdd()
        opcode = UnaryOpcode.UADD
    uop_ast = ast.UnaryOp(op=ast_op, operand=ast.Name(id='x', ctx=ast.Load()))
    ctx = {'x': operand_type}
    uop_type, transformed_ast = type_and_transform_expr(uop_ast, ctx)
    assert uop_type == exp_type
    assert transformed_ast == UnaryOp(opcode, Name('x'))


@pytest.mark.parametrize('operand_type,exp_type',
                         ([(t, bool)
                           for t in PRIMITIVE_TYPES] + [(t, Vec[bool])
                                                        for t in VEC_TYPES]))
def test_type_and_transform_expr_unary_op_not(operand_type, exp_type):
    uop_ast = ast.UnaryOp(op=ast.Not(),
                          operand=ast.Name(id='x', ctx=ast.Load()))
    ctx = {'x': operand_type}
    uop_type, transformed_ast = type_and_transform_expr(uop_ast, ctx)
    assert uop_type == exp_type
    assert transformed_ast == UnaryOp(UnaryOpcode.NOT, Name('x'))


@pytest.mark.parametrize('lhs_type', TRUTHY_TYPES)
@pytest.mark.parametrize('rhs_type', TRUTHY_TYPES)
@pytest.mark.parametrize('op', ['and', 'or'])
def test_type_and_transform_expr_bool_op_primitive_pairs(
        lhs_type, rhs_type, op):
    if op == 'and':
        ast_op = ast.And()
        opcode = BoolOpcode.AND
    elif op == 'or':
        ast_op = ast.Or()
        opcode = BoolOpcode.OR
    bool_op_ast = ast.BoolOp(op=ast_op,
                             values=[
                                 ast.Name(id='x', ctx=ast.Load()),
                                 ast.Name(id='y', ctx=ast.Load())
                             ])
    ctx = {'x': lhs_type, 'y': rhs_type}
    bool_op_type, transformed_ast = type_and_transform_expr(bool_op_ast, ctx)
    assert bool_op_type == bool
    assert transformed_ast == BoolOp(opcode, (Name('x'), Name('y')))


@pytest.mark.parametrize('lhs_type', TRUTHY_TYPES)
@pytest.mark.parametrize('rhs_type', TRUTHY_TYPES)
@pytest.mark.parametrize('op', ['and', 'or'])
def test_type_and_transform_expr_bool_op_primitive_triples(
        lhs_type, rhs_type, op):
    if op == 'and':
        ast_op = ast.And()
        opcode = BoolOpcode.AND
    elif op == 'or':
        ast_op = ast.Or()
        opcode = BoolOpcode.OR
    bool_op_ast = ast.BoolOp(op=ast_op,
                             values=[
                                 ast.Name(id='x', ctx=ast.Load()),
                                 ast.Name(id='y', ctx=ast.Load()),
                                 ast.Name(id='z', ctx=ast.Load())
                             ])
    ctx = {'x': lhs_type, 'y': rhs_type, 'z': bool}
    bool_op_type, transformed_ast = type_and_transform_expr(bool_op_ast, ctx)
    assert bool_op_type == bool
    assert transformed_ast == BoolOp(opcode, (Name('x'), Name('y'), Name('z')))


def test_type_and_transform_expr_bool_op_non_truthy_vecs():
    bool_op_ast = ast.BoolOp(op=ast.And(),
                             values=[
                                 ast.Name(id='x', ctx=ast.Load()),
                                 ast.Name(id='y', ctx=ast.Load())
                             ])
    ctx = {'x': Vec[bool], 'y': Vec[bool]}
    with pytest.raises(CompileError):
        type_and_transform_expr(bool_op_ast, ctx)


def test_type_and_transform_expr_bool_op_non_truthy_mixed():
    bool_op_ast = ast.BoolOp(op=ast.Or(),
                             values=[
                                 ast.Name(id='x', ctx=ast.Load()),
                                 ast.Name(id='y', ctx=ast.Load())
                             ])
    ctx = {'x': Vec[bool], 'y': bool}
    with pytest.raises(CompileError):
        type_and_transform_expr(bool_op_ast, ctx)


@pytest.fixture
def if_expr_ast():
    return ast.IfExp(test=ast.Name(id='x', ctx=ast.Load()),
                     body=ast.Name(id='y', ctx=ast.Load()),
                     orelse=ast.Name(id='z', ctx=ast.Load()))


def test_type_and_transform_expr_if_expr_same_type(if_expr_ast):
    ctx = {'x': bool, 'y': int, 'z': int}
    expected_ast = IfExpr(Name('x'), Name('y'), Name('z'))
    uop_type, transformed_ast = type_and_transform_expr(if_expr_ast, ctx)
    assert uop_type == int
    assert transformed_ast == expected_ast


def test_type_and_transform_expr_if_expr_different_types(if_expr_ast):
    ctx = {'x': int, 'y': int, 'z': float}
    expected_ast = IfExpr(Name('x'), Name('y'), Name('z'))
    uop_type, transformed_ast = type_and_transform_expr(if_expr_ast, ctx)
    assert uop_type == Union[int, float]
    assert transformed_ast == expected_ast


def test_type_and_transform_expr_if_expr_non_truthy_conditional(if_expr_ast):
    ctx = {'x': Vec[bool], 'y': int, 'z': int}
    with pytest.raises(CompileError):
        type_and_transform_expr(if_expr_ast, ctx)


@pytest.mark.parametrize('operand_type,exp_type', [
    (int, int),
    (bool, int),
    (Union[int, bool], int),
    (Vec[int], Vec[int]),
    (Vec[bool], Vec[int]),
    (Union[Vec[int], Vec[bool]], Vec[int]),
])
def test_type_and_transform_expr_unary_op_invert(operand_type, exp_type):
    uop_ast = ast.UnaryOp(op=ast.Invert(),
                          operand=ast.Name(id='x', ctx=ast.Load()))
    ctx = {'x': operand_type}
    uop_type, transformed_ast = type_and_transform_expr(uop_ast, ctx)
    assert uop_type == exp_type
    assert transformed_ast == UnaryOp(UnaryOpcode.INVERT, Name('x'))


def test_type_and_transform_expr_unary_op_invert_float():
    uop_ast = ast.UnaryOp(op=ast.Invert(),
                          operand=ast.Name(id='x', ctx=ast.Load()))
    ctx = {'x': float}
    with pytest.raises(CompileError):
        # Invert not supported on floats
        type_and_transform_expr(uop_ast, ctx)


@pytest.mark.parametrize('t', PRIMITIVE_TYPES)
def test_is_truthy_primitives(t):
    assert is_truthy(t)


@pytest.mark.parametrize('t', VEC_TYPES)
def test_is_truthy_non_primitives(t):
    assert not is_truthy(t)


@pytest.mark.parametrize('t', PRIMITIVE_TYPES)
def test_is_vec_primitives(t):
    assert not is_vec(t)


@pytest.mark.parametrize('t', SIMPLE_VEC_TYPES)
def test_is_vec_simple_vec_types(t):
    assert is_vec(t)


@pytest.mark.parametrize('t', VEC_TYPE_UNIONS)
def test_is_vec_vec_type_unions(t):
    assert not is_vec(t)


@pytest.mark.parametrize('t', PRIMITIVE_TYPES + VEC_TYPE_UNIONS)
def test_scalar_type_identity(t):
    assert scalar_type(t) == t


@pytest.mark.parametrize('t', SIMPLE_VEC_TYPES)
def test_scalar_type_vecs(t):
    assert scalar_type(t) == get_args(t)[0]


def test_type_union_primitives():
    assert type_union(float, int, bool) == Union[float, int, bool]


def test_type_union_primitives_none():
    assert type_union(float, None, int, bool) == Union[float, int, bool]


def test_type_union_unions_primitive():
    assert type_union(Union[float, int], Union[int, bool]) == Union[float, int,
                                                                    bool]


def test_type_union_primitives_unions_primitive_none():
    assert type_union(Union[float, int], None, bool) == Union[float, int, bool]


def test_type_union_primitives_unions_none():
    assert type_union(Union[float, int], None,
                      Union[int, bool]) == Union[float, int, bool]


def test_type_union_no_args():
    assert type_union() is None


def test_type_union_none_arg():
    assert type_union(None) is None


def test_ctx_union_not_in_context_primitive():
    assert ctx_union({}, 'x', int) == int


def test_ctx_union_not_in_context_primitive_union():
    assert ctx_union({}, 'x', float, int) == Union[float, int]


def test_ctx_union_in_context_primitive_same():
    assert ctx_union({'x': int}, 'x', int) == int


def test_ctx_union_in_context_primitive_different():
    assert ctx_union({'x': float}, 'x', int) == Union[int, float]


def test_ctx_union_not_in_context_primitive_union():
    assert ctx_union({'x': Union[int, float]}, 'x',
                     Union[float, bool]) == Union[float, int, bool]


@pytest.mark.parametrize('t', SIMPLE_TYPES)
def test_defined_type_product_simple_type(t):
    assert list(defined_type_product(t)) == [(t, )]


def test_defined_type_product_undefined():
    with pytest.raises(CompileError):
        defined_type_product(UndefinedVar)


@pytest.mark.parametrize('t', TYPE_UNIONS)
def test_defined_type_product_type_union(t):
    assert set(defined_type_product(t)) == set((t, ) for t in get_args(t))


@pytest.mark.parametrize('t', TYPE_UNIONS)
def test_defined_type_product_type_union_undefined(t):
    with pytest.raises(CompileError):
        defined_type_product(t, UndefinedVar)


@pytest.mark.parametrize('t', SIMPLE_TYPES)
def test_defined_type_product_simple_type_pairs(t):
    assert list(defined_type_product(t, t)) == [(t, t)]


@pytest.mark.parametrize('t', TYPE_UNIONS)
def test_defined_type_product_type_union_pairs(t):
    assert set(defined_type_product(t, t)) == set(
        product(get_args(t), get_args(t)))


@pytest.mark.parametrize('t', TYPE_UNIONS)
def test_defined_type_product_type_union_pairs(t):
    assert set(defined_type_product(t, t)) == set(
        product(get_args(t), get_args(t)))


@pytest.mark.parametrize('t', TYPE_UNIONS)
def test_defined_type_product_type_union_pairs_with_primitive(t):
    assert set(defined_type_product(t, t, int)) == set(
        (*ts, int) for ts in product(get_args(t), get_args(t)))


@pytest.mark.parametrize('t', TYPE_UNIONS)
def test_defined_type_product_type_union_pairs_with_undefined(t):
    with pytest.raises(CompileError):
        defined_type_product(t, t, Union[int, UndefinedVar])
