import ast
import inspect
import pytest
import networkx as nx
from copy import copy
from textwrap import dedent
from typing import Callable
from gerryopt.compile import (LoadedNamesVisitor, type_graph_column,
                              tally_columns, CompileError, to_ast,
                              load_function_ast, type_updater_columns,
                              DSLValidationVisitor, AssignmentNormalizer)
from gerrychain.updaters import Tally, cut_edges
from gerrychain.grid import create_grid_graph


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
    return ast.parse(dedent(inspect.getsource(fn)))


def ast_equal(a: ast.AST, b: ast.AST):
    """Compares two ASTs of two function."""
    # TODO: comparing ASTs is apparently a rather subtle business.
    # For now, we use the hack of comparing AST dumps, but this may
    # cause false negatives in some cases.
    # (see https://stackoverflow.com/q/3312989)
    assert len(a.body) == len(b.body) == 1
    assert isinstance(a.body[0], ast.FunctionDef)
    assert isinstance(b.body[0], ast.FunctionDef)
    a = copy(a)
    b = copy(b)
    a.body[0].name = ''
    b.body[0].name = ''
    return ast.dump(a.body[0]) == ast.dump(b.body[0])


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
