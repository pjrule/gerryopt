"""Compiler/transpiler for the GerryOpt DSL."""
import ast
import inspect
from textwrap import dedent
from typing import Callable, Iterable, Set, Dict, List, Union
from gerrychain import Graph
from gerrychain.updaters import Tally

PRIMITIVE_TYPES = [int, float, bool]
DSL_DISALLOWED_STATEMENTS = {
    ast.AsyncFunctionDef, ast.ClassDef, ast.Delete, ast.For, ast.AsyncFor,
    ast.While, ast.With, ast.AsyncWith, ast.Raise, ast.Try, ast.Assert,
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Expr, ast.Pass,
    ast.Break, ast.Continue
}
DSL_DISALLOWED_EXPRESSIONS = {
    ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom, ast.FormattedValue,
    ast.JoinedStr, ast.Starred, ast.List, ast.Tuple
}
Primitive = Union[int, float, bool]
Updaters = Dict[str, Callable]


class CompileError(Exception):
    """Raised when a function cannot be compiled to a GerryOpt AST."""


class DSLValidationVisitor(ast.NodeVisitor):
    """AST visitor for verifying that a function matches the GerryOpt DSL.
    
    For now, this consists of checking for explicitly disallowed statement
    or expression forms.
    """
    def generic_visit(self, node):
        if type(node) in DSL_DISALLOWED_STATEMENTS:
            raise CompileError('Encountered statement outside of GerryOpt DSL '
                               f'(statement type {type(node)}).')
        elif type(node) in DSL_DISALLOWED_EXPRESSIONS:
            raise CompileError(
                'Encountered expression outside of GerryOpt DSL '
                f'(expression type {type(node)}).')
        ast.NodeVisitor.generic_visit(self, node)


class AssignmentNormalizer(ast.NodeTransformer):
    """"AST transformer for normalizing augmented and annotated assignments.
    
    In general Python, augmented assignments are not *just* syntactic sugar for
    assignments. However, for the purposes of the GerryOpt DSL, we treat them
    as syntactic sugar. Type annotations are not relevant to the GerryOpt DSL,
    as the type system is quite simple, so we simply strip them without validating
    them.
    """
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.Assign:
        return ast.Assign(targets=[node.target],
                          value=ast.BinOp(left=ast.Name(id=node.target.id,
                                                        ctx=ast.Load()),
                                          op=node.op,
                                          right=node.value),
                          type_comment=None)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign:
        return ast.Assign(targets=[node.target],
                          value=node.value,
                          type_comment=None)


class LoadedNamesVisitor(ast.NodeVisitor):
    """AST visitor for finding loaded names."""
    def __init__(self, *args, **kwargs):
        self.loaded = set()
        super().__init__(*args, **kwargs)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)


class ClosureValuesTransformer(ast.NodeTransformer):
    """AST transformer that replaces references to captured values with
    their literal values, performing basic type checks along the way.
    """
    def __init__(self, *args, vals: Dict[str, Primitive], **kwargs):
        self.vals = vals
        super().__init__(*args, **kwargs)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.vals:
            return ast.Constant(value=self.vals[node.id], kind=None)
        return node


def find_names(fn_ast: ast.FunctionDef, ctx: inspect.ClosureVars) -> Set[str]:
    """Determines the names of bound locals and closure variables in a compilable function."""
    if ctx.unbound:
        raise CompileError(f'Function has unbound names {ctx.unbound}.')
    # TODO: filter closure variables to minimum necessary set.
    closure_vars = set.union(*(set(v.keys())
                               for v in (ctx.globals, ctx.nonlocals,
                                         ctx.builtins)))
    params = set(a.arg for a in fn_ast.args.args)
    closure_vars -= params
    bound_locals, _ = new_bindings(fn_ast.body, params, set(), closure_vars)
    return bound_locals, closure_vars


def new_bindings(statements: List[ast.AST], bound_locals: Set[str],
                 loaded_names: Set[str], closure_vars: Set[str]):
    bound_locals = bound_locals.copy()
    loaded_names = loaded_names.copy()

    def load_expr(expr):
        expr_visitor = LoadedNamesVisitor()
        expr_visitor.visit(expr)
        unbound = expr_visitor.loaded - bound_locals - closure_vars
        if unbound:
            raise CompileError(f'Unbound locals: cannot load names {unbound}.')
        return expr_visitor.loaded

    for statement in statements:
        if isinstance(statement, ast.If):
            loaded_names |= load_expr(statement.test)
            if_bindings, if_loaded = new_bindings(statement.body, bound_locals,
                                                  loaded_names, closure_vars)
            else_bindings, else_loaded = new_bindings(statement.orelse,
                                                      bound_locals,
                                                      loaded_names,
                                                      closure_vars)
            bound_locals |= (if_bindings & else_bindings)
            loaded_names |= (if_loaded | else_loaded)
        elif isinstance(statement, ast.Assign):
            statement_visitor = LoadedNamesVisitor()
            statement_visitor.visit(statement.value)
            loaded_names |= statement_visitor.loaded

            # We say that a local is unbound if either:
            #   (a) Its name is neither in the closure variables nor was previously
            #       on the l.h.s. of any assignment statement.
            #   (b) Its name is in the closure context but is on the l.h.s. of some
            #       assignment statement *after* its value is loaded.
            targets = set(t.id for t in statement.targets)
            unbound_b = targets & loaded_names & closure_vars
            if unbound_b:
                raise CompileError(
                    f'Unbound locals: cannot assign names {unbound_b} '
                    'that were previously loaded as globals or nonlocals.')
            unbound_a = statement_visitor.loaded - bound_locals - closure_vars
            if unbound_a:
                raise CompileError(
                    f'Unbound locals: cannot load names {unbound_a}.')
            bound_locals |= targets
        elif isinstance(statement, ast.Return):
            loaded_names |= load_expr(statement.value)
        else:
            raise CompileError(
                f'Encountered invalid statement (type {type(statement)}).')
    return bound_locals, loaded_names


def type_graph_column(graph: Graph, column: str):
    """Determines the type of a column in `graph`."""
    column_types = set(type(v) for _, v in graph.nodes(column))
    if len(column_types) > 1:
        raise TypeError(
            f'Column "{column}" has multiple types: {column_types}')
    return next(iter(column_types))


def tally_columns(updaters: Updaters) -> Dict[str, str]:
    """Extracts the columns used by updaters.
    
    Raises:
        ValueError: If a non-tally updater is encountered, or if 
        a tally is multi-column.
    """
    columns = {}
    for updater_name, updater in updaters.items():
        if not isinstance(updater, Tally):
            raise ValueError(
                'Cannot extract tally column from non-Tally updater.')
        if len(updater.fields) != 1:
            raise ValueError('Multi-column tallies not supported.')
        columns[updater_name] = updater.fields[0]
    return columns


def type_updater_columns(graph: Graph, updaters: Updaters) -> Dict:
    """Determines the types of graph columns used by Tally updaters."""
    column_dependencies = tally_columns(updaters)
    column_types = {
        col: type_graph_column(graph, col)
        for col in column_dependencies.values()
    }
    if set(column_types.values()) - set(PRIMITIVE_TYPES):
        raise CompileError('Tallies with non-primitive types not supported.')
    return column_types


def load_function_ast(fn: Callable) -> ast.FunctionDef:
    """Loads the AST of a compilable function."""
    raw_ast = ast.parse(dedent(inspect.getsource(fn)))
    if (not isinstance(raw_ast, ast.Module) or len(raw_ast.body) != 1
            or not isinstance(raw_ast.body[0], ast.FunctionDef)):
        raise CompileError('Cannot compile a non-function.')

    fn_ast = raw_ast.body[0]
    arg_names = set(arg.arg for arg in fn_ast.args.args)
    if arg_names != {'partition'} and arg_names != {'partition', 'store'}:
        raise CompileError(
            'Compiled functions must take a `partition` argument '
            'and an optional `store` argument.')
    return fn_ast


def to_ast(fn: Callable, fn_type: str, graph: Graph, updaters: Updaters):
    """Compiles a function to a GerryOpt AST."""
    if fn_type not in ('accept', 'constraint', 'score'):
        raise CompileError(
            'Can only compile acceptance, constraint, and score functions.')
    column_types = type_updater_columns(graph, updaters)

    fn_context = inspect.getclosurevars(fn)
    raw_fn_ast = load_function_ast(fn)
    DSLValidationVisitor().visit(raw_fn_ast)
    fn_ast = AssignmentNormalizer().visit(raw_fn_ast)
    # bound_locals, closure_vars = find_names(fn_ast, fn_context)
    # fn_ast = ClosureValuesTransformer(closure_vars).visit(fn_ast)

    for stmt in fn_ast.body:
        if isinstance(stmt, ast.Assign):
            pass
        elif isinstance(stmt, ast.If):
            pass
        elif isinstance(stmt, ast.Return):
            pass

    return fn_ast
