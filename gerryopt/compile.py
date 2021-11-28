"""Compiler/transpiler for the GerryOpt DSL."""
import ast
import json
import inspect
from copy import deepcopy
from textwrap import dedent
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
from itertools import product
from typing import (Callable, Iterable, Set, Dict, List, Union, Any, Optional,
                    Tuple, get_args, get_origin)
from gerrychain import Graph
from gerrychain.updaters import Tally
from gerryopt.vector import Vec

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
    them. Multiple-target assignment (e.g. `x, y = y, x`) is not allowed.
    """
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if isinstance(node.targets[0], ast.Tuple):
            # TODO
            raise CompileError(
                'Multiple-target assignment not supported by the GerryChain DSL.'
            )
        return node

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
            if type(self.vals[node.id]) in PRIMITIVE_TYPES:
                return ast.Constant(value=self.vals[node.id], kind=None)
            raise CompileError(
                f'Cannot substitute non-primitive value (name "{node.id}" '
                f'has type {type(self.vals[node.id])}).')
        return node


def merge_closure_vars(ctx: inspect.ClosureVars) -> Dict[str, Any]:
    """Merges nonlocals, globals, and builtins in `ctx`."""
    return {**ctx.globals, **ctx.nonlocals, **ctx.builtins}


def find_names(fn_ast: ast.FunctionDef, ctx: inspect.ClosureVars) -> Set[str]:
    """Determines the names of bound locals and closure variables in a compilable function."""
    if ctx.unbound:
        raise CompileError(f'Function has unbound names {ctx.unbound}.')
    # TODO: filter closure variables to minimum necessary set.
    closure_vars = set(merge_closure_vars(ctx).keys())
    params = set(a.arg for a in fn_ast.args.args)
    closure_vars -= params
    bound_locals, _ = new_bindings(fn_ast.body, params, set(), closure_vars)
    return bound_locals, closure_vars


def new_bindings(statements: List[ast.AST], bound_locals: Set[str],
                 loaded_names: Set[str], closure_vars: Set[str]):
    """Parses variable references in a list of statements.

    Args:
        statements: 
    
    We say that a local is unbound if either:
        (a) Its name is neither in the closure variables nor was previously
            on the l.h.s. of any assignment statement.
        (b) Its name is in the closure context but is on the l.h.s. of some
             assignment statement *after* its value is loaded.
    """
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


def always_returns(statements: List[ast.AST]) -> bool:
    """Determines if a list of statements is guaranteed to `return`."""
    # Recursively:
    #  * If the list of statements contains ≥1 return statements and
    #    does not branch (no if block), we are guaranteed to return.
    #  * If the list of statements *does* contain ≥1 if block, then
    #    (recursively) both parts of the block should be guaranteed to
    #    return *or* there should be a return statement *after* the block.
    for statement in statements:
        if isinstance(statement, ast.Return):
            return True
        if isinstance(statement, ast.If):
            if_returns = always_returns(statement.body)
            else_returns = always_returns(statement.orelse)
            if if_returns and else_returns:
                return True
    return False


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


def preprocess_ast(fn_ast: ast.FunctionDef,
                   ctx: inspect.ClosureVars) -> ast.FunctionDef:
    """Validates and transforms the AST of a compilable function.

    First, we validate that the AST represents a function within the GerryOpt
    DSL (this mostly involves verifying that no disallowed statement or
    expression forms are used). Then, we normalize assignment expressions and
    replace closed-over variable names with constants.

    Args:
        fn_ast: The raw function AST.
        ctx: The function's closure variables.

    Returns:
        The AST of the transformed function.

    Raises:
        CompileError: If validation or transformation fails---that is, the
        function is outside of the GerryOpt DSL, uses unbound locals, or
        closes over non-primitive variables.
    """
    DSLValidationVisitor().visit(fn_ast)
    fn_ast = AssignmentNormalizer().visit(fn_ast)
    bound_locals, closure_vars = find_names(fn_ast, ctx)
    all_closure_vals = merge_closure_vars(ctx)
    filtered_closure_vals = {k: all_closure_vals[k] for k in closure_vars}
    closed_ast = ClosureValuesTransformer(
        vals=filtered_closure_vals).visit(fn_ast)
    if not always_returns(closed_ast.body):
        raise CompileError(
            'GerryOpt functions must always return a non-`None` value.')
    return closed_ast


def is_truthy(t: type) -> bool:
    """Determines if a type is considered truthy in the GerryOpt DSL."""
    if get_origin(t) is Union:
        return all(member_t in PRIMITIVE_TYPES for member_t in get_args(t))
    return t in PRIMITIVE_TYPES


def scalar_type(t: type) -> type:
    """Returns the type of an element X of a Vec[X] (identity otherwise)."""
    if get_origin(t) is Vec:
        return get_args(t)[0]
    return t


def is_vec(t: type) -> bool:
    """Determines if a type is an instance of Vec[T]."""
    return get_origin(t) is Vec


class UndefinedVar:
    """A pseudotype for possibly undefined variables."""


TypeContext = TypeDelta = Dict[str, type]
ReturnType = Optional[type]
CompiledIdentifier = str


class AST:
    pass


class Expr(AST):
    pass


class Statement(AST):
    pass


def type_and_transform_expr(expr: ast.Expr,
                            ctx: TypeContext) -> Tuple[type, Expr]:
    raise NotImplementedError('stub for typing')


def type_and_transform_statements(
        statements: List[ast.AST], ctx: TypeContext, return_type: ReturnType
) -> Tuple[TypeDelta, ReturnType, List[Statement]]:
    raise NotImplementedError('stub for typing')


def type_union(*args: type) -> Optional[type]:
    """Finds the union of types, eliminating `None` where possible."""
    union_t = None
    for t in args:
        if union_t is None:
            union_t = t
        elif t is not None:
            union_t = Union[union_t, t]
    return union_t


def ctx_union(ctx: TypeContext, name: str, *args: type) -> type:
    """Finds the union of types with the existing type of `name` in `ctx`.

    If `name` is not available in `ctx`, we simply find the union of
    the directly passed types.
    """
    if name in ctx:
        return type_union(ctx[name], *args)
    return type_union(*args)


def defined_type_product(*args: type) -> Iterable:
    """Generates the Cartesian product of (union) types.
    
    Raises:
        CompileError: If the product contains `UndefinedVar`.
    """
    unrolled = [get_args(t) if get_origin(t) is Union else (t, ) for t in args]
    for types in unrolled:
        if UndefinedVar in types:
            raise CompileError(
                'Cannot compute type product for potentially undefined variables.'
            )
    return product(*unrolled)


@dataclass
class If(Statement):
    test: Expr
    body: List['Statement']
    orelse: List['Statement']

    def type_and_transform(
            cls, statement: ast.If, ctx: TypeContext, return_type: ReturnType
    ) -> Tuple[TypeDelta, ReturnType, Statement]:
        delta = {}
        test_type, test_ast = type_and_transform_expr(statement.test)
        if_types, if_return_type, if_asts = type_and_transform_statements(
            statement.body, ctx, return_type)
        else_types, else_return_type, else_asts = type_and_transform_statements(
            statement.orelse, ctx, return_type)
        if_names = set(if_types.keys())
        else_names = set(else_types.keys())
        for name in if_names & else_names:
            delta[name] = ctx_union(ctx, if_types[name], else_types[name])
        for name in if_names - else_names:
            delta[name] = ctx_union(ctx, if_types[name], UndefinedVar)
        for name in else_names - if_names:
            delta[name] = ctx_union(ctx, else_types[name], UndefinedVar)
        if if_return_type is not None:
            return_type = Union[return_type, if_return_type]
        if else_return_type is not None:
            return_type = Union[return_type, else_return_type]
        return delta, return_type, cls(test_ast, if_asts, else_asts)


@dataclass
class Return(Statement):
    value: Expr

    @classmethod
    def type_and_transform(cls, statement: ast.If,
                           ctx: TypeContext) -> Tuple[ReturnType, Statement]:
        branch_return_type, return_ast = type_and_transform_expr(
            statement.value, ctx)
        return branch_return_type, Return(return_ast)


@dataclass
class Assign(Statement):
    target: CompiledIdentifier
    value: Expr

    @classmethod
    def type_and_transform(cls, statement: ast.Assign,
                           ctx: TypeContext) -> Tuple[TypeDelta, Statement]:
        delta = {}
        rhs_type, rhs_ast = type_and_transform_expr(statement.value, ctx)
        lhs = statement.targets[0].id
        if lhs in ctx:
            ctx[lhs] = Union[ctx[lhs], rhs_type]
        else:
            ctx[lhs] = rhs_type
        delta[lhs] = ctx[lhs]
        return delta, cls(lhs, rhs_ast)


@dataclass
class Name(Expr):
    id: CompiledIdentifier

    @classmethod
    def type_and_transform(cls, expr: ast.Name,
                           ctx: TypeContext) -> Tuple[type, 'Name']:
        if isinstance(expr.ctx, ast.Store):
            raise CompileError('Cannot type name in store context.')
        try:
            return ctx[expr.id], Name(expr.id)
        except KeyError:
            raise CompileError(
                f'Could not resolve type for unbound local "{expr.id}".')


@dataclass
class Constant(Expr):
    value: Primitive

    @classmethod
    def type_and_transform(cls, expr: ast.Constant,
                           ctx: TypeContext) -> Tuple[type, 'Constant']:
        val = expr.value
        if isinstance(val, get_args(Primitive)):
            return type(val), Constant(val)
        raise CompileError(f'Cannot type non-primitive constant {val}')


BoolOpcode = Enum('BoolOpcode', 'AND OR')


@dataclass
class BoolOp(Expr):
    op: BoolOpcode
    values: Iterable[Expr]
    OPS = {ast.And: BoolOpcode.AND, ast.Or: BoolOpcode.OR}

    @classmethod
    def type_and_transform(cls, expr: ast.BoolOp,
                           ctx: TypeContext) -> Tuple[type, 'BoolOp']:
        arg_types, arg_asts = list(
            zip(*(type_and_transform_expr(e, ctx) for e in expr.values)))
        if not all(is_truthy(t) for t in arg_types):
            raise CompileError(
                'All arguments to a boolean operator must be truthy.')
        compiled_expr = cls(BoolOp.OPS[type(expr.op)], arg_asts)
        return bool, compiled_expr


UnaryOpcode = Enum('UnaryOpcode', 'UADD USUB INVERT NOT')


@dataclass
class UnaryOp(Expr):
    op: UnaryOpcode
    operand: Expr

    OPS = {
        ast.UAdd: UnaryOpcode.UADD,
        ast.USub: UnaryOpcode.USUB,
        ast.Invert: UnaryOpcode.INVERT,
        ast.Not: UnaryOpcode.NOT,
    }
    OP_TYPES = {
        (ast.UAdd, float): float,
        (ast.USub, float): float,
        # Invert not supported on floats
        (ast.Not, float): bool,
        (ast.UAdd, int): int,
        (ast.USub, int): int,
        (ast.Invert, int): int,
        (ast.Not, int): bool,
        (ast.UAdd, bool): int,
        (ast.USub, bool): int,
        (ast.Invert, bool): int,
        (ast.Not, bool): bool,
    }

    @classmethod
    def type_and_transform(cls, expr: ast.UnaryOp,
                           ctx: TypeContext) -> Tuple[type, 'UnaryOp']:
        operand_type, operand_ast = type_and_transform_expr(expr.operand, ctx)
        type_lb = None
        op_type = type(expr.op)
        try:
            expr_ast = cls(UnaryOp.OPS[op_type], operand_ast)
        except KeyError:
            raise CompileError(f'Unary operation {op_type} not supported.')

        for (t, ) in defined_type_product(operand_type):
            try:
                expr_type = UnaryOp.OP_TYPES[(op_type, scalar_type(t))]
            except KeyError:
                raise CompileError(
                    f'Unary operation {op_type} not supported for type {t}.')
            if is_vec(t):
                type_lb = type_union(Vec[expr_type], type_lb)
            else:
                type_lb = type_union(expr_type, type_lb)

        return type_lb, expr_ast


@dataclass
class IfExpr(Expr):
    test: Expr
    body: Expr
    orelse: Expr

    @classmethod
    def type_and_transform(cls, expr: ast.Expr,
                           ctx: TypeContext) -> Tuple[type, 'IfExpr']:
        test_type, test_ast = type_and_transform_expr(expr.test, ctx)
        if_type, if_ast = type_and_transform_expr(expr.body, ctx)
        else_type, else_ast = type_and_transform_expr(expr.orelse, ctx)
        if not is_truthy(test_type):
            raise CompileError('Test in conditional expression is not truthy.')
        return Union[if_type, else_type], cls(test_ast, if_ast, else_ast)


CmpOpcode = Enum('CmpOpcode', 'EQ NOT_EQ LT LTE GT GTE')


@dataclass
class CmpOp(Expr):
    OPS = {
        ast.Eq: CmpOpcode.EQ,
        ast.NotEq: CmpOpcode.NOT_EQ,
        ast.Lt: CmpOpcode.LT,
        ast.LtE: CmpOpcode.LTE,
        ast.Gt: CmpOpcode.GT,
        ast.GtE: CmpOpcode.GTE,
    }


class ASTEncoder(json.JSONEncoder):
    """JSON serializer for compiled ASTs."""

    # dataclass encoding: https://stackoverflow.com/a/51286749
    def default(self, o):
        if is_dataclass(o):
            # TODO: inject node type.
            return asdict(o)
        return super().default(o)


AST_EXPR_TO_COMPILED = {
    ast.UnaryOp: UnaryOp,
    ast.BoolOp: BoolOp,
    ast.Compare: CmpOp,
    ast.IfExp: IfExpr,
    ast.Constant: Constant,
    ast.Name: Name
}


def type_and_transform_expr(
        expr: ast.Expr,
        ctx: Optional[TypeContext] = None) -> Tuple[type, Expr]:
    if ctx is None:
        ctx = {}
    try:
        return AST_EXPR_TO_COMPILED[type(expr)].type_and_transform(expr, ctx)
    except KeyError:
        raise CompileError(f'expression type {type(expr)} unsupported or TODO')


def type_and_transform_statements(
        statements: List[ast.AST], ctx: TypeContext, return_type: ReturnType
) -> Tuple[TypeDelta, ReturnType, List[Statement]]:
    new_ctx = ctx.copy()
    compiled_statements = []
    delta = {}
    for statement in statements:
        new_return_type = None
        if isinstance(statement, ast.Assign):
            stmt_delta, statement = Assign.type_and_transform(statement, ctx)
        elif isinstance(statement, ast.If):
            stmt_delta, new_return_type, statement = If.type_and_transform(
                statement, ctx, return_type)
        elif isinstance(statement, ast.Return):
            new_return_type, statement = Return.type_and_transform(
                statement, ctx)
        else:
            raise CompileError(
                f'Encountered invalid statement (type {type(statement)}).')

        compiled_statements.append(statement)
        for name, t in stmt_delta.items():
            delta[name] = new_ctx[name] = ctx_union(ctx, t)
        if return_type is None and new_return_type is not None:
            return_type = new_return_type
        else:
            return_type = Union[return_type, new_return_type]

    return delta, return_type, compiled_statements


def to_ast(fn: Callable, fn_type: str, graph: Graph, updaters: Updaters):
    """Compiles a function to a GerryOpt AST."""
    if fn_type not in ('accept', 'constraint', 'score'):
        raise CompileError(
            'Can only compile acceptance, constraint, and score functions.')
    column_types = type_updater_columns(graph, updaters)

    fn_context = inspect.getclosurevars(fn)
    raw_fn_ast = load_function_ast(fn)
    fn_ast = preprocess_ast(raw_fn_ast, fn_context)
    for stmt in fn_ast.body:
        if isinstance(stmt, ast.Assign):
            pass
        elif isinstance(stmt, ast.If):
            pass
        elif isinstance(stmt, ast.Return):
            pass

    return fn_ast
