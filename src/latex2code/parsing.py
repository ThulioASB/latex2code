import ast
import difflib
import re

import sympy as sp
from sympy.core.function import UndefinedFunction
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

from .errors import (
    InvalidLaTeXSyntaxError,
    UnsupportedLaTeXFeatureError,
    _line_and_column,
    _syntax_location,
)
from .normalize import VALID_LATEX_COMMANDS, _normalize_latex_string

_LATEX_PARSE_ERRORS = (LaTeXParsingError,)
_LATEX_PARSE_ERRORS = (LaTeXParsingError,)


class _StableSigmoid(sp.Function):
    nargs = 1


class _StableSoftplus(sp.Function):
    nargs = 1


class _StableSwish(sp.Function):
    nargs = 1


class _StableBeta(sp.Function):
    nargs = 2


OPERATORNAME_FUNCTIONS = {
    "erf": sp.erf,
    "erfc": sp.erfc,
    "gamma": sp.gamma,
    "beta": _StableBeta,
    "abs": sp.Abs,
    "sign": sp.sign,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "acot": sp.acot,
    "asec": sp.asec,
    "acsc": sp.acsc,
    "coth": sp.coth,
    "sech": sp.sech,
    "csch": sp.csch,
    "hypot": lambda *args: sp.sqrt(sum(arg**2 for arg in args)),
    "sigmoid": _StableSigmoid,
    "softplus": _StableSoftplus,
    "relu": lambda value: sp.Max(value, 0),
    "logit": lambda value: sp.log(value / (1 - value)),
    "softsign": lambda value: value / (1 + sp.Abs(value)),
    "swish": _StableSwish,
}


def _find_matching_parenthesis(text: str, start_index: int) -> int | None:
    """Finds the closing parenthesis matching the opening one at start_index."""
    depth = 0
    for idx in range(start_index, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _split_top_level_commas(text: str) -> list[str]:
    """Split a comma-separated argument list while preserving nested parentheses/brackets."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    bracket_depth = 0
    brace_depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        if char == "," and depth == 0 and bracket_depth == 0 and brace_depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part != ""]


def _parse_latex(expression: str) -> sp.Expr:
    return parse_latex(expression, strict=True, backend="antlr")


def _parse_operatorname_expression(latex_str: str) -> sp.Expr:
    r"""Parses operatorname-style functions such as \operatorname{erf}(x)."""
    expression = _normalize_latex_string(latex_str)
    replacements: dict[str, sp.Expr] = {}

    while True:
        match = re.search(r"\\operatorname\s*\{([^}]+)\}", expression)
        if not match:
            break

        func_name = match.group(1)
        open_index = match.end()
        while open_index < len(expression) and expression[open_index].isspace():
            open_index += 1
        if open_index >= len(expression) or expression[open_index] != "(":
            break

        close_index = _find_matching_parenthesis(expression, open_index)
        if close_index is None:
            break

        args_text = expression[open_index + 1 : close_index]
        if args_text:
            arg_exprs = [
                _parse_operatorname_expression(_normalize_latex_string(arg.strip()))
                for arg in _split_top_level_commas(args_text)
            ]
        else:
            arg_exprs = [sp.Symbol("x")]

        if func_name == "hypot":
            value = (
                sp.sqrt(sum(arg**2 for arg in arg_exprs))
                if arg_exprs
                else sp.Symbol("x")
            )
        elif (
            func_name in {"sigmoid", "softplus", "relu", "logit", "softsign", "swish"}
            and len(arg_exprs) == 1
        ):
            value = OPERATORNAME_FUNCTIONS[func_name](arg_exprs[0])
        else:
            factory = OPERATORNAME_FUNCTIONS.get(func_name, sp.Function(func_name))
            if len(arg_exprs) == 1:
                value = factory(arg_exprs[0])
            else:
                value = factory(*arg_exprs)

        replacement_name = next(
            (
                chr(codepoint)
                for codepoint in range(ord("A"), ord("Z") + 1)
                if chr(codepoint) not in expression
            ),
            None,
        )
        if replacement_name is None:
            raise UnsupportedLaTeXFeatureError(
                latex_str, "too many uppercase symbols to safely normalize operatorname"
            )
        replacements[replacement_name] = value
        expression = (
            expression[: match.start()]
            + replacement_name
            + expression[close_index + 1 :]
        )
    parsed = _parse_latex(expression)
    for symbol_name, value in replacements.items():
        parsed = parsed.subs(sp.Symbol(symbol_name), value)
    return parsed


def _split_environment_rows(content: str, content_offset: int) -> list[tuple[str, int]]:
    rows = []
    start = 0
    for separator in re.finditer(r"\\\\{1,2}", content):
        segment = content[start : separator.start()]
        leading = len(segment) - len(segment.lstrip())
        row = segment.strip()
        if row:
            rows.append((row, content_offset + start + leading))
        start = separator.end()
    segment = content[start:]
    leading = len(segment) - len(segment.lstrip())
    row = segment.strip()
    if row:
        rows.append((row, content_offset + start + leading))
    return rows


def _parse_latex_segment(segment: str, raw_expression: str, offset: int) -> sp.Expr:
    try:
        return _parse_latex(segment)
    except _LATEX_PARSE_ERRORS as exc:
        relative_location = _syntax_location(segment, exc)
        if relative_location[0] is None or relative_location[1] is None:
            relative_location = (1, 1)
        assert relative_location[0] is not None
        assert relative_location[1] is not None
        relative_index = (
            sum(
                len(line) + 1
                for line in segment.splitlines()[: relative_location[0] - 1]
            )
            + relative_location[1]
            - 1
        )
        raise InvalidLaTeXSyntaxError(
            raw_expression,
            original_error=exc,
            location=_line_and_column(raw_expression, offset + relative_index),
        ) from exc


def _parse_matrix_environment(latex_str: str) -> sp.Matrix | None:
    """Detects and parses matrix-style environments directly into a SymPy Matrix."""
    pattern = (
        r"\\begin\{(?P<environment>pmatrix|matrix|bmatrix|Bmatrix|vmatrix|Vmatrix|array|aligned|align|align\*|gathered|eqnarray|eqnarray\*)\}"
        r"(?:\{.*?\})?"
        r"(?P<content>.*?)\\end\{(?P=environment)\}"
    )
    stripped_expression = latex_str.strip()
    expression_offset = len(latex_str) - len(latex_str.lstrip())
    match = re.fullmatch(pattern, stripped_expression, re.DOTALL)
    if not match:
        return None

    content = match.group("content")
    content_offset = expression_offset + match.start("content")
    rows = _split_environment_rows(content, content_offset)
    matrix_rows = []
    row_offsets = []
    try:
        for row, row_offset in rows:
            elements = []
            segment_start = 0
            for segment in row.split("&"):
                raw_start = row.find(segment, segment_start)
                segment_start = raw_start + len(segment) + 1
                leading = len(segment) - len(segment.lstrip())
                elem = segment.strip()
                if not elem:
                    continue
                elem_offset = row_offset + raw_start + leading
                if elem.startswith("="):
                    elem = elem[1:].strip()
                    elem_offset += 1
                if elem.startswith("=="):
                    elem = elem[2:].strip()
                    elem_offset += 2
                elements.append((elem, elem_offset))
            if not elements:
                continue
            parsed_elements = [
                _parse_latex_segment(elem, latex_str, elem_offset)
                for elem, elem_offset in elements
                if elem
            ]
            if not parsed_elements:
                continue
            matrix_rows.append(parsed_elements)
            row_offsets.append(row_offset)
        if not matrix_rows:
            return None
        expected_width = len(matrix_rows[0])
        inconsistent_row = next(
            (
                index
                for index, row in enumerate(matrix_rows[1:], start=1)
                if len(row) != expected_width
            ),
            None,
        )
        if inconsistent_row is not None:
            raise InvalidLaTeXSyntaxError(
                latex_str,
                hint="Every row in a matrix or array must contain the same number of entries.",
                location=_line_and_column(latex_str, row_offsets[inconsistent_row]),
            )
        return sp.Matrix(matrix_rows)
    except InvalidLaTeXSyntaxError:
        raise


def _parse_cases_environment(latex_str: str) -> sp.Expr | None:
    r"""Parses piecewise expressions from \begin{cases} ... \end{cases} into SymPy Piecewise."""
    pattern = r"\\begin\{cases\}(.*?)\\end\{cases\}"
    stripped_expression = latex_str.strip()
    expression_offset = len(latex_str) - len(latex_str.lstrip())
    match = re.fullmatch(pattern, stripped_expression, re.DOTALL)
    if not match:
        return None

    content = match.group(1)
    rows = _split_environment_rows(
        content,
        expression_offset + match.start(1),
    )
    if not rows:
        return None

    pieces = []
    try:
        for row, row_offset in rows:
            if "&" not in row:
                raise InvalidLaTeXSyntaxError(
                    latex_str,
                    hint="Each cases row must separate its value and condition with '&'.",
                    location=_line_and_column(latex_str, row_offset),
                )
            separator = row.index("&")
            expression_segment = row[:separator]
            condition_segment = row[separator + 1 :]
            expression_leading = len(expression_segment) - len(
                expression_segment.lstrip()
            )
            condition_leading = len(condition_segment) - len(condition_segment.lstrip())
            expr_part = expression_segment.strip()
            cond_part = condition_segment.strip()
            expr = _parse_latex_segment(
                expr_part,
                latex_str,
                row_offset + expression_leading,
            )
            cond = (
                True
                if cond_part.lower() in {"otherwise", "else"}
                else _parse_latex_segment(
                    cond_part,
                    latex_str,
                    row_offset + separator + 1 + condition_leading,
                )
            )
            pieces.append((expr, cond))
        if not pieces:
            return None

        return sp.Piecewise(*pieces, evaluate=False)
    except InvalidLaTeXSyntaxError:
        raise


PYTHON_FUNCTIONS = {
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "max": sp.Max,
    "min": sp.Min,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "gamma": sp.gamma,
    "Gamma": sp.gamma,
    "beta": _StableBeta,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "acot": sp.acot,
    "asec": sp.asec,
    "acsc": sp.acsc,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "coth": sp.coth,
    "sech": sp.sech,
    "csch": sp.csch,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "sigmoid": _StableSigmoid,
    "softplus": _StableSoftplus,
    "relu": lambda value: sp.Max(value, 0),
    "logit": lambda value: sp.log(value / (1 - value)),
    "softsign": lambda value: value / (1 + sp.Abs(value)),
    "swish": _StableSwish,
}


def _python_ast_to_sympy(node: ast.AST) -> sp.Expr:
    if isinstance(node, ast.Expression):
        return _python_ast_to_sympy(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sp.sympify(node.value)
    if isinstance(node, ast.Name):
        if node.id in {"oo", "infty"}:
            return sp.oo
        return sp.Symbol(node.id)
    if isinstance(node, ast.BinOp):
        left = _python_ast_to_sympy(node.left)
        right = _python_ast_to_sympy(node.right)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left**right,
            ast.BitXor: lambda: left**right,
            ast.Mod: lambda: left % right,
        }
        operation = operations.get(type(node.op))
        if operation is not None:
            return operation()
    if isinstance(node, ast.UnaryOp):
        operand = _python_ast_to_sympy(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            return sp.Not(operand)
    if isinstance(node, ast.BoolOp):
        values = [_python_ast_to_sympy(value) for value in node.values]
        if isinstance(node.op, ast.And):
            return sp.And(*values)
        if isinstance(node.op, ast.Or):
            return sp.Or(*values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = PYTHON_FUNCTIONS.get(node.func.id)
        if function is not None and not node.keywords:
            return function(*[_python_ast_to_sympy(arg) for arg in node.args])
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and len(node.comparators) == 1
    ):
        left = _python_ast_to_sympy(node.left)
        right = _python_ast_to_sympy(node.comparators[0])
        comparisons = {
            ast.Eq: sp.Eq,
            ast.NotEq: sp.Ne,
            ast.Lt: sp.Lt,
            ast.LtE: sp.Le,
            ast.Gt: sp.Gt,
            ast.GtE: sp.Ge,
        }
        comparison = comparisons.get(type(node.ops[0]))
        if comparison is not None:
            return comparison(left, right)
    raise ValueError(
        "Expression contains syntax unsupported by the safe fallback parser."
    )


def _parse_nth_root_expression(latex_str: str) -> sp.Expr | None:
    """Parses expressions such as \\sqrt[n]{x} into x**(1/n)."""
    pattern = r"\\sqrt\s*\[(?P<index>.+?)\]\s*\{(?P<argument>.+)\}"
    match = re.fullmatch(pattern, latex_str.strip(), re.DOTALL)
    if not match:
        return None

    index_expr = _parse_latex(match.group("index").strip())
    argument_expr = _parse_latex(match.group("argument").strip())
    return argument_expr ** (1 / index_expr)


def _replace_extrema_calls(expression: str) -> tuple[str, dict[str, sp.Expr]]:
    replacements: dict[str, sp.Expr] = {}
    while match := re.search(r"\\(max|min)(?![A-Za-z])\s*\(", expression):
        open_index = expression.find("(", match.start())
        close_index = _find_matching_parenthesis(expression, open_index)
        if close_index is None:
            break
        arguments = _split_top_level_commas(expression[open_index + 1 : close_index])
        parsed_arguments = [_parse_expression(argument) for argument in arguments]
        function = sp.Max if match.group(1) == "max" else sp.Min
        value = function(*parsed_arguments)
        replacement_name = next(
            (
                chr(codepoint)
                for codepoint in range(ord("A"), ord("Z") + 1)
                if chr(codepoint) not in expression
            ),
            None,
        )
        if replacement_name is None:
            raise UnsupportedLaTeXFeatureError(
                expression, "too many uppercase symbols to safely normalize extrema"
            )
        replacements[replacement_name] = value
        expression = (
            expression[: match.start()]
            + replacement_name
            + expression[close_index + 1 :]
        )
    return expression, replacements


def _parse_expression(latex_str: str) -> sp.Expr:
    """Use the restricted Python fallback only for structurally plain call syntax."""
    normalized = _normalize_latex_string(latex_str)
    if normalized in {"oo", "infty"}:
        return sp.oo

    nth_root_expr = _parse_nth_root_expression(normalized)
    if nth_root_expr is not None:
        return nth_root_expr
    normalized, extrema_replacements = _replace_extrema_calls(normalized)

    plain_python_calls = (
        "\\" not in normalized
        and re.search(r"\b[A-Za-z_]\w*\s*\(", normalized) is not None
    )
    if plain_python_calls:
        try:
            parsed = ast.parse(normalized, mode="eval")
        except SyntaxError:
            pass
        else:
            try:
                expression = _python_ast_to_sympy(parsed)
            except ValueError as exc:
                raise InvalidLaTeXSyntaxError(
                    latex_str,
                    original_error=exc,
                ) from exc
            return expression.xreplace(
                {sp.Symbol(name): value for name, value in extrema_replacements.items()}
            )
    try:
        expression = _parse_latex(normalized)
    except _LATEX_PARSE_ERRORS as parser_error:
        if not plain_python_calls:
            raise
        try:
            parsed = ast.parse(normalized, mode="eval")
        except SyntaxError:
            raise parser_error
        try:
            expression = _python_ast_to_sympy(parsed)
        except ValueError as exc:
            raise InvalidLaTeXSyntaxError(
                latex_str,
                original_error=exc,
            ) from exc
    return expression.xreplace(
        {sp.Symbol(name): value for name, value in extrema_replacements.items()}
    )


def _ordered_free_symbols(expr: sp.Expr, raw_latex: str) -> list[str]:
    """Select arguments in their first-appearance order in the source."""
    free_symbols = [str(symbol) for symbol in expr.free_symbols]
    ordered = []
    seen = set()
    for token in re.findall(r"[A-Za-z_]\w*", raw_latex):
        if token in free_symbols and token not in seen:
            ordered.append(token)
            seen.add(token)
    ordered.extend(symbol for symbol in free_symbols if symbol not in seen)
    return ordered


def _check_for_undefined_commands(expr: sp.Expr, raw_str: str) -> None:
    """Check if the raw LaTeX string or parsed expression contains unsupported commands."""
    unhandled_commands = re.findall(r"\\[A-Za-z]+", raw_str)

    for cmd in unhandled_commands:
        if cmd not in VALID_LATEX_COMMANDS:
            match = difflib.get_close_matches(
                cmd, VALID_LATEX_COMMANDS, n=1, cutoff=0.65
            )
            suggestion = repr(match[0]) if match else None
            raise UnsupportedLaTeXFeatureError(raw_str, cmd, suggestion=suggestion)

    for func in expr.atoms(sp.Function):
        if isinstance(func.func, UndefinedFunction):
            name = func.func.__name__
            known_functions = set(PYTHON_FUNCTIONS) | set(OPERATORNAME_FUNCTIONS)
            match = difflib.get_close_matches(name, known_functions, n=1, cutoff=0.65)
            suggestion = (
                f"use {match[0]!r} or add support for {name!r}"
                if match
                else f"{name!r} is not in the supported function set"
            )
            raise UnsupportedLaTeXFeatureError(
                raw_str, name, "function", suggestion=suggestion
            )
