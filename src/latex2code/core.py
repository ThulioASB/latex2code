import ast
import keyword
import re
import sympy as sp
from sympy.core.function import UndefinedFunction
from sympy.parsing.latex import parse_latex
from sympy.printing.numpy import NumPyPrinter


class LaTeXTranspilerError(Exception):
    """Base exception raised for errors during LaTeX transpilation."""

    pass


class InvalidPythonIdentifierError(LaTeXTranspilerError):
    """Raised when generated Python would contain an invalid identifier."""

    def __init__(self, identifier: object, kind: str):
        super().__init__(f"{kind} must be a valid non-keyword Python identifier: {identifier!r}.")


VALID_LATEX_COMMANDS = {
    r"\frac", r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
    r"\sinh", r"\cosh", r"\tanh", r"\coth", r"\sech", r"\csch",
    r"\asin", r"\arcsin", r"\acos", r"\arccos", r"\atan", r"\arctan",
    r"\asinh", r"\acosh", r"\atanh", r"\sqrt", r"\sum", r"\prod",
    r"\lim", r"\max", r"\min", r"\binom", r"\Gamma", r"\beta",
    r"\left", r"\right", r"\begin", r"\end", r"\exp", r"\log", r"\ln",
    r"\int", r"\diff", r"\partial", r"\pm", r"\mp", r"\times", r"\cdot",
    r"\div", r"\ge", r"\le", r"\neq", r"\pi", r"\theta", r"\alpha",
    r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon", r"\zeta",
    r"\eta", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi", r"\rho",
    r"\sigma", r"\tau", r"\upsilon", r"\phi", r"\varphi", r"\chi",
    r"\psi", r"\omega", r"\Gamma", r"\Delta", r"\Lambda", r"\Sigma",
    r"\Theta", r"\Omega", r"\Phi", r"\Pi", r"\Psi", r"\Xi", r"\Upsilon",
    r"\operatorname", r"\mathrm", r"\cases", r"\vert", r"\mid", r"\lvert",
    r"\rvert", r"\langle", r"\rangle", r"\lfloor", r"\rfloor", r"\lceil",
    r"\rceil", r"\to",
}


class InvalidLaTeXSyntaxError(LaTeXTranspilerError):
    """Raised when the provided LaTeX string cannot be parsed."""

    def __init__(self, raw_expression: str, original_error: Exception | None = None):
        self.raw_expression = raw_expression
        self.original_error = original_error
        message = (
            f"Failed to parse LaTeX expression: '{raw_expression}'. "
            "Please verify brackets, syntax, and LaTeX mathematical commands."
        )
        super().__init__(message)


class UnsupportedLaTeXFeatureError(InvalidLaTeXSyntaxError):
    """Raised when syntax is valid-looking but uses an unsupported command or function."""

    def __init__(self, raw_expression: str, feature: str, kind: str = "LaTeX command"):
        self.raw_expression = raw_expression
        self.original_error = None
        self.feature = feature
        self.kind = kind
        LaTeXTranspilerError.__init__(self, f"Unsupported {kind}: {feature}.")


class CodeGenerationError(LaTeXTranspilerError):
    """Raised when a parsed expression cannot be represented as executable Python."""


def _validate_python_identifier(identifier: object, kind: str) -> None:
    if (
        not isinstance(identifier, str)
        or not identifier.isidentifier()
        or keyword.iskeyword(identifier)
    ):
        raise InvalidPythonIdentifierError(identifier, kind)


OPERATORNAME_FUNCTIONS = {
    "erf": sp.erf,
    "erfc": sp.erfc,
    "gamma": sp.gamma,
    "beta": sp.beta,
    "abs": sp.Abs,
    "sign": sp.sign,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
}


def _normalize_latex_string(latex_str: str) -> str:
    """Normalizes known LaTeX wrappers and function-style operators into SymPy-friendly syntax."""
    normalized = latex_str.strip()

    normalized = re.sub(r"\\operatorname\s*\{([A-Za-z]+)\}\s*\(", r"\1(", normalized)
    normalized = re.sub(r"\\operatorname\s*\{([A-Za-z]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\s*\{([A-Za-z]+)\}\s*\(", r"\1(", normalized)
    normalized = re.sub(r"\\mathrm\s*\{([A-Za-z]+)\}", r"\1", normalized)
    normalized = normalized.replace(r"\max", "max").replace(r"\min", "min")

    normalized = re.sub(r"\\left\s*\|\s*(.+?)\s*\\right\s*\|", r"Abs(\1)", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\\left\s*\\lvert\s*(.+?)\s*\\right\s*\\rvert", r"Abs(\1)", normalized, flags=re.DOTALL)

    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\vert", "|").replace(r"\mid", "|")
    return normalized


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


def _parse_operatorname_expression(latex_str: str) -> sp.Expr:
    r"""Parses operatorname-style functions such as \operatorname{erf}(x)."""
    expression = latex_str
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
            arg_expr = _parse_operatorname_expression(_normalize_latex_string(args_text))
        else:
            arg_expr = sp.Symbol("x")

        factory = OPERATORNAME_FUNCTIONS.get(func_name, sp.Function(func_name))
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
        replacements[replacement_name] = factory(arg_expr)
        expression = (
            expression[: match.start()] + replacement_name + expression[close_index + 1 :]
        )

    parsed = parse_latex(expression)
    for symbol_name, value in replacements.items():
        parsed = parsed.subs(sp.Symbol(symbol_name), value)
    return parsed


def _parse_matrix_environment(latex_str: str) -> sp.Matrix | None:
    """Detects and parses matrix environments directly into a SymPy Matrix."""
    pattern = (
        r"\\begin\{(?P<environment>pmatrix|matrix|bmatrix|Bmatrix|vmatrix|Vmatrix)\}"
        r"(?P<content>.*?)\\end\{(?P=environment)\}"
    )
    match = re.fullmatch(pattern, latex_str.strip(), re.DOTALL)
    if not match:
        return None

    content = match.group("content").strip()
    rows = [row.strip() for row in re.split(r"\\\\{1,2}", content) if row.strip()]

    matrix_rows = []
    try:
        for row in rows:
            elements = [elem.strip() for elem in row.split("&")]
            parsed_elements = [parse_latex(elem) for elem in elements]
            matrix_rows.append(parsed_elements)
        return sp.Matrix(matrix_rows)
    except Exception as exc:
        raise InvalidLaTeXSyntaxError(latex_str, original_error=exc) from exc


def _parse_cases_environment(latex_str: str) -> sp.Expr | None:
    r"""Parses piecewise expressions from \begin{cases} ... \end{cases} into SymPy Piecewise."""
    pattern = r"\\begin\{cases\}(.*?)\\end\{cases\}"
    match = re.fullmatch(pattern, latex_str.strip(), re.DOTALL)
    if not match:
        return None

    content = match.group(1).strip()
    rows = [row.strip() for row in re.split(r"\\\\{1,2}", content) if row.strip()]
    if not rows:
        return None

    pieces = []
    try:
        for row in rows:
            if "&" not in row:
                raise ValueError(f"Unsupported cases row: {row!r}")
            expr_part, cond_part = [segment.strip() for segment in row.split("&", 1)]
            expr = parse_latex(expr_part)
            cond = parse_latex(cond_part) if cond_part.lower() not in {"otherwise", "else"} else True
            pieces.append((expr, cond))
        if not pieces:
            return None

        return sp.Piecewise(*pieces)
    except Exception as exc:
        raise InvalidLaTeXSyntaxError(latex_str, original_error=exc) from exc


PYTHON_FUNCTIONS = {
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "max": sp.Max,
    "min": sp.Min,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "gamma": sp.gamma,
    "Gamma": sp.gamma,
    "beta": sp.beta,
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
}

NUMPY_PRINTER = NumPyPrinter(
    {
        "user_functions": {
            "erf": "numpy.vectorize(math.erf)",
            "erfc": "numpy.vectorize(math.erfc)",
            "gamma": "numpy.vectorize(math.gamma)",
        }
    }
)


def _python_ast_to_sympy(node: ast.AST) -> sp.Expr:
    if isinstance(node, ast.Expression):
        return _python_ast_to_sympy(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sp.sympify(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sp.pi
        if node.id == "e":
            return sp.E
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
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = PYTHON_FUNCTIONS.get(node.func.id)
        if function is not None and not node.keywords:
            return function(*[_python_ast_to_sympy(arg) for arg in node.args])
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
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
    raise ValueError("Expression contains syntax unsupported by the safe fallback parser.")


def _parse_expression(latex_str: str) -> sp.Expr:
    """Parses supported math expressions using SymPy's LaTeX parser and a restricted fallback."""
    normalized = _normalize_latex_string(latex_str)

    python_signature = re.compile(
        r"(?:Abs\(|(?<!\\)(?:erf|erfc|gamma|Gamma|beta|floor|ceiling|max|min|sin|cos|tan|cot|sec|csc|asin|acos|atan|sinh|cosh|tanh|coth|sech|csch|asinh|acosh|atanh|exp|log|ln|sqrt)\s*\()"
    )
    if python_signature.search(normalized):
        sanitized = normalized
        for command, name in {
            r"\sin": "sin",
            r"\cos": "cos",
            r"\tan": "tan",
            r"\cot": "cot",
            r"\sec": "sec",
            r"\csc": "csc",
            r"\sinh": "sinh",
            r"\cosh": "cosh",
            r"\tanh": "tanh",
            r"\asin": "asin",
            r"\acos": "acos",
            r"\atan": "atan",
            r"\asinh": "asinh",
            r"\acosh": "acosh",
            r"\atanh": "atanh",
            r"\exp": "exp",
            r"\log": "log",
            r"\ln": "ln",
            r"\sqrt": "sqrt",
            r"\pi": "pi",
            r"\theta": "theta",
            r"\Gamma": "Gamma",
            r"\alpha": "alpha",
            r"\beta": "beta",
            r"\gamma": "gamma",
            r"\delta": "delta",
            r"\lambda": "lambda",
            r"\sigma": "sigma",
            r"\phi": "phi",
            r"\psi": "psi",
            r"\omega": "omega",
            r"\mu": "mu",
        }.items():
            sanitized = sanitized.replace(command, name)
        parsed = ast.parse(sanitized, mode="eval")
        return _python_ast_to_sympy(parsed)

    return parse_latex(normalized)


def _ordered_free_symbols(expr: sp.Expr, raw_latex: str) -> list[str]:
    """Selects a stable argument order. For binomial expression, maintain the left-to-right variable order from the source form."""
    free_symbols = [str(symbol) for symbol in expr.free_symbols]
    if r"\binom" in raw_latex or "binomial" in str(expr):
        ordered = []
        seen = set()
        for token in re.findall(r"[A-Za-z_]+", raw_latex):
            if token in free_symbols and token not in seen:
                ordered.append(token)
                seen.add(token)
        for symbol in free_symbols:
            if symbol not in seen:
                ordered.append(symbol)
        return ordered
    return sorted(free_symbols)


def _check_for_undefined_commands(expr: sp.Expr, raw_str: str) -> None:
    """Check if the raw LaTeX string or parsed expression contains unsupported commands."""
    unhandled_commands = re.findall(r"\\[A-Za-z]+", raw_str)

    for cmd in unhandled_commands:
        if cmd not in VALID_LATEX_COMMANDS:
            raise UnsupportedLaTeXFeatureError(raw_str, cmd)

    for func in expr.atoms(sp.Function):
        if isinstance(func.func, UndefinedFunction):
            raise UnsupportedLaTeXFeatureError(raw_str, func.func.__name__, "function")


def transpile_latex(
    latex_str: str,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool = False,
) -> str:
    """Converts a LaTeX string into executable Python function source code."""
    if not latex_str or not latex_str.strip():
        raise LaTeXTranspilerError("LaTeX expression string cannot be empty.")
    _validate_python_identifier(function_name, "Function name")

    normalized_latex = _normalize_latex_string(latex_str)

    # 1. Parse matrix, piecewise, or general expression
    try:
        expr = _parse_matrix_environment(normalized_latex)
        is_matrix = expr is not None

        if not is_matrix:
            expr = _parse_cases_environment(normalized_latex)
            if expr is not None:
                is_matrix = False

        if expr is None:
            _check_for_undefined_commands(sp.Symbol("temp"), normalized_latex)
            if r"\operatorname" in latex_str:
                expr = _parse_operatorname_expression(latex_str)
            else:
                expr = _parse_expression(normalized_latex)
            _check_for_undefined_commands(expr, normalized_latex)

            # Evaluate derivatives, limits, or integrals symbolically if present
            if (
                expr.has(sp.Derivative)
                or expr.has(sp.Integral)
                or expr.has(sp.Limit)
                or expr.has(sp.Product)
            ):
                expr = expr.doit()
    except InvalidLaTeXSyntaxError:
        raise
    except Exception as exc:
        raise InvalidLaTeXSyntaxError(latex_str, original_error=exc) from exc

    # 2. Extract free variables
    variables = _ordered_free_symbols(expr, normalized_latex)
    for variable in variables:
        _validate_python_identifier(variable, "Generated argument name")

    # 3. Build argument signature
    if type_hints:
        if is_matrix:
            argument_type = "float | np.ndarray" if use_numpy else "float"
            return_type = "np.ndarray"
        elif use_numpy:
            argument_type = "float | np.ndarray"
            return_type = "float | np.ndarray"
        else:
            argument_type = return_type = "float"
        args_str = ", ".join([f"{var}: {argument_type}" for var in variables])
        return_hint = f" -> {return_type}"
    else:
        args_str = ", ".join(variables)
        return_hint = ""

    # 4. Generate Python code representation
    try:
        if is_matrix:
            matrix_list = expr.tolist()
            python_expr_code = f"np.array({matrix_list})"
        elif use_numpy:
            python_expr_code = NUMPY_PRINTER.doprint(expr).replace("numpy.", "np.")
        else:
            python_expr_code = sp.pycode(expr)
    except Exception as exc:
        raise CodeGenerationError(
            f"Unable to generate Python code for parsed expression: {expr}"
        ) from exc

    # 5. Assemble required imports
    imports_list = []
    if "builtins." in python_expr_code:
        imports_list.append("import builtins")

    if use_numpy or is_matrix or "np." in python_expr_code:
        imports_list.append("import numpy as np")
    if "math." in python_expr_code:
        imports_list.append("import math")
    if "functools." in python_expr_code:
        imports_list.append("import functools")

    imports = "\n".join(imports_list)
    if imports:
        imports += "\n\n"

    # 6. Construct and return complete source code
    code = f"{imports}def {function_name}({args_str}){return_hint}:\n"
    code += f"    return {python_expr_code}\n"

    return code