import keyword
import re
import sympy as sp
from sympy.core.function import UndefinedFunction
from sympy.parsing.latex import parse_latex


class LaTeXTranspilerError(Exception):
    """Base exception raised for errors during LaTeX transpilation."""

    pass


class InvalidPythonIdentifierError(LaTeXTranspilerError):
    """Raised when generated Python would contain an invalid identifier."""

    def __init__(self, identifier: object, kind: str):
        super().__init__(f"{kind} must be a valid non-keyword Python identifier: {identifier!r}.")


VALID_LATEX_COMMANDS = {
    r"\frac", r"\sin", r"\cos", r"\tan", r"\sqrt", r"\sum", r"\prod",
    r"\left", r"\right", r"\begin", r"\end", r"\exp", r"\log", r"\ln",
    r"\int", r"\diff", r"\partial", r"\pm", r"\mp", r"\times", r"\cdot",
    r"\div", r"\ge", r"\le", r"\neq", r"\pi", r"\theta", r"\alpha",
    r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon", r"\zeta",
    r"\eta", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi", r"\rho",
    r"\sigma", r"\tau", r"\upsilon", r"\phi", r"\varphi", r"\chi",
    r"\psi", r"\omega", r"\Gamma", r"\Delta", r"\Lambda", r"\Sigma",
    r"\Theta", r"\Omega", r"\Phi", r"\Pi", r"\Psi", r"\Xi", r"\Upsilon",
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


def _validate_python_identifier(identifier: object, kind: str) -> None:
    if (
        not isinstance(identifier, str)
        or not identifier.isidentifier()
        or keyword.iskeyword(identifier)
    ):
        raise InvalidPythonIdentifierError(identifier, kind)


def _parse_matrix_environment(latex_str: str) -> sp.Matrix | None:
    """Detects and parses matrix environments directly into a SymPy Matrix."""
    pattern = r"\\begin\{(?:pmatrix|matrix|bmatrix)\}(.*?)\\end\{(?:pmatrix|matrix|bmatrix)\}"
    match = re.search(pattern, latex_str.strip(), re.DOTALL)
    if not match:
        return None

    content = match.group(1).strip()
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


def _check_for_undefined_commands(expr: sp.Expr, raw_str: str) -> None:
    """Check if the raw LaTeX string or parsed expression contains unsupported commands."""
    unhandled_commands = re.findall(r"\\[A-Za-z]+", raw_str)

    for cmd in unhandled_commands:
        if cmd not in VALID_LATEX_COMMANDS:
            raise InvalidLaTeXSyntaxError(raw_str)

    for func in expr.atoms(sp.Function):
        if isinstance(func.func, UndefinedFunction):
            raise InvalidLaTeXSyntaxError(raw_str)


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

    # 1. Parse matrix or general expression
    try:
        expr = _parse_matrix_environment(latex_str)
        is_matrix = expr is not None

        if not is_matrix:
            _check_for_undefined_commands(sp.Symbol("temp"), latex_str)
            expr = parse_latex(latex_str.strip())
            _check_for_undefined_commands(expr, latex_str)

            # Evaluate derivatives or integrals symbolically if present
            if expr.has(sp.Derivative) or expr.has(sp.Integral):
                expr = expr.doit()
    except InvalidLaTeXSyntaxError:
        raise
    except Exception as exc:
        raise InvalidLaTeXSyntaxError(latex_str, original_error=exc) from exc

    # 2. Extract free variables
    variables = sorted([str(symbol) for symbol in expr.free_symbols])
    for variable in variables:
        _validate_python_identifier(variable, "Generated argument name")

    # 3. Build argument signature
    if type_hints:
        type_str = "np.ndarray" if (use_numpy or is_matrix) else "float"
        args_str = ", ".join([f"{var}: {type_str}" for var in variables])
        return_hint = f" -> {type_str}"
    else:
        args_str = ", ".join(variables)
        return_hint = ""

    # 4. Generate Python code representation
    if is_matrix:
        matrix_list = expr.tolist()
        python_expr_code = f"np.array({matrix_list})"
    elif use_numpy:
        python_expr_code = sp.pycode(expr, user_functions={}, fully_qualified_modules=False)
        for fn in ["sin", "cos", "tan", "sqrt", "exp", "log", "pi"]:
            python_expr_code = re.sub(rf"\bmath\.{fn}\b", f"np.{fn}", python_expr_code)
    else:
        python_expr_code = sp.pycode(expr)

    # 5. Assemble required imports
    imports_list = []
    if "builtins." in python_expr_code:
        imports_list.append("import builtins")

    if use_numpy or is_matrix or "np." in python_expr_code:
        imports_list.append("import numpy as np")
    elif "math." in python_expr_code or any(
        fn in python_expr_code for fn in ["sin(", "cos(", "sqrt(", "exp(", "tan(", "pi"]
    ):
        imports_list.append("import math")

    imports = "\n".join(imports_list)
    if imports:
        imports += "\n\n"

    # 6. Construct and return complete source code
    code = f"{imports}def {function_name}({args_str}){return_hint}:\n"
    code += f"    return {python_expr_code}\n"

    return code