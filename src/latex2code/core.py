import re
import sympy as sp
from sympy.core.function import UndefinedFunction
from sympy.parsing.latex import parse_latex


class LaTeXTranspilerError(Exception):
    """Base exception raised for errors during LaTeX transpilation."""

    pass


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
    """Check if the raw LaTeX string or parsed expression contains unknown/unhandled commands."""
    unhandled_commands = re.findall(r"\\[a-zA-Z]+", raw_str)
    valid_commands = {
        r"\frac", r"\sin", r"\cos", r"\tan", r"\sqrt", r"\sum", r"\prod",
        r"\left", r"\right", r"\begin", r"\end", r"\exp", r"\log", r"\ln",
        r"\int", r"\diff", r"\partial"
    }

    for cmd in unhandled_commands:
        if cmd not in valid_commands:
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
    """Converts a LaTeX string into executable Python function source code.

    Args:
        latex_str: The LaTeX math expression.
        function_name: Name of the output function.
        type_hints: Whether to add type annotations.
        use_numpy: If True, uses NumPy arrays/functions.

    Returns:
        String containing valid Python source code.
    """
    if not latex_str or not latex_str.strip():
        raise LaTeXTranspilerError("LaTeX expression string cannot be empty.")

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
    else:
        python_expr_code = sp.pycode(expr)

    # 5. Assemble required imports
    imports_list = []
    if "builtins." in python_expr_code:
        imports_list.append("import builtins")
    if use_numpy or is_matrix:
        imports_list.append("import numpy as np")
    elif any(fn in python_expr_code for fn in ["sin", "cos", "sqrt", "exp", "tan", "log"]):
        imports_list.append("import math")

    imports = "\n".join(imports_list)
    if imports:
        imports += "\n\n"

    # 6. Construct and return complete source code
    code = f"{imports}def {function_name}({args_str}){return_hint}:\n"
    code += f"    return {python_expr_code}\n"

    return code