import ast
import difflib
import keyword
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Mapping
import sympy as sp
from sympy.core.function import UndefinedFunction
from sympy.parsing.latex import parse_latex
from sympy.printing.pycode import PythonCodePrinter
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
    r"\acot", r"\arccot", r"\asec", r"\arcsec", r"\acsc", r"\arccsc",
    r"\asinh", r"\acosh", r"\atanh", r"\sqrt", r"\sum", r"\prod",
    r"\lim", r"\max", r"\min", r"\binom", r"\Gamma", r"\beta",
    r"\left", r"\right", r"\begin", r"\end", r"\exp", r"\log", r"\ln",
    r"\int", r"\diff", r"\partial", r"\pm", r"\mp", r"\times", r"\cdot",
    r"\div", r"\ge", r"\le", r"\neq", r"\approx", r"\equiv", r"\propto",
    r"\sim", r"\land", r"\lor", r"\neg", r"\not", r"\infty", r"\pi",
    r"\theta", r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
    r"\varepsilon", r"\zeta", r"\eta", r"\kappa", r"\lambda", r"\mu",
    r"\nu", r"\xi", r"\rho", r"\sigma", r"\tau", r"\upsilon", r"\phi",
    r"\varphi", r"\chi", r"\psi", r"\omega", r"\Gamma", r"\Delta",
    r"\Lambda", r"\Sigma", r"\Theta", r"\Omega", r"\Phi", r"\Pi",
    r"\Psi", r"\Xi", r"\Upsilon", r"\operatorname", r"\mathrm",
    r"\cases", r"\array", r"\align", r"\align*", r"\aligned",
    r"\gathered", r"\eqnarray", r"\eqnarray*", r"\vert", r"\mid",
    r"\lvert", r"\rvert", r"\langle", r"\rangle", r"\lfloor",
    r"\rfloor", r"\lceil", r"\rceil", r"\to",
}


class InvalidLaTeXSyntaxError(LaTeXTranspilerError):
    """Raised when the provided LaTeX string cannot be parsed."""

    def __init__(
        self,
        raw_expression: str,
        original_error: Exception | None = None,
        hint: str | None = None,
        location: tuple[int, int] | None = None,
    ):
        self.raw_expression = raw_expression
        self.original_error = original_error
        hint = hint or _syntax_hint(raw_expression)
        self.line, self.column = location or _syntax_location(raw_expression, original_error)
        message = (
            f"Failed to parse LaTeX expression: '{raw_expression}'. "
            "Please verify the expression syntax and supported LaTeX subset."
        )
        if self.line is not None and self.column is not None:
            message += (
                f" Location: line {self.line}, column {self.column}."
                f"\n{_source_context(raw_expression, self.line, self.column)}"
            )
        if hint:
            message += f" Hint: {hint}"
        super().__init__(message)


class UnsupportedLaTeXFeatureError(InvalidLaTeXSyntaxError):
    """Raised when syntax is valid-looking but uses an unsupported command or function."""

    def __init__(
        self,
        raw_expression: str,
        feature: str,
        kind: str = "LaTeX command",
        suggestion: str | None = None,
    ):
        self.raw_expression = raw_expression
        self.original_error = None
        self.feature = feature
        self.kind = kind
        location_index = raw_expression.find(feature)
        if location_index < 0 and feature.startswith("\\"):
            location_index = raw_expression.find(feature[1:])
        self.line, self.column = (
            _line_and_column(raw_expression, location_index)
            if location_index >= 0
            else (None, None)
        )
        message = f"Unsupported {kind}: {feature}."
        if self.line is not None and self.column is not None:
            message += (
                f" Location: line {self.line}, column {self.column}."
                f"\n{_source_context(raw_expression, self.line, self.column)}"
            )
        if suggestion:
            message += f" Did you mean {suggestion}?"
        LaTeXTranspilerError.__init__(self, message)


class CodeGenerationError(LaTeXTranspilerError):
    """Raised when a parsed expression cannot be represented as executable Python."""


class PiecewiseEvaluationWarning(UserWarning):
    """Warns when array backends eagerly evaluate all branches of a piecewise expression."""


@dataclass(frozen=True)
class TranspilationInfo:
    """Parsed expression and generated source returned by :func:`inspect_latex`."""

    input_expression: str
    backend: str
    parsed_expression: str
    expression_tree: dict[str, object]
    variables: tuple[str, ...]
    generated_code: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CompiledFormula:
    """Callable generated formula together with its source and inspection details."""

    info: TranspilationInfo
    function: Callable[..., object]

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.function(*args, **kwargs)

    @property
    def source(self) -> str:
        return self.info.generated_code


def _line_and_column(expression: str, offset: int) -> tuple[int, int]:
    line = expression.count("\n", 0, offset) + 1
    line_start = expression.rfind("\n", 0, offset) + 1
    return line, offset - line_start + 1


def _source_context(expression: str, line: int, column: int) -> str:
    lines = expression.splitlines() or [expression]
    source_line = lines[line - 1] if line <= len(lines) else ""
    start = max(0, column - 61)
    end = min(len(source_line), start + 120)
    prefix = "..." if start else ""
    suffix = "..." if end < len(source_line) else ""
    line_prefix = source_line[start : column - 1]
    caret = " " * (len(prefix) + len(line_prefix.expandtabs())) + "^"
    return f"  {prefix}{source_line[start:end]}{suffix}\n  {caret}"


def _syntax_location(
    expression: str,
    original_error: Exception | None,
) -> tuple[int | None, int | None]:
    delimiters = {"(": ")", "[": "]", "{": "}"}
    stack: list[tuple[str, int]] = []
    for index, character in enumerate(expression):
        if character in delimiters or character in delimiters.values():
            backslashes = 0
            previous = index - 1
            while previous >= 0 and expression[previous] == "\\":
                backslashes += 1
                previous -= 1
            if backslashes % 2:
                continue
            if character in delimiters:
                stack.append((character, index))
            elif not stack or delimiters[stack[-1][0]] != character:
                return _line_and_column(expression, index)
            else:
                stack.pop()
    if stack:
        return _line_and_column(expression, stack[0][1])

    begins = list(re.finditer(r"\\begin\{([^}]+)\}", expression))
    endings = list(re.finditer(r"\\end\{([^}]+)\}", expression))
    begin_names = [match.group(1) for match in begins]
    ending_names = [match.group(1) for match in endings]
    if begin_names != ending_names:
        for begin, ending in zip(begins, endings):
            if begin.group(1) != ending.group(1):
                return _line_and_column(expression, begin.start())
        if len(begins) > len(endings):
            return _line_and_column(expression, begins[len(endings)].start())
        if endings:
            return _line_and_column(expression, endings[0].start())

    if original_error is not None:
        match = re.search(r"\n([^\n]*)\n([~ ]*)\^", str(original_error))
        if match:
            excerpt = match.group(1)
            for line_number, source_line in enumerate(expression.splitlines(), start=1):
                if source_line.strip() == excerpt.strip():
                    return line_number, len(match.group(2)) + 1
        trailing_operator = re.search(r"[+*/^=<>-]\s*$", expression)
        if trailing_operator:
            return _line_and_column(expression, len(expression.rstrip()))
        repeated_operator = re.search(r"[+*/^=<>]\s*([*/^=<>])", expression)
        if repeated_operator:
            return _line_and_column(expression, repeated_operator.start(1))
    return None, None


def _syntax_hint(expression: str) -> str | None:
    for opening, closing, label in (("{", "}", "curly braces"), ("(", ")", "parentheses"), ("[", "]", "square brackets")):
        if expression.count(opening) != expression.count(closing):
            return f"Check that {label} are balanced."

    environments = re.findall(r"\\begin\{([^}]+)\}", expression)
    endings = re.findall(r"\\end\{([^}]+)\}", expression)
    if environments != endings:
        return "Make sure each \\begin{...} has a matching \\end{...} of the same environment."
    if re.search(r"[+*/^=<>-]\s*$", expression):
        return "The expression ends with an operator; add the missing operand or remove the operator."
    if re.search(r"[+*/^=<>]\s*[*/^=<>]", expression):
        return "Check for consecutive operators that do not form a valid expression."
    return None


def _validate_python_identifier(identifier: object, kind: str) -> None:
    if (
        not isinstance(identifier, str)
        or not identifier.isidentifier()
        or keyword.iskeyword(identifier)
    ):
        raise InvalidPythonIdentifierError(identifier, kind)


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
    "hypot": lambda *args: sp.sqrt(sum(arg**2 for arg in args)),
    "sigmoid": _StableSigmoid,
    "softplus": _StableSoftplus,
    "relu": lambda value: sp.Max(value, 0),
    "logit": lambda value: sp.log(value / (1 - value)),
    "softsign": lambda value: value / (1 + sp.Abs(value)),
    "swish": _StableSwish,
}


def _normalize_latex_string(latex_str: str) -> str:
    """Normalizes known LaTeX wrappers and function-style operators into SymPy-friendly syntax."""
    normalized = latex_str.strip()

    normalized = re.sub(r"\\operatorname\s*\{([A-Za-z]+)\}\s*\(", r"\1(", normalized)
    normalized = re.sub(r"\\operatorname\s*\{([A-Za-z]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\s*\{([A-Za-z]+)\}\s*\(", r"\1(", normalized)
    normalized = re.sub(r"\\mathrm\s*\{([A-Za-z]+)\}", r"\1", normalized)
    normalized = normalized.replace(r"\max", "max").replace(r"\min", "min")
    normalized = normalized.replace(r"\infty", "oo").replace(r"\infty", "oo")
    normalized = normalized.replace(r"\approx", "==").replace(r"\equiv", "==")
    normalized = normalized.replace(r"\propto", "~").replace(r"\sim", "~")
    normalized = normalized.replace(r"\land", " and ").replace(r"\lor", " or ")
    normalized = normalized.replace(r"\not", " not ")

    spacing_commands = (
        r"\\(?:,|;|!|:|quad\b|qquad\b|enspace\b|thinspace\b|medspace\b|"
        r"thickspace\b|negthinspace\b|hfill\b|vfill\b|allowbreak\b)"
    )
    normalized = re.sub(
        spacing_commands,
        lambda match: "".join(
            "\n" if character == "\n" else " " for character in match.group(0)
        ),
        normalized,
    )
    sized_spacing_commands = r"\\(?:hspace|vspace)\*?\s*(?:\{[^{}]*\}|\[[^\[\]]*\])"
    normalized = re.sub(
        sized_spacing_commands,
        lambda match: "".join(
            "\n" if character == "\n" else " " for character in match.group(0)
        ),
        normalized,
    )
    style_commands = r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b"
    normalized = re.sub(
        style_commands,
        lambda match: " " * len(match.group(0)),
        normalized,
    )

    normalized = normalized.replace(r"\left[", "(").replace(r"\right]", ")")
    normalized = normalized.replace(r"\left\{", "(").replace(r"\right\}", ")")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")

    normalized = re.sub(r"\\left\s*\|\s*(.+?)\s*\\right\s*\|", r"Abs(\1)", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\\left\s*\\lvert\s*(.+?)\s*\\right\s*\\rvert", r"Abs(\1)", normalized, flags=re.DOTALL)

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
            arg_exprs = [
                _parse_operatorname_expression(_normalize_latex_string(arg.strip()))
                for arg in _split_top_level_commas(args_text)
            ]
        else:
            arg_exprs = [sp.Symbol("x")]

        if func_name == "hypot":
            value = sp.sqrt(sum(arg**2 for arg in arg_exprs)) if arg_exprs else sp.Symbol("x")
        elif func_name in {"sigmoid", "softplus", "relu", "logit", "softsign", "swish"} and len(arg_exprs) == 1:
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
            expression[: match.start()] + replacement_name + expression[close_index + 1 :]
        )

    parsed = parse_latex(expression, strict=True)
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
        return parse_latex(segment, strict=True)
    except Exception as exc:
        relative_location = _syntax_location(segment, exc) or (1, 1)
        relative_index = sum(
            len(line) + 1
            for line in segment.splitlines()[: relative_location[0] - 1]
        ) + relative_location[1] - 1
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
    except Exception as exc:
        raise InvalidLaTeXSyntaxError(latex_str, original_error=exc) from exc


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
            expression_leading = len(expression_segment) - len(expression_segment.lstrip())
            condition_leading = len(condition_segment) - len(condition_segment.lstrip())
            expr_part = expression_segment.strip()
            cond_part = condition_segment.strip()
            expr = _parse_latex_segment(
                expr_part,
                latex_str,
                row_offset + expression_leading,
            )
            cond = True if cond_part.lower() in {"otherwise", "else"} else _parse_latex_segment(
                cond_part,
                latex_str,
                row_offset + separator + 1 + condition_leading,
            )
            pieces.append((expr, cond))
        if not pieces:
            return None

        return sp.Piecewise(*pieces)
    except InvalidLaTeXSyntaxError:
        raise
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

class _PythonPrinter(PythonCodePrinter):
    def _print__StableSigmoid(self, expr: _StableSigmoid) -> str:
        value = self._print(expr.args[0])
        return f"(0.5 * (1 + math.tanh(({value}) / 2)))"

    def _print__StableSoftplus(self, expr: _StableSoftplus) -> str:
        value = self._print(expr.args[0])
        return f"(max(({value}), 0) + math.log1p(math.exp(-abs({value}))))"

    def _print__StableSwish(self, expr: _StableSwish) -> str:
        value = self._print(expr.args[0])
        sigmoid = self._print(_StableSigmoid(expr.args[0]))
        return f"(({value}) * {sigmoid})"

    def _print__StableBeta(self, expr: _StableBeta) -> str:
        first, second = (self._print(value) for value in expr.args)
        sign_first = f"(1 if ({first}) > 0 else math.copysign(1, math.sin(math.pi * ({first}))))"
        sign_second = f"(1 if ({second}) > 0 else math.copysign(1, math.sin(math.pi * ({second}))))"
        total = f"(({first}) + ({second}))"
        sign_total = f"(1 if {total} > 0 else math.copysign(1, math.sin(math.pi * {total})))"
        log_magnitude = (
            f"(math.lgamma({first}) + math.lgamma({second}) - math.lgamma({total}))"
        )
        return f"(({sign_first}) * ({sign_second}) / ({sign_total}) * math.exp({log_magnitude}))"


class _FrameworkPrinter(NumPyPrinter):
    def __init__(self, backend: Literal["numpy", "torch", "jax"]):
        self.backend = backend
        super().__init__(
            {
                "user_functions": {
                    "erf": (
                        "torch.erf"
                        if backend == "torch"
                        else "numpy.vectorize(math.erf)"
                        if backend == "numpy"
                        else "jax.scipy.special.erf"
                    ),
                    "erfc": (
                        "torch.special.erfc"
                        if backend == "torch"
                        else "numpy.vectorize(math.erfc)"
                        if backend == "numpy"
                        else "jax.scipy.special.erfc"
                    ),
                    "gamma": (
                        "torch.special.gamma"
                        if backend == "torch"
                        else "numpy.vectorize(math.gamma)"
                        if backend == "numpy"
                        else "jax.scipy.special.gamma"
                    ),
                }
            }
        )

    def _print_gamma(self, expr: sp.Expr) -> str:
        value = self._print(expr.args[0])
        if self.backend == "torch":
            return (
                f"torch.where(({value}) < 0, "
                f"torch.sign(torch.sin(torch.pi * ({value}))), 1) "
                f"* torch.exp(torch.lgamma({value}))"
            )
        if self.backend == "numpy":
            return f"numpy.vectorize(math.gamma)({value})"
        return f"jax.scipy.special.gamma({value})"

    def _print__StableSigmoid(self, expr: _StableSigmoid) -> str:
        value = self._print(expr.args[0])
        if self.backend == "numpy":
            return f"numpy.exp(-numpy.logaddexp(0, -({value})))"
        module = "torch" if self.backend == "torch" else "jax.nn"
        return f"{module}.sigmoid({value})"

    def _print__StableSoftplus(self, expr: _StableSoftplus) -> str:
        value = self._print(expr.args[0])
        if self.backend == "numpy":
            return f"numpy.logaddexp(0, {value})"
        module = "torch.nn.functional" if self.backend == "torch" else "jax.nn"
        return f"{module}.softplus({value})"

    def _print__StableSwish(self, expr: _StableSwish) -> str:
        value = self._print(expr.args[0])
        return f"({value} * {self._print(_StableSigmoid(expr.args[0]))})"

    def _print__StableBeta(self, expr: _StableBeta) -> str:
        first, second = (self._print(value) for value in expr.args)
        total = f"({first} + {second})"
        if self.backend == "numpy":
            log_magnitude = (
                f"numpy.vectorize(math.lgamma)({first})"
                f" + numpy.vectorize(math.lgamma)({second})"
                f" - numpy.vectorize(math.lgamma)({total})"
            )
            sign = (
                f"numpy.where({first} < 0, numpy.sign(numpy.sin(numpy.pi * {first})), 1)"
                f" * numpy.where({second} < 0, numpy.sign(numpy.sin(numpy.pi * {second})), 1)"
                f" / numpy.where({total} < 0, numpy.sign(numpy.sin(numpy.pi * {total})), 1)"
            )
            return f"({sign} * numpy.exp({log_magnitude}))"
        module = "torch" if self.backend == "torch" else "jnp"
        if self.backend == "torch":
            log_gamma = "torch.lgamma"
            sign_gamma = (
                lambda value: f"torch.where(({value}) < 0, "
                f"torch.sign(torch.sin(torch.pi * ({value}))), 1)"
            )
        else:
            log_gamma = "jax.scipy.special.gammaln"
            sign_gamma = lambda value: f"jax.scipy.special.gammasgn({value})"
        log_magnitude = (
            f"{log_gamma}({first}) + {log_gamma}({second}) - {log_gamma}({total})"
        )
        sign = f"{sign_gamma(first)} * {sign_gamma(second)} / {sign_gamma(total)}"
        return f"({sign} * {module}.exp({log_magnitude}))"

    def _print_Piecewise(self, expr: sp.Piecewise) -> str:
        module = {"numpy": "numpy", "torch": "torch", "jax": "jnp"}[self.backend]
        default = "float('nan')" if self.backend == "torch" else f"{module}.nan"
        branches = list(expr.args)
        for value, condition in reversed(branches):
            printed_value = self._print(value)
            if condition is sp.true:
                default = printed_value
                continue
            printed_condition = self._print(condition).replace("numpy.", f"{module}.")
            default = f"{module}.where({printed_condition}, {printed_value}, {default})"
        return default

    def _print_Max(self, expr: sp.Max) -> str:
        return self._print_extreme(expr, "maximum")

    def _print_Min(self, expr: sp.Min) -> str:
        return self._print_extreme(expr, "minimum")

    def _print_extreme(self, expr: sp.Expr, operation: str) -> str:
        module = {
            "numpy": "numpy",
            "torch": "torch",
            "jax": "jnp",
        }[self.backend]
        args = list(expr.args)
        if self.backend == "torch":
            dynamic_args = [arg for arg in args if not arg.is_number]
            if dynamic_args:
                first = dynamic_args.pop(0)
                result = self._print(first).replace("numpy.", "torch.")
                args.remove(first)
                for arg in args:
                    value = self._print(arg).replace("numpy.", "torch.")
                    if arg.is_number:
                        value = (
                            f"torch.as_tensor({value}, dtype={result}.dtype, "
                            f"device={result}.device)"
                        )
                    result = f"torch.{operation}({result}, {value})"
                return result
        result = self._print(args[0]).replace("numpy.", f"{module}.")
        for arg in args[1:]:
            result = f"{module}.{operation}({result}, {self._print(arg).replace('numpy.', f'{module}.')})"
        return result


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


def _parse_nth_root_expression(latex_str: str) -> sp.Expr | None:
    """Parses expressions such as \\sqrt[n]{x} into x**(1/n)."""
    pattern = r"\\sqrt\s*\[(?P<index>.+?)\]\s*\{(?P<argument>.+)\}"
    match = re.fullmatch(pattern, latex_str.strip(), re.DOTALL)
    if not match:
        return None

    index_expr = parse_latex(match.group("index").strip(), strict=True)
    argument_expr = parse_latex(match.group("argument").strip(), strict=True)
    return argument_expr ** (1 / index_expr)


def _parse_expression(latex_str: str) -> sp.Expr:
    """Parses supported math expressions using SymPy's LaTeX parser and a restricted fallback."""
    normalized = _normalize_latex_string(latex_str)
    if normalized in {"oo", "infty"}:
        return sp.oo

    nth_root_expr = _parse_nth_root_expression(normalized)
    if nth_root_expr is not None:
        return nth_root_expr

    python_signature = re.compile(
        r"(?:Abs\(|(?<!\\)(?:erf|erfc|gamma|Gamma|beta|floor|ceiling|max|min|sin|cos|tan|cot|sec|csc|asin|acos|atan|acot|asec|acsc|sinh|cosh|tanh|coth|sech|csch|asinh|acosh|atanh|exp|log|ln|sqrt|sigmoid|softplus|relu|logit|softsign|swish)\s*\()"
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
            r"\arccot": "acot",
            r"\arcsec": "asec",
            r"\arccsc": "acsc",
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
            r"\approx": "==",
            r"\equiv": "==",
            r"\land": " and ",
            r"\lor": " or ",
            r"\not": " not ",
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

    return parse_latex(normalized, strict=True)


def _ordered_free_symbols(expr: sp.Expr, raw_latex: str) -> list[str]:
    """Select a stable argument order. Matrix-like expressions preserve source order; scalar expressions stay alphabetically sorted."""
    free_symbols = [str(symbol) for symbol in expr.free_symbols]
    if expr.is_Matrix or r"\binom" in raw_latex or "binomial" in str(expr):
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


def _expression_tree(expr: sp.Expr | sp.MatrixBase) -> dict[str, object]:
    if isinstance(expr, sp.MatrixBase):
        return {
            "type": type(expr).__name__,
            "shape": list(expr.shape),
            "args": [
                [_expression_tree(value) for value in row]
                for row in expr.tolist()
            ],
        }
    node: dict[str, object] = {
        "type": getattr(expr.func, "__name__", type(expr).__name__),
    }
    if expr.args:
        node["args"] = [_expression_tree(argument) for argument in expr.args]
    else:
        node["value"] = str(expr)
    return node


def _check_for_undefined_commands(expr: sp.Expr, raw_str: str) -> None:
    """Check if the raw LaTeX string or parsed expression contains unsupported commands."""
    unhandled_commands = re.findall(r"\\[A-Za-z]+", raw_str)

    for cmd in unhandled_commands:
        if cmd not in VALID_LATEX_COMMANDS:
            match = difflib.get_close_matches(cmd, VALID_LATEX_COMMANDS, n=1, cutoff=0.65)
            suggestion = repr(match[0]) if match else None
            raise UnsupportedLaTeXFeatureError(raw_str, cmd, suggestion=suggestion)

    for func in expr.atoms(sp.Function):
        if isinstance(func.func, UndefinedFunction):
            name = func.func.__name__
            known_functions = set(PYTHON_FUNCTIONS) | set(OPERATORNAME_FUNCTIONS)
            match = difflib.get_close_matches(name, known_functions, n=1, cutoff=0.65)
            suggestion = (
                f"use {match[0]!r} or add support for {name!r}" if match else
                f"{name!r} is not in the supported function set"
            )
            raise UnsupportedLaTeXFeatureError(
                raw_str, name, "function", suggestion=suggestion
            )


def _transpile_latex(
    latex_str: str,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool = False,
    backend: Literal["python", "numpy", "torch", "jax"] | None = None,
    variable_map: Mapping[str, str] | None = None,
) -> TranspilationInfo:
    """Build generated source and inspection details from a LaTeX expression."""
    if not latex_str or not latex_str.strip():
        raise LaTeXTranspilerError("LaTeX expression string cannot be empty.")
    _validate_python_identifier(function_name, "Function name")
    if backend is None:
        backend = "numpy" if use_numpy else "python"
    elif not isinstance(backend, str) or backend not in {"python", "numpy", "torch", "jax"}:
        raise LaTeXTranspilerError(
            f"Unknown backend {backend!r}; choose 'python', 'numpy', 'torch', or 'jax'."
        )
    if use_numpy and backend != "numpy":
        raise LaTeXTranspilerError(
            "use_numpy=True conflicts with an explicit backend; use backend='numpy' instead."
        )

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

    expr = expr.subs({sp.Symbol("pi"): sp.pi, sp.Symbol("e"): sp.E})

    # 2. Extract free variables
    variables = _ordered_free_symbols(expr, normalized_latex)
    if variable_map is not None:
        if not isinstance(variable_map, Mapping):
            raise LaTeXTranspilerError(
                "variable_map must map symbol names to Python identifiers."
            )
        if not all(
            isinstance(source, str) and isinstance(target, str)
            for source, target in variable_map.items()
        ):
            raise LaTeXTranspilerError(
                "variable_map keys and values must be strings."
            )
        unknown_symbols = set(variable_map) - set(variables)
        if unknown_symbols:
            raise LaTeXTranspilerError(
                "variable_map contains symbols not present in the expression: "
                f"{', '.join(sorted(unknown_symbols))}."
            )
        mapped_variables = [
            variable_map.get(variable, variable)
            for variable in variables
        ]
        for variable in mapped_variables:
            _validate_python_identifier(variable, "Mapped argument name")
        if len(set(mapped_variables)) != len(mapped_variables):
            raise LaTeXTranspilerError(
                "variable_map must not map multiple symbols to the same Python identifier."
            )
        expr = expr.xreplace(
            {
                sp.Symbol(source): sp.Symbol(target)
                for source, target in variable_map.items()
            }
        )
        variables = mapped_variables
    transpilation_warnings = []
    for variable in variables:
        _validate_python_identifier(variable, "Generated argument name")
    if backend in {"numpy", "torch", "jax"} and expr.has(sp.Piecewise):
        warning_message = (
            "Array-backend piecewise expressions use eager elementwise selection; "
            "all branch expressions may be evaluated, including branches not selected "
            "for an element. Keep every branch valid over the full input domain."
        )
        transpilation_warnings.append(warning_message)
        warnings.warn(
            warning_message,
            PiecewiseEvaluationWarning,
            stacklevel=2,
        )

    # 3. Build argument signature
    backend_types = {
        "python": ("float", "float"),
        "numpy": ("float | np.ndarray", "np.ndarray" if is_matrix else "float | np.ndarray"),
        "torch": ("torch.Tensor", "torch.Tensor"),
        "jax": ("jax.Array", "jax.Array"),
    }
    argument_type, return_type = backend_types[backend]
    if type_hints:
        if is_matrix and backend == "numpy":
            return_type = "np.ndarray"
        elif is_matrix and backend == "python":
            return_type = "np.ndarray"
        args_str = ", ".join([f"{var}: {argument_type}" for var in variables])
        return_hint = f" -> {return_type}"
    else:
        args_str = ", ".join(variables)
        return_hint = ""

    # 4. Generate Python code representation
    try:
        if is_matrix and backend in {"torch", "jax"}:
            matrix_rows = [
                "[" + ", ".join(
                    _FrameworkPrinter(backend).doprint(value).replace(
                        "numpy.", "torch." if backend == "torch" else "jnp."
                    )
                    for value in row
                ) + "]"
                for row in expr.tolist()
            ]
            if backend == "torch" and variables:
                like = variables[0]
                rows_with_tensors = [
                    "[" + ", ".join(
                        f"({value} if torch.is_tensor({value}) else "
                        f"torch.as_tensor({value}, dtype={like}.dtype, device={like}.device))"
                        for value in row
                    ) + "]"
                    for row in expr.tolist()
                ]
                python_expr_code = (
                    "torch.stack([torch.stack(row) for row in ["
                    + ", ".join(rows_with_tensors)
                    + "]])"
                )
            elif backend == "torch":
                python_expr_code = f"torch.tensor([{', '.join(matrix_rows)}])"
            else:
                python_expr_code = f"jnp.array([{', '.join(matrix_rows)}])"
        elif is_matrix:
            matrix_list = expr.tolist()
            python_expr_code = (
                f"np.array({matrix_list})" if backend == "numpy"
                else f"np.array({matrix_list})"
            )
        elif backend == "numpy":
            python_expr_code = _FrameworkPrinter("numpy").doprint(expr).replace("numpy.", "np.")
        elif backend in {"torch", "jax"}:
            module = "torch" if backend == "torch" else "jnp"
            printer = _FrameworkPrinter(backend)
            python_expr_code = printer.doprint(expr).replace("numpy.", f"{module}.")
        else:
            python_expr_code = _PythonPrinter().doprint(expr)
        if backend in {"torch", "jax"} and "math." in python_expr_code:
            raise CodeGenerationError(
                f"The {backend} backend has no differentiable native implementation for "
                f"part of this expression: {expr}"
            )
    except Exception as exc:
        if isinstance(exc, CodeGenerationError):
            raise
        raise CodeGenerationError(
            f"Unable to generate Python code for parsed expression: {expr}"
        ) from exc

    # 5. Assemble required imports
    imports_list = []
    if "builtins." in python_expr_code:
        imports_list.append("import builtins")

    if (
        backend == "numpy"
        or (backend == "python" and is_matrix)
        or re.search(r"(?<![A-Za-z0-9_])np\.", python_expr_code)
    ):
        imports_list.append("import numpy as np")
    if backend == "torch":
        imports_list.append("import torch")
    if backend == "jax":
        imports_list.extend(("import jax", "import jax.numpy as jnp"))
        if "jax.scipy.special." in python_expr_code:
            imports_list.append("import jax.scipy.special")
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

    return TranspilationInfo(
        input_expression=latex_str,
        backend=backend,
        parsed_expression=str(expr),
        expression_tree=_expression_tree(expr),
        variables=tuple(variables),
        generated_code=code,
        warnings=tuple(transpilation_warnings),
    )


def transpile_latex(
    latex_str: str,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool = False,
    backend: Literal["python", "numpy", "torch", "jax"] | None = None,
    variable_map: Mapping[str, str] | None = None,
) -> str:
    """Convert LaTeX into Python source targeting Python math, NumPy, PyTorch, or JAX."""
    return _transpile_latex(
        latex_str, function_name, type_hints, use_numpy, backend, variable_map
    ).generated_code


def inspect_latex(
    latex_str: str,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool = False,
    backend: Literal["python", "numpy", "torch", "jax"] | None = None,
    variable_map: Mapping[str, str] | None = None,
) -> TranspilationInfo:
    """Return the parsed expression, variables, warnings, and generated source."""
    return _transpile_latex(
        latex_str, function_name, type_hints, use_numpy, backend, variable_map
    )


def compile_latex(
    latex_str: str,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool = False,
    backend: Literal["python", "numpy", "torch", "jax"] | None = None,
    variable_map: Mapping[str, str] | None = None,
) -> CompiledFormula:
    """Compile LaTeX into a callable and retain its generated source and metadata."""
    info = inspect_latex(
        latex_str,
        function_name=function_name,
        type_hints=type_hints,
        use_numpy=use_numpy,
        backend=backend,
        variable_map=variable_map,
    )
    namespace: dict[str, object] = {}
    exec(info.generated_code, namespace)
    function = namespace[function_name]
    if not callable(function):
        raise CodeGenerationError(
            f"Generated object {function_name!r} is not callable."
        )
    return CompiledFormula(info=info, function=function)