import keyword
import re

RESERVED_MODULE_NAMES = {
    "builtins",
    "functools",
    "jax",
    "jnp",
    "math",
    "np",
    "numpy",
    "torch",
}


class LaTeXTranspilerError(Exception):
    """Base exception raised for errors during LaTeX transpilation."""

    pass


class InvalidPythonIdentifierError(LaTeXTranspilerError):
    """Raised when generated Python would contain an invalid identifier."""

    def __init__(self, identifier: object, kind: str):
        super().__init__(
            f"{kind} must be a valid non-keyword Python identifier: {identifier!r}."
        )


class _LaTeXSyntaxError(LaTeXTranspilerError):
    """Shared base for errors caused by LaTeX syntax or unsupported features."""

    line: int | None
    column: int | None


class InvalidLaTeXSyntaxError(_LaTeXSyntaxError):
    """Raised when the provided LaTeX string cannot be parsed."""

    def __init__(
        self,
        raw_expression: str,
        original_error: Exception | None = None,
        hint: str | None = None,
        location: tuple[int | None, int | None] | None = None,
    ):
        self.raw_expression = raw_expression
        self.original_error = original_error
        hint = hint or _syntax_hint(raw_expression)
        resolved_location = (
            location
            if location is not None
            else _syntax_location(raw_expression, original_error)
        )
        self.line, self.column = resolved_location
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


class UnsupportedLaTeXFeatureError(_LaTeXSyntaxError):
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
        super().__init__(message)


class CodeGenerationError(LaTeXTranspilerError):
    """Raised when a parsed expression cannot be represented as executable Python."""


class PiecewiseEvaluationWarning(UserWarning):
    """Warns when array backends eagerly evaluate all branches of a piecewise expression."""


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
    for opening, closing, label in (
        ("{", "}", "curly braces"),
        ("(", ")", "parentheses"),
        ("[", "]", "square brackets"),
    ):
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
        or identifier in RESERVED_MODULE_NAMES
    ):
        raise InvalidPythonIdentifierError(identifier, kind)
