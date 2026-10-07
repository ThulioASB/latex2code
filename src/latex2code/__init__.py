from importlib.metadata import PackageNotFoundError, version

from latex2code.core import (
    Backend,
    CodeGenerationError,
    CompiledFormula,
    InvalidLaTeXSyntaxError,
    InvalidPythonIdentifierError,
    LaTeXTranspilerError,
    PiecewiseEvaluationWarning,
    TranspilationInfo,
    UnsupportedLaTeXFeatureError,
    compile_latex,
    inspect_latex,
    latex_to_code,
    transpile_latex,
)

try:
    __version__ = version("latex2code")
except PackageNotFoundError:
    __version__ = "0+unknown"
__all__ = [
    "__version__",
    "Backend",
    "transpile_latex",
    "compile_latex",
    "inspect_latex",
    "latex_to_code",
    "CompiledFormula",
    "TranspilationInfo",
    "LaTeXTranspilerError",
    "CodeGenerationError",
    "InvalidLaTeXSyntaxError",
    "UnsupportedLaTeXFeatureError",
    "InvalidPythonIdentifierError",
    "PiecewiseEvaluationWarning",
]
