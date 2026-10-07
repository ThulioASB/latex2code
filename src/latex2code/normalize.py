import re

LATEX_COMMAND_TRANSLATIONS = {
    r"\coth": "coth",
    r"\sech": "sech",
    r"\csch": "csch",
    r"\infty": "oo",
    r"\equiv": "==",
    r"\propto": "~",
    r"\sim": "~",
    r"\land": " and ",
    r"\lor": " or ",
    r"\not": " not ",
    r"\asinh": "asinh",
    r"\acosh": "acosh",
    r"\atanh": "atanh",
}

LATEX_SYMBOL_COMMANDS = {
    r"\frac",
    r"\sin",
    r"\cos",
    r"\tan",
    r"\cot",
    r"\sec",
    r"\csc",
    r"\max",
    r"\min",
    r"\sinh",
    r"\cosh",
    r"\tanh",
    r"\asin",
    r"\acos",
    r"\atan",
    r"\arcsin",
    r"\arccos",
    r"\arctan",
    r"\arccot",
    r"\arcsec",
    r"\arccsc",
    r"\asinh",
    r"\acosh",
    r"\atanh",
    r"\coth",
    r"\sech",
    r"\csch",
    r"\sqrt",
    r"\sum",
    r"\prod",
    r"\lim",
    r"\binom",
    r"\Gamma",
    r"\beta",
    r"\left",
    r"\right",
    r"\exp",
    r"\log",
    r"\ln",
    r"\int",
    r"\partial",
    r"\pm",
    r"\mp",
    r"\times",
    r"\cdot",
    r"\div",
    r"\ge",
    r"\le",
    r"\neq",
    r"\neg",
    r"\infty",
    r"\pi",
    r"\theta",
    r"\alpha",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\varepsilon",
    r"\zeta",
    r"\eta",
    r"\kappa",
    r"\lambda",
    r"\mu",
    r"\nu",
    r"\xi",
    r"\rho",
    r"\sigma",
    r"\tau",
    r"\upsilon",
    r"\phi",
    r"\varphi",
    r"\chi",
    r"\psi",
    r"\omega",
    r"\Delta",
    r"\Lambda",
    r"\Sigma",
    r"\Theta",
    r"\Omega",
    r"\Phi",
    r"\Pi",
    r"\Psi",
    r"\Xi",
    r"\Upsilon",
    r"\operatorname",
    r"\mathrm",
    r"\text",
    r"\textbf",
    r"\begin",
    r"\end",
    r"\vert",
    r"\mid",
    r"\lvert",
    r"\rvert",
    r"\langle",
    r"\rangle",
    r"\lfloor",
    r"\rfloor",
    r"\lceil",
    r"\rceil",
    r"\to",
    r"\quad",
    r"\qquad",
    r"\vspace",
    r"\hspace",
}

LATEX_ENVIRONMENTS = {
    "cases",
    "matrix",
    "pmatrix",
    "bmatrix",
    "Bmatrix",
    "vmatrix",
    "Vmatrix",
    "array",
    "aligned",
    "align",
    "align*",
    "gathered",
    "eqnarray",
    "eqnarray*",
}

VALID_LATEX_COMMANDS = LATEX_SYMBOL_COMMANDS | set(LATEX_COMMAND_TRANSLATIONS)


def _normalize_latex_string(latex_str: str) -> str:
    """Normalizes known LaTeX wrappers and function-style operators into SymPy-friendly syntax."""
    normalized = latex_str.strip()

    normalized = re.sub(r"\\mathrm\s*\{([A-Za-z]+)\}", r"\1", normalized)
    command_pattern = re.compile(
        "(?:"
        + "|".join(
            re.escape(command)
            for command in sorted(LATEX_COMMAND_TRANSLATIONS, key=len, reverse=True)
        )
        + ")"
        + r"(?![A-Za-z])"
    )
    normalized = command_pattern.sub(
        lambda match: LATEX_COMMAND_TRANSLATIONS[match.group(0)],
        normalized,
    )

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

    normalized = re.sub(
        r"\\left\s*\|\s*(.+?)\s*\\right\s*\|", r"Abs(\1)", normalized, flags=re.DOTALL
    )
    normalized = re.sub(
        r"\\left\s*\\lvert\s*(.+?)\s*\\right\s*\\rvert",
        r"Abs(\1)",
        normalized,
        flags=re.DOTALL,
    )
    normalized = re.sub(r"\\left\s*[\[(]\s*", "(", normalized)
    normalized = re.sub(r"\\right\s*[\])]\s*", ")", normalized)
    normalized = re.sub(r"\\left\s*\\\{\s*", "(", normalized)
    normalized = re.sub(r"\\right\s*\\\}\s*", ")", normalized)
    normalized = re.sub(r"\\(?:left|right)\b", "", normalized)
    normalized = re.sub(r"\\(?:vert|mid|lvert|rvert)\b", "|", normalized)
    return normalized
