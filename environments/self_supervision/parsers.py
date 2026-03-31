from __future__ import annotations

from math import isclose
import re

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


CORRECTNESS_PATTERN = re.compile(r"CORRECTNESS\s*:\s*(YES|NO)", flags=re.IGNORECASE)
CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE\s*:\s*(HIGH|LOW)", flags=re.IGNORECASE)
BOXED_PREFIX = r"\boxed{"
TEXT_WRAPPER_PATTERN = re.compile(r"^\\text\{(.*)\}$")
MBOX_PATTERN = re.compile(r"\\mbox\{.*?\}")
STYLE_WRAPPER_PATTERN = re.compile(r"\\(?:textbf|mathbf|mathrm)\{([^{}]*)\}")
IDENTIFIER_WITH_OPTIONAL_ARGS_PATTERN = re.compile(
    r"^(?:\\?[A-Za-z]+(?:_[A-Za-z0-9{}]+)?)(?:\([^()]*\))?$"
)


def normalize_math_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    text = text.replace("\n", "")
    text = text.rstrip(".")
    text = text.replace("\\!", "")

    text_wrapper_match = TEXT_WRAPPER_PATTERN.fullmatch(text)
    if text_wrapper_match:
        text = text_wrapper_match.group(1).strip()

    wrappers = (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    changed = True
    while changed:
        changed = False
        for prefix, suffix in wrappers:
            if text.startswith(prefix) and text.endswith(suffix):
                text = text[len(prefix) : len(text) - len(suffix)].strip()
                changed = True

    text = text.replace("tfrac", "frac")
    text = text.replace("dfrac", "frac")
    text = text.replace("cfrac", "frac")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    while True:
        updated_text = STYLE_WRAPPER_PATTERN.sub(r"\1", text)
        if updated_text == text:
            break
        text = updated_text

    text = text.replace("\\mathbf", "")
    text = text.replace("\\mathrm", "")
    text = MBOX_PATTERN.sub("", text)
    text = text.replace("\\$", "")
    text = text.replace("$", "")
    text = text.replace("infinity", r"\infty")
    if r"\infty" not in text:
        text = text.replace("inf", r"\infty")

    for spacing_command in (r"\,", r"\;", r"\:", r"\!"):
        text = text.replace(spacing_command, "")

    text = _fix_sqrt(text)
    text = _fix_tan(text)
    text = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", text)
    text = re.sub(r"(\d+)\.0+$", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = _fix_fracs(text)
    text = _fix_a_slash_b(text)
    return re.sub(r"(\\|,|\.)+$", "", text)


def _fix_fracs(text: str) -> str:
    parts = text.split(r"\frac")
    rebuilt = parts[0]
    if len(parts) == 1:
        return text

    for part in parts[1:]:
        rebuilt += r"\frac"
        if part.startswith("{"):
            rebuilt += part
            continue

        if len(part) < 2:
            return text

        numerator = part[0]
        denominator = part[1]
        if denominator != "{":
            rebuilt += "{" + numerator + "}{" + denominator + "}" + part[2:]
        else:
            rebuilt += "{" + numerator + "}" + denominator + part[2:]

    return rebuilt


def _fix_a_slash_b(text: str) -> str:
    if text.count("/") != 1:
        return text

    numerator, denominator = text.split("/")
    try:
        if "sqrt" not in numerator:
            numerator = str(int(numerator))
        if "sqrt" not in denominator:
            denominator = str(int(denominator))
    except ValueError:
        return text

    if text != f"{numerator}/{denominator}":
        return text
    return rf"\frac{{{numerator}}}{{{denominator}}}"


def _fix_sqrt(text: str) -> str:
    text = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", text)
    return re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", text)


def _fix_tan(text: str) -> str:
    text = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", text)
    return re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", text)


def _parse_digits(text: str) -> float | None:
    normalized = re.sub(r",", "", str(text).strip())
    try:
        return float(normalized)
    except ValueError:
        if normalized.endswith("%"):
            normalized = normalized[:-1].rstrip("\\")
            try:
                return float(normalized) / 100
            except ValueError:
                return None
    return None


def _parse_symbolic_expression(text: str):
    for parser in (parse_latex, parse_expr):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def _symbolic_equal(left: str, right: str) -> bool:
    left_expr = _parse_symbolic_expression(left)
    right_expr = _parse_symbolic_expression(right)
    if left_expr is None or right_expr is None:
        return False

    try:
        if simplify(left_expr - right_expr) == 0:
            return True
    except Exception:
        pass

    try:
        return isclose(float(N(left_expr)), float(N(right_expr)), abs_tol=1e-3)
    except Exception:
        return False


def _split_top_level(text: str, separator: str) -> list[str]:
    if len(separator) != 1:
        raise ValueError("separator must be a single character")

    parts = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    for index, char in enumerate(text):
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif (
            char == separator
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
        ):
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _split_top_level_equality(normalized_text: str) -> tuple[str, str] | None:
    parts = _split_top_level(normalized_text, "=")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _has_balanced_outer_delimiters(text: str, opener: str, closer: str) -> bool:
    if not text.startswith(opener) or not text.endswith(closer):
        return False

    depth = 0
    for index, char in enumerate(text):
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0


def _extract_sequence_parts(normalized_text: str) -> list[str] | None:
    for opener, closer in (("(", ")"), ("[", "]")):
        if _has_balanced_outer_delimiters(normalized_text, opener, closer):
            inner = normalized_text[1:-1].strip()
            if not inner:
                return []
            return _split_top_level(inner, ",")
    return None


def _extract_matrix_rows(normalized_text: str) -> list[list[str]] | None:
    for env_name in ("pmatrix", "bmatrix"):
        prefix = rf"\begin{{{env_name}}}"
        suffix = rf"\end{{{env_name}}}"
        if normalized_text.startswith(prefix) and normalized_text.endswith(suffix):
            inner = normalized_text[len(prefix) : len(normalized_text) - len(suffix)]
            rows = [row.strip() for row in inner.split(r"\\") if row.strip()]
            return [[entry.strip() for entry in row.split("&")] for row in rows]
    return None


def _looks_like_named_lhs(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    return IDENTIFIER_WITH_OPTIONAL_ARGS_PATTERN.fullmatch(normalized_text) is not None


def _math_answers_equal_normalized(
    normalized_predicted: str, normalized_gold: str
) -> bool:
    if normalized_predicted == normalized_gold:
        return True

    predicted_number = _parse_digits(normalized_predicted)
    gold_number = _parse_digits(normalized_gold)
    if predicted_number is not None and gold_number is not None:
        if isclose(predicted_number, gold_number, abs_tol=1e-3):
            return True

        gold_candidates = (gold_number / 100, gold_number, gold_number * 100)
        return any(
            isclose(predicted_number, candidate, abs_tol=1e-3)
            for candidate in gold_candidates
        )

    predicted_parts = _extract_sequence_parts(normalized_predicted)
    gold_parts = _extract_sequence_parts(normalized_gold)
    if predicted_parts is not None and gold_parts is not None:
        if len(predicted_parts) != len(gold_parts):
            return False
        return all(
            _math_answers_equal_normalized(predicted_part, gold_part)
            for predicted_part, gold_part in zip(
                predicted_parts, gold_parts, strict=True
            )
        )

    predicted_matrix = _extract_matrix_rows(normalized_predicted)
    gold_matrix = _extract_matrix_rows(normalized_gold)
    if predicted_matrix is not None and gold_matrix is not None:
        if len(predicted_matrix) != len(gold_matrix):
            return False
        return all(
            len(predicted_row) == len(gold_row)
            and all(
                _math_answers_equal_normalized(predicted_entry, gold_entry)
                for predicted_entry, gold_entry in zip(
                    predicted_row, gold_row, strict=True
                )
            )
            for predicted_row, gold_row in zip(
                predicted_matrix, gold_matrix, strict=True
            )
        )

    predicted_equation = _split_top_level_equality(normalized_predicted)
    gold_equation = _split_top_level_equality(normalized_gold)
    if predicted_equation and gold_equation:
        predicted_lhs, predicted_rhs = predicted_equation
        gold_lhs, gold_rhs = gold_equation
        predicted_difference = f"{predicted_lhs} - ({predicted_rhs})"
        gold_difference = f"{gold_lhs} - ({gold_rhs})"
        return _symbolic_equal(
            predicted_difference, gold_difference
        ) or _symbolic_equal(f"-({predicted_difference})", gold_difference)

    if predicted_equation and not gold_equation:
        predicted_lhs, predicted_rhs = predicted_equation
        if _looks_like_named_lhs(predicted_lhs):
            return _math_answers_equal_normalized(predicted_rhs, normalized_gold)

    if gold_equation and not predicted_equation:
        gold_lhs, gold_rhs = gold_equation
        if _looks_like_named_lhs(gold_lhs):
            return _math_answers_equal_normalized(normalized_predicted, gold_rhs)

    return _symbolic_equal(normalized_predicted, normalized_gold)


def math_answers_equal(predicted: str, gold: str) -> bool:
    normalized_predicted = normalize_math_answer(predicted)
    normalized_gold = normalize_math_answer(gold)
    return _math_answers_equal_normalized(normalized_predicted, normalized_gold)


def extract_scored_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    closing_index = text.rfind("</think>")
    if closing_index >= 0:
        return text[closing_index + len("</think>") :].strip()
    return ""


def has_valid_think_format(
    prompt_text: str, completion_text: str | None = None
) -> bool:
    prompt_text = prompt_text or ""
    if completion_text is None:
        open_index = prompt_text.find("<think>")
        close_index = prompt_text.find("</think>")
        return open_index >= 0 and close_index > open_index

    completion_text = (completion_text or "").strip()
    open_index = prompt_text.rfind("<think>")
    close_index = prompt_text.rfind("</think>")
    if open_index < 0 or close_index > open_index:
        return False

    if completion_text.count("<think>") != 0:
        return False
    if completion_text.count("</think>") != 1:
        return False

    close_index = completion_text.find("</think>")
    reasoning_text = completion_text[:close_index].strip()
    answer_text = completion_text[close_index + len("</think>") :].strip()
    if not reasoning_text or not answer_text:
        return False

    return extract_last_boxed_answer(answer_text) != ""


def extract_last_boxed_answer(text: str) -> str:
    text = text or ""
    last_boxed_answer = ""
    search_start = 0

    while True:
        boxed_start = text.find(BOXED_PREFIX, search_start)
        if boxed_start < 0:
            return last_boxed_answer

        content_start = boxed_start + len(BOXED_PREFIX)
        depth = 1
        index = content_start
        while index < len(text) and depth > 0:
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            index += 1

        if depth == 0:
            last_boxed_answer = text[content_start : index - 1].strip()
            search_start = index
            continue

        return last_boxed_answer


def extract_final_answer(text: str) -> str:
    text = extract_scored_text(text)
    if not text:
        return ""

    return extract_last_boxed_answer(text)


def parse_correctness_label(text: str) -> float:
    match = CORRECTNESS_PATTERN.search(text or "")
    if match:
        return 1.0 if match.group(1).upper() == "YES" else 0.0
    return 0.0


def parse_confidence_label(text: str) -> float:
    match = CONFIDENCE_PATTERN.search(text or "")
    if match:
        return 1.0 if match.group(1).upper() == "HIGH" else 0.0
    return 0.0
