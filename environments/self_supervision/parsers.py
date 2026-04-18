from __future__ import annotations

import re

from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig


CORRECTNESS_PATTERN = re.compile(r"CORRECTNESS\s*:\s*(YES|NO)", flags=re.IGNORECASE)
CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE\s*:\s*(HIGH|LOW)", flags=re.IGNORECASE)
BOXED_PREFIX = r"\boxed{"
LITERAL_ANSWERS = {"yes", "no", "true", "false"}
WRAPPED_LATEX_EXTRACTION_CONFIG = [LatexExtractionConfig(boxed_match_priority=0)]
REASONING_TOKEN_PATTERN = re.compile(r"(?:\\[A-Za-z]+|[A-Za-z0-9]+)")
MIN_REASONING_NONSPACE_CHARS = 12
MIN_REASONING_TOKENS = 3
LATEX_TEXT_WRAPPER_PATTERN = re.compile(
    r"^\\(?:text|mathrm)\s*\{\s*(.*?)\s*\}$",
    flags=re.IGNORECASE,
)


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
    if not _has_substantive_reasoning(reasoning_text) or not answer_text:
        return False

    return extract_last_boxed_answer(answer_text) != ""


def _has_substantive_reasoning(reasoning_text: str) -> bool:
    compact_reasoning = "".join((reasoning_text or "").split())
    if len(compact_reasoning) < MIN_REASONING_NONSPACE_CHARS:
        return False

    tokens = REASONING_TOKEN_PATTERN.findall(reasoning_text)
    if len(tokens) < MIN_REASONING_TOKENS:
        return False

    return any(any(char.isalpha() for char in token) for token in tokens)


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


def is_literal_answer(text: str) -> bool:
    return _canonicalize_boolean_literal(text) is not None


def _canonicalize_boolean_literal(text: str) -> str | None:
    normalized = (text or "").strip()
    if not normalized:
        return None

    wrapped_match = LATEX_TEXT_WRAPPER_PATTERN.fullmatch(normalized)
    if wrapped_match:
        normalized = wrapped_match.group(1).strip()

    lowered = normalized.lower()
    if lowered in {"yes", "true"}:
        return "yes"
    if lowered in {"no", "false"}:
        return "no"
    return None


def normalize_exact_answer(text: str) -> str:
    normalized = (text or "").strip()

    canonical_literal = _canonicalize_boolean_literal(normalized)
    if canonical_literal is not None:
        return canonical_literal

    while True:
        updated = normalized
        if updated.startswith(r"\[") and updated.endswith(r"\]"):
            updated = updated[2:-2].strip()
        elif updated.startswith(r"\(") and updated.endswith(r"\)"):
            updated = updated[2:-2].strip()
        elif updated.startswith("$") and updated.endswith("$") and len(updated) >= 2:
            updated = updated[1:-1].strip()

        if updated == normalized:
            break
        normalized = updated

    canonical_literal = _canonicalize_boolean_literal(normalized)
    if canonical_literal is not None:
        return canonical_literal

    normalized = normalized.replace(r"\dfrac", r"\frac")
    normalized = normalized.replace(r"\tfrac", r"\frac")
    normalized = normalized.replace(r"\left", "")
    normalized = normalized.replace(r"\right", "")
    normalized = "".join(normalized.split())

    canonical_literal = _canonicalize_boolean_literal(normalized)
    if canonical_literal is not None:
        return canonical_literal
    return normalized


def semantic_match_answers(gold_answer: str, predicted_answer: str) -> bool | None:
    gold_answer = (gold_answer or "").strip()
    predicted_answer = (predicted_answer or "").strip()
    if not gold_answer or not predicted_answer:
        return None
    if is_literal_answer(gold_answer) or is_literal_answer(predicted_answer):
        return None

    gold_parsed = parse(f"${gold_answer}$", extraction_mode="first_match")
    predicted_parsed = parse(
        f"${predicted_answer}$",
        extraction_config=WRAPPED_LATEX_EXTRACTION_CONFIG,
        extraction_mode="first_match",
    )
    if not gold_parsed or not predicted_parsed:
        return None

    try:
        return bool(verify(gold_parsed, predicted_parsed))
    except Exception:
        return None


def match_answers_hybrid(gold_answer: str, predicted_answer: str) -> tuple[float, bool]:
    semantic_match = semantic_match_answers(gold_answer, predicted_answer)
    if semantic_match is not None:
        return float(semantic_match), False

    exact_match = normalize_exact_answer(gold_answer) == normalize_exact_answer(
        predicted_answer
    )
    return float(exact_match), False


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
