from __future__ import annotations

import re


BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
HASH_PATTERN = re.compile(r"^####\s*(.+)$", flags=re.MULTILINE)
CORRECTNESS_PATTERN = re.compile(r"CORRECTNESS\s*:\s*(YES|NO)", flags=re.IGNORECASE)
CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE\s*:\s*(HIGH|LOW)", flags=re.IGNORECASE)


def extract_scored_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    closing_index = text.rfind("</think>")
    if closing_index >= 0:
        return text[closing_index + len("</think>") :].strip()
    return ""


def has_valid_think_format(text: str) -> bool:
    text = text or ""
    open_index = text.find("<think>")
    close_index = text.find("</think>")
    return open_index >= 0 and close_index > open_index


def extract_final_answer(text: str) -> str:
    text = extract_scored_text(text)
    if not text:
        return ""

    boxed_match = BOXED_PATTERN.search(text)
    if boxed_match:
        return boxed_match.group(1).strip()

    hash_match = HASH_PATTERN.search(text)
    if hash_match:
        return hash_match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].removeprefix("Final answer:").strip(" :-")


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
