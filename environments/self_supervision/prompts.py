from __future__ import annotations


def build_self_eval_messages(
    prompt_messages: list[dict[str, str]], answer_text: str
) -> list[dict[str, str]]:
    verifier_prompt = (
        "Evaluate your previous answer without solving the problem again.\n"
        "Reply with exactly two lines and nothing else:\n"
        "CORRECTNESS: YES or CORRECTNESS: NO\n"
        "CONFIDENCE: HIGH or CONFIDENCE: LOW\n\n"
        "Use only the answer already in the conversation above. /no_think"
    )
    return [
        *prompt_messages,
        {"role": "assistant", "content": answer_text},
        {"role": "user", "content": verifier_prompt},
    ]
