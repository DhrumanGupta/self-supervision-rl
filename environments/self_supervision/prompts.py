from __future__ import annotations


def build_self_eval_messages(
    prompt_messages: list[dict[str, str]], answer_text: str
) -> list[dict[str, str]]:
    original_prompt = prompt_messages[-1]["content"] if prompt_messages else ""
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating your own earlier answer. Do not solve the problem again. "
                "Judge only the answer you already gave."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{original_prompt}\n\n"
                f"Your earlier answer:\n{answer_text}\n\n"
                "Return exactly two lines:\n"
                "CORRECTNESS: YES or CORRECTNESS: NO\n"
                "CONFIDENCE: HIGH or CONFIDENCE: LOW"
            ),
        },
    ]
