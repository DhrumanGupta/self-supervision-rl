from __future__ import annotations


MAIN_RESPONSE_INSTRUCTION = (
    "Think and reason about the question inside <think> and </think> tags.\n"
    "After closing </think>, write the final answer only as:\n"
    "\\[\n"
    "\\boxed{answer here}\n"
    "\\]"
)


def build_main_prompt_messages(
    prompt_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    if prompt_messages and prompt_messages[0].get("role") == "system":
        merged_system_message = dict(prompt_messages[0])
        merged_system_message["content"] = (
            f"{MAIN_RESPONSE_INSTRUCTION}\n\n{prompt_messages[0].get('content', '').strip()}"
        ).strip()
        return [merged_system_message, *prompt_messages[1:]]

    return [
        {"role": "system", "content": MAIN_RESPONSE_INSTRUCTION},
        *prompt_messages,
    ]


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
