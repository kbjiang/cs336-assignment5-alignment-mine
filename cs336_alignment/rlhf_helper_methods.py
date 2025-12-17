from typing import Any
import re

def parse_mmlu_response(
    response: str,
    mmlu_example: dict[str, Any] | None = None,
):
    chars = re.findall(r'the correct answer is (.)', response.lower())
    for char in chars[::-1]:
        if char in ["a", "b", "c", "d"]:
            return char.upper()
    return None