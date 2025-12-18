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

def parse_gsm8k_response(
    response: str,
):
    # 1. the `replace` takes care of strings like "1,600 grams"
    # 2. the `\.?` optional decimal point for strings like "$8.00"
    # 3. last `?` to avoid confusion with period.
    numbers = re.findall(r'\d+(?:\.\d+)?', response.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None