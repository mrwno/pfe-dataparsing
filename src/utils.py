"""Shared utilities for dataset standardization and evaluation."""
import re
import json


def extract_json(text: str) -> dict | None:
    """
    Extract the first valid JSON object found in a raw model response.

    Handles reasoning-model <think>...</think> blocks, markdown fences,
    and uses brace-depth tracking so nested objects don't confuse the parser.

    Args:
        text: Raw string output from the model.

    Returns:
        Parsed dict, or None if no valid JSON object is found.
    """
    def _find_json(s: str) -> dict | None:
        """Extract the first valid JSON object from string s."""
        fenced = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", s, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass
        start = s.find("{")
        if start == -1:
            return None
        depth, in_string, escape_next = 0, False, False
        for i, ch in enumerate(s[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start: i + 1])
                    except json.JSONDecodeError:
                        break
        return None

    # Repair common malformed key pattern produced by small models:
    # "key"_suffix": → "key_suffix":
    text = re.sub(r'"(\w+)"_(\w+)"\s*:', r'"\1_\2":', text)

    # 1. Try full text first (JSON may appear before any think block).
    result = _find_json(text)
    if result:
        return result

    # 2. Extract think-block content and search inside it.
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        result = _find_json(think_match.group(1))
        if result:
            return result

    # 3. Strip think blocks and try what remains.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    return _find_json(cleaned)


def generate_code(mapping: dict) -> str:
    """
    Generate Unitxt preprocess_steps code string from a field mapping.

    Args:
        mapping: Dict mapping standard field names to source column names.

    Returns:
        Comma-separated string of Unitxt Rename steps, or a comment if empty.
    """
    if not mapping:
        return "# Mapping failed"
    steps = [
        f"Rename(field='{src}', to_field='{field}')"
        for field, src in mapping.items()
        if field != "task" and isinstance(src, str) and src != field
    ]
    return ", ".join(steps) if steps else "# No rename steps needed"


def score_mapping(dataset, mapping: dict, n: int = 10) -> float:
    """
    Score mapping validity by checking that referenced columns exist in N rows.

    Args:
        dataset: Dataset object (must support .take() or slicing).
        mapping: Dict mapping standard field names to source column names.
        n: Number of samples to validate against.

    Returns:
        Fraction of rows where all referenced columns are present (0.0–1.0).
    """
    if not mapping:
        return 0.0
    try:
        samples = list(dataset.take(n)) if hasattr(dataset, "take") else dataset[:n]
        if isinstance(samples, dict):
            samples = [dict(zip(samples.keys(), vals)) for vals in zip(*samples.values())]
        if not samples:
            return 0.0
        # Only count values that actually refer to columns (not literal metadata).
        col_refs = [
            v for k, v in mapping.items()
            if k != "task" and isinstance(v, str) and v in samples[0]
        ]
        return sum(all(col in row for col in col_refs) for row in samples) / n
    except Exception:
        return 0.0
