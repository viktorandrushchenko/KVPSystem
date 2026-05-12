from __future__ import annotations

import json
import re
from typing import Any


GROUNDING_RE = re.compile(
    r"<\|ref\|>(?P<ref>.*?)<\|/ref\|>\s*<\|det\|>(?P<bbox>.*?)<\|/det\|>(?P<value>.*)",
    flags=re.DOTALL,
)


def parse_grounded_value(text: str, query_key: str) -> dict[str, Any]:
    """Parse DeepSeek-OCR grounded answer: <|ref|>...<|/ref|><|det|>bbox<|/det|>value."""
    raw = text or ""
    matches = list(GROUNDING_RE.finditer(raw))
    if not matches:
        return {
            "key": query_key,
            "value": _clean_value(raw),
            "bbox": None,
            "ref": None,
            "confidence": None,
        }

    match = matches[-1]
    bbox = _parse_bbox(match.group("bbox"))
    return {
        "key": query_key,
        "value": _clean_value(match.group("value")),
        "bbox": bbox,
        "ref": match.group("ref").strip(),
        "confidence": None,
    }


def parse_key_values(text: str) -> list[dict[str, Any]]:
    """Parse model output into a normalized list of key-value dicts."""
    if not text:
        return []

    parsed = _try_json(text)
    if parsed is not None:
        return _normalize(parsed)

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        parsed = _try_json(fenced.group(1))
        if parsed is not None:
            return _normalize(parsed)

    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        parsed = _try_json(array_match.group(0))
        if parsed is not None:
            return _normalize(parsed)

    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        parsed = _try_json(object_match.group(0))
        if parsed is not None:
            return _normalize(parsed)

    pairs: list[dict[str, Any]] = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip(" -*\t")
        value = value.strip()
        if key and value:
            pairs.append({"key": key, "value": value, "confidence": None})
    return pairs


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _normalize(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if "key" in value and "value" in value:
            return [_pair(value)]
        return [_pair({"key": key, "value": item}) for key, item in value.items()]

    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                if "key" in item and "value" in item:
                    items.append(_pair(item))
                elif len(item) == 1:
                    key, val = next(iter(item.items()))
                    items.append(_pair({"key": key, "value": val}))
                else:
                    for key, val in item.items():
                        if key not in {"confidence", "score"}:
                            items.append(_pair({"key": key, "value": val, "confidence": item.get("confidence")}))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                items.append(_pair({"key": item[0], "value": item[1]}))
        return items

    return []


def _pair(item: dict[str, Any]) -> dict[str, Any]:
    confidence = item.get("confidence", item.get("score"))
    return {
        "key": "" if item.get("key") is None else str(item.get("key")).strip(),
        "value": "" if item.get("value") is None else str(item.get("value")).strip(),
        "bbox": item.get("bbox"),
        "confidence": confidence,
    }


def _parse_bbox(text: str) -> list[list[int]] | None:
    parsed = _try_json(text)
    if parsed is None:
        return None
    if isinstance(parsed, list) and parsed and all(isinstance(x, (int, float)) for x in parsed):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None
    boxes: list[list[int]] = []
    for item in parsed:
        if isinstance(item, list) and len(item) == 4 and all(isinstance(x, (int, float)) for x in item):
            boxes.append([int(round(x)) for x in item])
    return boxes or None


def _clean_value(value: str) -> str:
    value = value.strip()
    value = re.sub(r"<\|.*?\|>", "", value)
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[0] if lines else ""
