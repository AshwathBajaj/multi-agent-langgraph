"""Small helpers for the JSON contracts each agent asks the model to return."""

from __future__ import annotations

import json
import re


def system_prompt(role: str, body: str) -> str:
    return f"Role: {role}\n\n{body.strip()}\n"


def dump_payload(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str)


def parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            raise ValueError("Model response did not contain a JSON object.") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Model response JSON was not an object.")
    return value


def estimate_tokens(text: str) -> int:
    """Rough token count used only when a provider does not return usage."""

    return max(1, (len(text) + 3) // 4)
