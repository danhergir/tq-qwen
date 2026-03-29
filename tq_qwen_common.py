from __future__ import annotations

import json
import re
from typing import Any

SENTINEL_PATTERN = re.compile(r"<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>")


def clean_response(text: str) -> str:
    return SENTINEL_PATTERN.split(text, maxsplit=1)[0].strip()


def encode_ndjson(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
