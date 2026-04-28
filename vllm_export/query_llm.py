#!/usr/bin/env python3
"""Simple OpenAI-compatible query script for vLLM."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


def call_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    json_mode: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a concise helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/v1/chat/completions",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--json", action="store_true", help="Request JSON output mode.")
    args = ap.parse_args()

    out = call_chat(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt=args.prompt,
        temperature=args.temperature,
        timeout=args.timeout,
        json_mode=args.json,
    )
    msg = (
        ((out.get("choices") or [{}])[0].get("message") or {}).get("content")
        if isinstance(out, dict)
        else None
    )
    print("[response]")
    print(msg if isinstance(msg, str) else json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
