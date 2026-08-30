"""Tier 0 — optional OpenAI-assisted provenance check, behind `--use-provenance-api`.

Honesty note: OpenAI does not publish a dedicated "verify this image's provenance" endpoint.
This module is a best-effort supplementary signal only — it asks a vision-capable chat model to
report any visible AI-generation indicators (visible watermarks, generator-specific artifacts it
recognizes) via the standard `openai` chat completions API. Treat its output as a weak hint, not
a verified credential the way a valid C2PA manifest (`c2pa_check.py`) is. It is OFF by default,
never runs without `OPENAI_API_KEY` set, and never blocks the pipeline if unset or if the call
fails.
"""

from __future__ import annotations

import base64
import os

from dotenv import load_dotenv

load_dotenv()


def openai_check_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def check_via_openai(image_path: str, model: str = "gpt-4o-mini") -> dict:
    """Returns {"detected": bool, "note": str} — `detected` is a heuristic opinion, not a
    cryptographic guarantee. Returns detected=False, note="unavailable" if no API key is set
    or the call errors, so callers can safely call this unconditionally behind the CLI flag."""
    if not openai_check_available():
        return {"detected": False, "note": "OPENAI_API_KEY not set"}

    try:
        from openai import OpenAI

        client = OpenAI()
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Does this image contain any visible AI-generation watermark, "
                                "signature, or generator-specific artifact you recognize "
                                "(e.g. a DALL-E, Midjourney, or SynthID mark)? Answer with "
                                "exactly 'yes' or 'no' on the first line, then a one-sentence reason."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=60,
        )
        text = response.choices[0].message.content or ""
        first_line = text.strip().splitlines()[0].lower() if text.strip() else ""
        return {"detected": first_line.startswith("yes"), "note": text.strip()}
    except Exception as exc:  # noqa: BLE001 — this is an optional, best-effort signal
        return {"detected": False, "note": f"openai check failed: {exc}"}
