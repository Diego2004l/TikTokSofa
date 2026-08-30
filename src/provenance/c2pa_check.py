"""Tier 0 — C2PA provenance check (spec section 4, Tier 0).

Wraps the `c2patool` CLI (Content Authenticity Initiative, Apache/MIT). If a valid, trusted C2PA
manifest is present the caller can early-exit the cascade instead of running Tiers 1-3.

Install: https://github.com/contentauth/c2pa-rs/releases (a Rust binary, not pip-installable) —
put it on PATH. If it's missing, `check_c2pa` degrades to `detected=False` with a note rather
than raising, since Tier 0 being unavailable should never block the rest of the pipeline.
"""

from __future__ import annotations

import json
import shutil
import subprocess


def c2patool_available() -> bool:
    return shutil.which("c2patool") is not None


def check_c2pa(image_path: str, timeout: float = 10.0) -> dict:
    """Returns {"detected": bool, "issuer": str|None, "validation_state": str|None, "note": str|None}."""
    if not c2patool_available():
        return {"detected": False, "issuer": None, "validation_state": None, "note": "c2patool not installed"}

    try:
        result = subprocess.run(
            ["c2patool", image_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"detected": False, "issuer": None, "validation_state": None, "note": "c2patool timed out"}

    if result.returncode != 0 or not result.stdout.strip():
        # No manifest is the common case for an image with no embedded Content Credentials.
        return {"detected": False, "issuer": None, "validation_state": None, "note": result.stderr.strip() or None}

    try:
        manifest_store = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"detected": False, "issuer": None, "validation_state": None, "note": "unparsable c2patool output"}

    active_id = manifest_store.get("active_manifest")
    manifests = manifest_store.get("manifests", {})
    active = manifests.get(active_id, {}) if active_id else {}

    issuer = None
    claim_generator = active.get("claim_generator_info")
    if isinstance(claim_generator, list) and claim_generator:
        issuer = claim_generator[0].get("name")
    issuer = issuer or active.get("claim_generator")

    validation_state = manifest_store.get("validation_state") or active.get("validation_state")

    return {
        "detected": bool(active_id),
        "issuer": issuer,
        "validation_state": validation_state,
        "note": None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check an image for a C2PA manifest.")
    parser.add_argument("image_path")
    args = parser.parse_args()
    print(json.dumps(check_c2pa(args.image_path), indent=2))
