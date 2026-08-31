"""Deterministic "social-media screenshot" simulation (Feature 1).

This does NOT claim to reproduce a real screenshot taken on a phone and re-uploaded through a
platform's pipeline. It is a *simulation* of the transformations such a screenshot typically
stacks: viewport rescaling, forced RGB, an optional UI chrome border, JPEG recompression, and
an optional sub-pixel resample. Documented as a simulation everywhere it is reported.
"""

from __future__ import annotations

import io

from PIL import Image


def screenshot_sim(
    img: Image.Image,
    scale: float = 0.85,
    border: int = 0,
    border_color: tuple[int, int, int] = (245, 245, 245),
    jpeg_quality: int = 80,
    resample_jitter: bool = True,
) -> Image.Image:
    """Apply the screenshot-like transform chain deterministically.

    scale           -- viewport rescale factor (screenshot is rarely 1:1 with the source)
    border          -- px of solid UI-chrome margin added on every side (0 disables)
    jpeg_quality    -- recompression quality the platform re-encodes the screenshot at
    resample_jitter -- one extra +1px/-1px bicubic resample round-trip, approximating the
                       non-integer scaling a device viewport applies
    """
    out = img.convert("RGB")
    w, h = out.size

    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    out = out.resize((sw, sh), Image.BICUBIC)

    if resample_jitter:
        out = out.resize((sw + 1, sh + 1), Image.BICUBIC).resize((sw, sh), Image.BICUBIC)

    if border > 0:
        framed = Image.new("RGB", (sw + 2 * border, sh + 2 * border), border_color)
        framed.paste(out, (border, border))
        out = framed

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
