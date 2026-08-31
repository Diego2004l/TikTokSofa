"""The canonical list of benchmark conditions (Feature 1).

`build_conditions()` returns an ordered list of `Condition` objects. Each carries the metadata
the benchmark writes into every result row (`family`, `severity`, `severity_rank`,
`compound_id`, `seed`) and the callable transform itself. The robustness benchmark, the
degradation-curve plotter, and the profiler's synthetic-data generator all consume this one
list so they can never drift apart.

`severity` is a monotonic "how degraded" scalar within a family (higher = more degradation),
chosen so degradation curves are plottable; it is NOT comparable across families. `severity_rank`
is the integer position on that family's severity ladder (0 = clean-ish).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Iterator

from PIL import Image

from src.transforms import catalog
from src.transforms.chains import NAMED_CHAINS, random_compound_chain
from src.transforms.screenshot import screenshot_sim

Transform = Callable[[Image.Image], Image.Image]

DEFAULT_SEED = 0
N_RANDOM_COMPOUND = 8  # deterministic random compound chains, seeds DEFAULT_SEED..+N-1


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    transform: Transform
    severity: float = 0.0
    severity_rank: int = 0
    compound_id: str | None = None
    seed: int | None = None
    optional: bool = False  # motion blur / salt-and-pepper — off unless include_optional
    params: dict = field(default_factory=dict)

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.transform(img)


def _isolated(seed: int) -> list[Condition]:
    C = Condition
    conds: list[Condition] = [C("clean", "clean", catalog.identity, 0.0, 0)]

    # ---- Compression -----------------------------------------------------
    for rank, q in enumerate([95, 85, 70, 50, 30], start=1):
        conds.append(C(f"jpeg_q{q}", "compression", partial(catalog.jpeg_compress, quality=q),
                       severity=(100 - q) / 100, severity_rank=rank, params={"quality": q}))

    # ---- Resolution ----------------------------------------------------
    for rank, s in enumerate([0.9, 0.75, 0.5, 0.25], start=1):
        conds.append(C(f"resize_{int(s * 100)}pct", "resolution", partial(catalog.resize_scale, scale=s),
                       severity=1 - s, severity_rank=rank, params={"scale": s}))
    conds.append(C("downsample_upsample_50", "resolution", partial(catalog.resize_down_up, scale=0.5),
                   severity=0.55, severity_rank=5, params={"scale": 0.5}))

    # ---- Blur ----------------------------------------------------------
    for rank, sig in enumerate([0.5, 1.0, 2.0, 3.0], start=1):
        conds.append(C(f"gauss_blur_s{sig}", "blur", partial(catalog.gaussian_blur, sigma=sig),
                       severity=sig / 3.0, severity_rank=rank, params={"sigma": sig}))
    conds.append(C("motion_blur_k9", "blur", partial(catalog.motion_blur, kernel_size=9, angle_deg=30.0),
                   severity=0.6, severity_rank=5, optional=True, params={"kernel_size": 9, "angle_deg": 30.0}))

    # ---- Noise -------------------------------------------------------
    for rank, sig in enumerate([0.02, 0.05, 0.1], start=1):
        conds.append(C(f"gauss_noise_s{sig}", "noise",
                       partial(catalog.gaussian_noise, sigma=sig, seed=seed),
                       severity=sig / 0.1, severity_rank=rank, seed=seed, params={"sigma": sig}))
    conds.append(C("salt_pepper_0.02", "noise", partial(catalog.salt_pepper_noise, amount=0.02, seed=seed),
                   severity=0.5, severity_rank=4, optional=True, seed=seed, params={"amount": 0.02}))

    # ---- Sharpening --------------------------------------------------
    for rank, pct in enumerate([150, 300], start=1):
        conds.append(C(f"sharpen_p{pct}", "sharpen", partial(catalog.unsharp_mask, percent=pct),
                       severity=pct / 300, severity_rank=rank, params={"percent": pct}))

    # ---- Cropping --------------------------------------------------
    for rank, frac in enumerate([0.9, 0.75, 0.5], start=1):
        conds.append(C(f"center_crop_{int(frac * 100)}", "crop", partial(catalog.center_crop, frac=frac),
                       severity=1 - frac, severity_rank=rank, params={"frac": frac}))
        conds.append(C(f"random_crop_{int(frac * 100)}", "crop",
                       partial(catalog.random_crop, frac=frac, seed=seed),
                       severity=1 - frac, severity_rank=rank, seed=seed, params={"frac": frac}))

    # ---- Colour --------------------------------------------------
    for rank, (nm, fn, factor) in enumerate([
        ("brightness_up", catalog.adjust_brightness, 1.2),
        ("brightness_down", catalog.adjust_brightness, 0.8),
        ("contrast_up", catalog.adjust_contrast, 1.3),
        ("contrast_down", catalog.adjust_contrast, 0.7),
        ("saturation_up", catalog.adjust_saturation, 1.4),
        ("saturation_down", catalog.adjust_saturation, 0.6),
    ], start=1):
        conds.append(C(f"color_{nm}", "color", partial(fn, factor=factor),
                       severity=abs(1 - factor), severity_rank=rank, params={"factor": factor}))
    for rank, deg in enumerate([7, 15], start=1):
        conds.append(C(f"hue_shift_{deg}", "color", partial(catalog.adjust_hue, shift_deg=deg),
                       severity=deg / 15, severity_rank=rank, params={"shift_deg": deg}))

    # ---- Screenshot simulation ------------------------------------
    conds.append(C("screenshot_light", "screenshot",
                   partial(screenshot_sim, scale=0.9, border=0, jpeg_quality=90),
                   severity=0.3, severity_rank=1, params={"scale": 0.9, "jpeg_quality": 90}))
    conds.append(C("screenshot_typical", "screenshot",
                   partial(screenshot_sim, scale=0.85, border=10, jpeg_quality=80),
                   severity=0.6, severity_rank=2, params={"scale": 0.85, "border": 10, "jpeg_quality": 80}))
    conds.append(C("screenshot_heavy", "screenshot",
                   partial(screenshot_sim, scale=0.7, border=16, jpeg_quality=60),
                   severity=0.9, severity_rank=3, params={"scale": 0.7, "border": 16, "jpeg_quality": 60}))

    return conds


def _compound(seed: int) -> list[Condition]:
    conds: list[Condition] = []
    for rank, (name, ch) in enumerate(NAMED_CHAINS.items(), start=1):
        conds.append(Condition(name, "compound", ch, severity=0.5 + 0.05 * rank,
                               severity_rank=rank, compound_id=name))
    for i in range(N_RANDOM_COMPOUND):
        s = seed + i
        n_ops = 2 + (i % 3)  # 2, 3, 4, 2, 3, 4, ...
        ch = random_compound_chain(seed=s, n_ops=n_ops)
        cid = f"randcompound_s{s}_n{n_ops}"
        conds.append(Condition(cid, "random_compound", ch, severity=0.5 + 0.05 * n_ops,
                               severity_rank=n_ops, compound_id=cid, seed=s,
                               params={"ops": list(ch.labels)}))
    return conds


def build_conditions(seed: int = DEFAULT_SEED, include_optional: bool = True,
                     include_compound: bool = True) -> list[Condition]:
    conds = _isolated(seed)
    if not include_optional:
        conds = [c for c in conds if not c.optional]
    if include_compound:
        conds += _compound(seed)
    return conds


def iter_conditions(**kwargs) -> Iterator[Condition]:
    yield from build_conditions(**kwargs)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="List every benchmark condition.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-optional", action="store_true")
    args = ap.parse_args()
    rows = build_conditions(seed=args.seed, include_optional=not args.no_optional)
    print(f"{len(rows)} conditions:\n")
    for c in rows:
        extra = f"  [{', '.join(c.params.get('ops', []))}]" if c.params.get("ops") else ""
        print(f"  {c.family:16s} {c.name:28s} sev={c.severity:.2f} rank={c.severity_rank}{extra}")
