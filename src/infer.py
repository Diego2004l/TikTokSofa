"""End-to-end cascade inference: image directory -> JSON [{"image_path", "pred", ...}].

Cascade (spec sections 3, 4, 8):
  Tier 0 (C2PA) + Tier 2 (forensic features) run on EVERY image — both are cheap, CPU-only.
  Tier 1 (CNN) + Tier 3 (CLIP) only run when Tier 0/Tier 2 are inconclusive or disagree, mirroring
  a real platform's cost constraints (cheap gate on every upload, expensive tiers on escalation).
  Fusion combines whatever tiers ran into one calibrated score via degradation-aware reweighting.

This script is designed to degrade gracefully: if a tier's trained artifact (checkpoint /
classifier / probe / fusion model) isn't present yet, that tier is skipped with a note instead of
crashing, so `python -m src.infer` always produces valid output — see spec build-order item 3
("infer.py producing valid JSON end-to-end" as the safety net to get working first).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import joblib
import numpy as np
from PIL import Image

from src.frequency.features import extract_all_features
from src.fusion import build_features
from src.provenance.c2pa_check import check_c2pa
from src.provenance.openai_check import check_via_openai, openai_check_available
from src.router.features import Evidence

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# Known AI-generator claim_generator substrings for the Tier 0 heuristic. Not exhaustive —
# absence of a match is NOT evidence of a real image, only "provenance didn't confidently say AI".
KNOWN_AI_ISSUERS = ("dall", "midjourney", "firefly", "bing image creator", "stable diffusion", "imagen", "synthid")


def list_images(folder: str) -> list[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(paths)


def tier0_score(image_path: str, use_provenance_api: bool) -> tuple[float | None, bool, dict]:
    """Returns (score_or_None, early_exit_as_aigc, raw_result). score is 1.0/0.0 only when
    confident; None means "no usable signal", so it's excluded from fusion rather than
    forced to a misleading 0.5."""
    c2pa = check_c2pa(image_path)
    if c2pa["detected"] and c2pa.get("validation_state") in ("Valid", "Trusted", None):
        issuer = (c2pa.get("issuer") or "").lower()
        if any(name in issuer for name in KNOWN_AI_ISSUERS):
            return 1.0, True, {"c2pa": c2pa}

    if use_provenance_api and openai_check_available():
        oai = check_via_openai(image_path)
        if oai["detected"]:
            return 1.0, False, {"c2pa": c2pa, "openai": oai}  # weak signal — contributes but doesn't early-exit
        return None, False, {"c2pa": c2pa, "openai": oai}

    return None, False, {"c2pa": c2pa}


def load_optional(path: str | None):
    if path and os.path.exists(path):
        return joblib.load(path)
    return None


def _multi_crop(img, args, tier1_model, tier2_clf, tier3_embedder, tier3_probe,
                fusion_model, router, profiler, n):
    """Feature 4: re-score deterministic crops and measure final-score stability."""
    from src.confidence.multicrop import multi_crop_eval

    def score_fn(crops):
        out = []
        for c in crops:
            f = extract_all_features(c, max_dim=args.max_image_dim)
            t2 = float(tier2_clf.predict_proba(f["vector"].reshape(1, -1))[0, 1]) if tier2_clf is not None else 0.5
            deg = f["degradation_score"]
            t1 = t3 = None
            if tier1_model is not None:
                from src.model.evaluate import score_images_from_pil

                t1 = score_images_from_pil(tier1_model, [c])[0]
            if tier3_embedder is not None and tier3_probe is not None:
                emb = tier3_embedder.embed_image(c).numpy().reshape(1, -1)
                t3 = float(tier3_probe.predict_proba(emb)[0, 1])
            prof = profiler.predict(c, forensic=f) if profiler is not None else None
            if router is not None:
                out.append(router.predict_one(Evidence(tier1=t1, tier2=t2, tier3=t3, degradation=deg,
                                                       profile=prof, min_side=min(c.size))))
            elif fusion_model is not None:
                out.append(float(fusion_model.predict_proba(build_features(
                    0.0, t1 if t1 is not None else 0.5, t2, t3 if t3 is not None else 0.5, deg
                ).reshape(1, -1))[0, 1]))
            else:
                vals = [v for v in (t1, t2, t3) if v is not None]
                out.append(float(np.mean(vals)) if vals else 0.5)
        return out

    return multi_crop_eval(score_fn, img, n=n)


def main():
    parser = argparse.ArgumentParser(description="Run the full AIGC-detection cascade over a directory of images.")
    parser.add_argument("image_dir")
    parser.add_argument("--out", default="outputs/predictions.json")
    parser.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    parser.add_argument("--tier1-checkpoint", default="outputs/tier1_efficientnet_b0.pt")
    parser.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    parser.add_argument("--fusion-model", default="outputs/fusion_model.joblib")
    parser.add_argument("--transformation-profiler", default="outputs/transformation_profiler.joblib",
                        help="Feature 2 profiler. If present, each result gets a 'transformation_profile' block.")
    parser.add_argument("--router-model", default="outputs/router_model.joblib",
                        help="Feature 3 adaptive router. If present, it is the primary combiner (fusion is the fallback).")
    parser.add_argument("--confidence-model", default="outputs/confidence_model.joblib",
                        help="Feature 4. With --abstention-policy, adds confidence + AI/REAL/UNKNOWN outcome.")
    parser.add_argument("--abstention-policy", default="outputs/abstention_policy.joblib")
    parser.add_argument("--multicrop-n", type=int, default=5,
                        help="Deterministic crops for hard cases (Feature 4). 0 disables.")
    parser.add_argument("--use-provenance-api", action="store_true", help="Also call the optional OpenAI-assisted check (needs OPENAI_API_KEY).")
    parser.add_argument("--max-image-dim", type=int, default=None, help="Must match the value used for src/frequency/train_svm.py (Tier 2). Leave unset unless Tier 2 was trained with it.")
    parser.add_argument("--escalation-margin", type=float, default=0.15, help="Escalate to Tier 1/3 if Tier 2's score is within this margin of 0.5.")
    parser.add_argument("--always-escalate", action="store_true", help="Force Tier 1 + Tier 3 to run on every image (used by the robustness eval).")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    tier2_clf = load_optional(args.tier2_classifier)
    fusion_model = load_optional(args.fusion_model)

    tier1_model = None
    if os.path.exists(args.tier1_checkpoint):
        from src.model.model import load_checkpoint

        tier1_model = load_checkpoint(args.tier1_checkpoint)

    tier3_embedder = tier3_probe = None
    tier3_probe = load_optional(args.tier3_probe)
    if tier3_probe is not None:
        from src.semantic.embed import ClipEmbedder

        tier3_embedder = ClipEmbedder()

    profiler = None
    if os.path.exists(args.transformation_profiler):
        from src.transformation.model import load_profiler

        profiler = load_profiler(args.transformation_profiler)

    router = None
    if os.path.exists(args.router_model):
        from src.router.model import load_router

        router = load_router(args.router_model)

    confidence_model = abstention_policy = None
    if os.path.exists(args.confidence_model) and os.path.exists(args.abstention_policy):
        import joblib as _joblib

        from src.confidence.model import load_policy

        confidence_model = _joblib.load(args.confidence_model)
        abstention_policy = load_policy(args.abstention_policy)

    if tier2_clf is None:
        print("[warn] no Tier 2 classifier found — Tier 2 score defaults to 0.5 (neutral). Run src/frequency/train_svm.py.")
    if tier1_model is None:
        print("[warn] no Tier 1 checkpoint found — Tier 1 will be skipped. Run src/model/train.py.")
    if tier3_probe is None:
        print("[warn] no Tier 3 probe found — Tier 3 will be skipped. Run src/semantic/train_probe.py.")
    if fusion_model is None:
        print("[warn] no fusion model found — falling back to a simple mean of available tier scores. Train one via src/fusion.py's train_fusion().")

    results = []
    for path in list_images(args.image_dir):
        img = Image.open(path).convert("RGB")

        t0_score, early_exit, t0_raw = tier0_score(path, args.use_provenance_api)
        if early_exit:
            results.append({
                "image_path": path,
                "pred": 1.0,
                "label": "fake",
                "tiers_used": ["tier0"],
                "note": "early exit: trusted C2PA manifest names a known AI generator",
            })
            continue

        forensic = extract_all_features(img, max_dim=args.max_image_dim)
        t2_score = float(tier2_clf.predict_proba(forensic["vector"].reshape(1, -1))[0, 1]) if tier2_clf is not None else 0.5
        degradation = forensic["degradation_score"]

        transformation_profile = profiler.predict(img, forensic=forensic) if profiler is not None else None

        tiers_used = ["tier2"] if tier2_clf is not None else []
        t1_score = t3_score = None

        should_escalate = args.always_escalate or abs(t2_score - 0.5) < args.escalation_margin or t0_score is not None and abs(t0_score - t2_score) > 0.4

        if should_escalate and tier1_model is not None:
            from src.model.evaluate import score_images

            t1_score = score_images(tier1_model, [path])[0]
            tiers_used.append("tier1")

        if should_escalate and tier3_embedder is not None and tier3_probe is not None:
            emb = tier3_embedder.embed_image(img).numpy().reshape(1, -1)
            t3_score = float(tier3_probe.predict_proba(emb)[0, 1])
            tiers_used.append("tier3")

        fusion_pred = None
        if fusion_model is not None:
            fusion_pred = float(
                fusion_model.predict_proba(
                    build_features(t0_score or 0.0, t1_score if t1_score is not None else 0.5, t2_score, t3_score if t3_score is not None else 0.5, degradation).reshape(1, -1)
                )[0, 1]
            )

        ev = Evidence(tier1=t1_score, tier2=t2_score, tier3=t3_score, tier0=t0_score,
                      degradation=degradation, profile=transformation_profile,
                      min_side=min(img.size))
        router_pred = router.predict_one(ev) if router is not None else None

        # Adaptive router (Feature 3) is the primary combiner when available; the existing
        # degradation-aware fusion is the fallback, then a plain mean.
        if router_pred is not None:
            pred, combiner = router_pred, "adaptive_router"
        elif fusion_pred is not None:
            pred, combiner = fusion_pred, "existing_fusion"
        else:
            available = [s for s in (t0_score, t1_score, t2_score, t3_score) if s is not None]
            pred, combiner = (float(np.mean(available)) if available else 0.5), "mean"

        if t0_score is not None:
            tiers_used.insert(0, "tier0")

        result = {
            "image_path": path,
            "pred": pred,
            "label": "fake" if pred >= args.threshold else "real",
            "combiner": combiner,
            "tiers_used": tiers_used,
            "tier_scores": {"tier0": t0_score, "tier1": t1_score, "tier2": t2_score, "tier3": t3_score},
            "fusion_pred": fusion_pred,
            "router_pred": router_pred,
            "degradation_score": degradation,
            "transformation_profile": transformation_profile,
        }

        # ---- Feature 4: confidence + abstention (AI / REAL / UNKNOWN) ----------
        if confidence_model is not None and abstention_policy is not None:
            from src.confidence.signals import confidence_signals, is_hard_case

            ev4 = ev
            multicrop = None
            if args.multicrop_n > 0 and is_hard_case(pred, ev4):
                multicrop = _multi_crop(img, args, tier1_model, tier2_clf, tier3_embedder,
                                        tier3_probe, fusion_model, router, profiler, args.multicrop_n)
                result["multicrop"] = multicrop
            sig, sig_detail = confidence_signals(ev4, pred, multicrop)
            conf = float(confidence_model.predict_confidence(sig.reshape(1, -1))[0])
            decision = abstention_policy.decide(pred, conf)
            result.update({
                "outcome": decision["label"],          # AI / REAL / UNKNOWN
                "confidence": decision["confidence"],   # HIGH / MEDIUM / LOW
                "confidence_score": conf,
                "abstained": decision["abstained"],
                "confidence_signals": sig_detail,
            })

        results.append(result)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
