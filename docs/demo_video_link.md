# Demo Video

- **YouTube link (public)**: _TODO — record and paste the public YouTube URL here before submission._
- **Length**: 2-4 minutes.
- **Suggested structure**:
  1. Problem framing (10-20s): why AIGC detection matters for platform trust/moderation, why
     generalization to new generators + robustness to transforms is the hard part.
  2. Architecture walkthrough (60-90s): the 4-tier cascade + adaptive fusion, using the README
     diagram — call out what's cheap (Tier 0/2, every upload) vs. expensive (Tier 1/3, escalation
     only).
  3. Live demo (60-90s): run `python -m src.infer <dir>` on a few clean + transformed images,
     show the JSON output and a spectrum plot from `src/frequency/visualize.py`.
  4. Robustness table + cross-generator result (20-30s): show `outputs/robustness_summary.csv`
     and the cross-generator gap from `docs/shortcut_learning_check.md`.
  5. Close: what's novel vs. standard practice (README "Related Work" section).
- Ensure the video contains no third-party IP (music, footage) you don't have rights to.
