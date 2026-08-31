"""Checks for Feature 5 (cross-generator holdout + cost model + research table)."""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd

from src.eval.cost_model import COST_UNITS, method_cost
from src.eval.cross_generator import discover_families


def test_cost_model_ordering_and_multicrop():
    # forensic is the reference unit
    assert method_cost("baseline_forensic") == COST_UNITS["tier2_forensic"]
    assert method_cost("baseline_static_ensemble") > method_cost("baseline_cnn")
    cheap = method_cost("router_profiler", escalated=False, profiler=True)
    dear = method_cost("router_profiler", escalated=True, profiler=True)
    assert dear > cheap
    assert method_cost("full_system", multicrop_crops=5) > method_cost("full_system", multicrop_crops=0)


def test_discover_families():
    with tempfile.TemporaryDirectory() as d:
        for fam in ("gan", "diffusion", "not_a_family"):
            for cls in ("real", "fake"):
                os.makedirs(os.path.join(d, fam, cls), exist_ok=True)
        # break one family by removing its 'fake' subdir
        os.rmdir(os.path.join(d, "not_a_family", "fake"))
        fams = discover_families(d)
        assert fams == ["diffusion", "gan"]


def test_research_table_merges_partial_inputs():
    from src.eval import research_table as rt

    with tempfile.TemporaryDirectory() as d:
        rob = pd.DataFrame({"model": ["cnn", "adaptive_router"],
                            "clean_auc": [0.75, 0.95], "robust_auc_mean": [0.7, 0.9]})
        rob.to_csv(os.path.join(d, "summary.csv"), index=False)
        abst = {"summary": {"abstention_rate": 0.12}}
        with open(os.path.join(d, "abst.json"), "w") as f:
            json.dump(abst, f)

        import sys

        argv = sys.argv
        sys.argv = ["research_table", "--robustness", os.path.join(d, "summary.csv"),
                    "--ablation", os.path.join(d, "missing.csv"),
                    "--cross-gen", os.path.join(d, "missing2.csv"),
                    "--abstention", os.path.join(d, "abst.json"),
                    "--out-dir", os.path.join(d, "out")]
        try:
            rt.main()
        finally:
            sys.argv = argv
        df = pd.read_csv(os.path.join(d, "out", "research_table.csv"))
        assert "Method" in df.columns and len(df) == 7
        cnn = df[df.Method == "CNN only"].iloc[0]
        assert abs(cnn["Clean AUC"] - 0.75) < 1e-6
        full = df[df.Method.str.startswith("Full system")].iloc[0]
        assert abs(full["Abstention rate"] - 0.12) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
