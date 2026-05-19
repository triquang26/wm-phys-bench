#!/usr/bin/env python3
"""Multi-aggregator calibration — keep real SAFE, catch more hallu.

Goal: with the same per-frame signals, find which video-level aggregator
(or combination) gives the best ratio of (gen caught) to (real flagged).

Aggregators evaluated:
  H_max         max frame H_score                  — sensitive to 1 bad frame
  H_peak        80th percentile                    — current default
  H_robust      trimmed mean (10% tails)           — robust to single-frame noise
  H_count_hi    fraction of frames > 0.90          — counts persistent anomaly
  H_count_med   fraction of frames > 0.75          — counts mild anomaly
  AND_vote      flag iff (H_peak > T1) AND (H_robust > T2)  — both must agree

Calibration policy:
  Set per-aggregator threshold = (max of real) + epsilon → FPR = 0% on real.
  Report: which aggregator catches the most generated videos at FPR=0?

Requires per-frame H scores from v3 eval — re-runs score_video.py if needed.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
BENCH = REPO_ROOT / "paper-physical-gr1"
EVAL = BENCH / "eval_results_v3"
CACHE = BENCH / "ref_cache"


def run_per_frame_collection(out_dir: Path) -> Path:
    """Run a modified eval that saves per-frame H scores into a JSON map.

    Re-uses score_video.py — adds JSON output for each video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "per_frame_scores.json"
    if out_json.exists():
        print(f"Per-frame scores already exist: {out_json}")
        return out_json

    print(f"Per-frame scores not found — need to re-run eval with JSON dump.")
    print(f"Run scripts/eval_videos.py with extended JSON output first.")
    sys.exit(1)


def aggregate_from_per_frame(per_frame_h: list[float]) -> dict[str, float]:
    """Compute all candidate aggregators from a list of per-frame H_scores."""
    h = np.asarray(per_frame_h, dtype=np.float32)
    if h.size == 0:
        return {}
    n = h.size
    # trimmed mean (10%)
    k = max(int(n * 0.1), 0)
    sorted_h = np.sort(h)
    h_trim = sorted_h[k : n - k].mean() if (n - 2 * k) > 0 else h.mean()

    return {
        "h_max":       float(h.max()),
        "h_peak":      float(np.percentile(h, 80)),
        "h_robust":    float(h_trim),
        "h_p95":       float(np.percentile(h, 95)),
        "h_count_hi":  float((h > 0.90).mean()),    # fraction
        "h_count_med": float((h > 0.75).mean()),
        "h_mean":      float(h.mean()),
        "h_median":    float(np.median(h)),
    }


def evaluate_aggregator(name: str,
                        real_scores: np.ndarray,
                        gen_scores:  np.ndarray) -> dict:
    """Compute FPR-0% threshold + gen catch rate for one aggregator."""
    from sklearn.metrics import roc_auc_score
    # Threshold = max(real) + epsilon → 0% FPR guarantee on calibration set
    T_safe = float(real_scores.max()) + 1e-6
    n_gen_above = int((gen_scores > T_safe).sum())
    # Also: threshold = p95(real) → 5% FPR
    T_p95 = float(np.quantile(real_scores, 0.95))
    n_real_above_p95 = int((real_scores > T_p95).sum())
    n_gen_above_p95 = int((gen_scores > T_p95).sum())
    # Threshold = p99(real) → 1% FPR
    T_p99 = float(np.quantile(real_scores, 0.99))
    n_gen_above_p99 = int((gen_scores > T_p99).sum())

    # AUROC (using gen as positive — purely diagnostic)
    if len(set([0] * len(real_scores) + [1] * len(gen_scores))) >= 2:
        y = np.concatenate([np.zeros_like(real_scores), np.ones_like(gen_scores)])
        s = np.concatenate([real_scores, gen_scores])
        auroc = float(roc_auc_score(y, s))
    else:
        auroc = float("nan")

    return {
        "name":             name,
        "real_max":         float(real_scores.max()),
        "real_p95":         float(np.quantile(real_scores, 0.95)),
        "real_mean":        float(real_scores.mean()),
        "gen_mean":         float(gen_scores.mean()),
        "auroc":            auroc,
        "threshold_safe":   T_safe,
        "gen_caught_safe":  n_gen_above,
        "gen_rate_safe":    float(n_gen_above / len(gen_scores)),
        "threshold_p95":    T_p95,
        "gen_caught_p95":   n_gen_above_p95,
        "real_flagged_p95": n_real_above_p95,
        "threshold_p99":    T_p99,
        "gen_caught_p99":   n_gen_above_p99,
    }


def main():
    # ── Load per-video data
    rows = list(csv.DictReader(open(EVAL / "video_table.csv")))
    rows = [r for r in rows if r.get("h_peak") not in ("", "None")]

    # The CSV has h_peak, h_robust, h_max — that's enough for basic aggregators.
    # For h_count_hi/med we need per-frame which isn't in CSV; rerun if needed.
    # For now: evaluate h_peak, h_robust, h_max, plus AND-vote.

    reals = [r for r in rows if r["type"] == "REAL"]
    gens  = [r for r in rows if r["type"] == "GEN"]

    aggregators = {
        "h_peak":   np.array([float(r["h_peak"])   for r in rows]),
        "h_robust": np.array([float(r["h_robust"]) for r in rows]),
        "h_max":    np.array([float(r["h_max"])    for r in rows]),
    }

    print("=" * 78)
    print("MULTI-AGGREGATOR CALIBRATION — keep real safe, catch more hallu")
    print("=" * 78)

    results = {}
    for name, scores in aggregators.items():
        real_s = np.array([float(r[name]) for r in reals])
        gen_s  = np.array([float(r[name]) for r in gens])
        res = evaluate_aggregator(name, real_s, gen_s)
        results[name] = res

    # ── Print comparison table
    print(f"\n{'Aggregator':<14} {'AUROC':>7} {'real_mean':>10} {'gen_mean':>10} "
          f"{'T@FPR=0':>10} {'gen@FPR=0':>11} {'gen_rate':>10}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:<14} {r['auroc']:>7.4f} {r['real_mean']:>10.4f} {r['gen_mean']:>10.4f} "
              f"{r['threshold_safe']:>10.4f} "
              f"{r['gen_caught_safe']:>3d}/{len(gens):d}"
              f"{'':>5} {r['gen_rate_safe']:>10.4f}")

    print(f"\n{'Aggregator':<14} {'T@FPR≤5%':>10} {'real flagged':>14} {'gen caught':>11}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:<14} {r['threshold_p95']:>10.4f} "
              f"{(np.array([float(rr[name]) for rr in reals]) > r['threshold_p95']).sum():>3d}/{len(reals):d}"
              f"{'':>8} {r['gen_caught_p95']:>3d}/{len(gens):d}")

    # ── AND-voting: real safe at p95 of EACH signal
    print("\n" + "=" * 78)
    print("AND-VOTING (multi-signal agreement)")
    print("=" * 78)

    # Per-signal thresholds at max(real)+eps each
    T_peak   = float(np.array([float(r["h_peak"])   for r in reals]).max()) + 1e-6
    T_robust = float(np.array([float(r["h_robust"]) for r in reals]).max()) + 1e-6
    print(f"  threshold (h_peak)   = {T_peak:.4f}   (max real + eps)")
    print(f"  threshold (h_robust) = {T_robust:.4f}   (max real + eps)")

    # Real videos: which trigger AND-vote?
    real_and = sum(
        1 for r in reals
        if float(r["h_peak"]) > T_peak and float(r["h_robust"]) > T_robust
    )
    gen_and = sum(
        1 for r in gens
        if float(r["h_peak"]) > T_peak and float(r["h_robust"]) > T_robust
    )
    print(f"  Real flagged (AND): {real_and}/{len(reals)}  (FPR = {real_and/len(reals):.4f})")
    print(f"  Gen caught (AND):   {gen_and}/{len(gens)}    (rate = {gen_and/len(gens):.4f})")

    # OR-voting: video flagged if EITHER aggregator triggers
    real_or = sum(
        1 for r in reals
        if float(r["h_peak"]) > T_peak or float(r["h_robust"]) > T_robust
    )
    gen_or = sum(
        1 for r in gens
        if float(r["h_peak"]) > T_peak or float(r["h_robust"]) > T_robust
    )
    print(f"\nOR-vote (less strict, more sensitive):")
    print(f"  Real flagged (OR): {real_or}/{len(reals)}  (FPR = {real_or/len(reals):.4f})")
    print(f"  Gen caught (OR):   {gen_or}/{len(gens)}    (rate = {gen_or/len(gens):.4f})")

    # ── Pick best single aggregator at FPR=0
    best = max(results.items(), key=lambda kv: kv[1]["gen_caught_safe"])
    print(f"\n{'=' * 78}")
    print(f"BEST AGGREGATOR AT FPR=0% on real: {best[0].upper()}")
    print(f"{'=' * 78}")
    print(f"  threshold:        {best[1]['threshold_safe']:.4f}")
    print(f"  real flagged:     0/{len(reals)}  (FPR = 0% by construction)")
    print(f"  gen caught:       {best[1]['gen_caught_safe']}/{len(gens)}  "
          f"({best[1]['gen_rate_safe']:.1%})")

    # ── Update threshold.json using best aggregator
    threshold_payload = {
        "threshold":    best[1]["threshold_safe"],
        "config":       "S2+S4",
        "aggregator":   best[0],   # which aggregator to use at scoring time
        "mode":         "conformal_safe_real_only",
        "calibration_set_size": len(reals),
        "n_gen_above":  best[1]["gen_caught_safe"],
        "real_max":     best[1]["real_max"],
        "all_aggregator_results": results,
    }
    (CACHE / "threshold.json").write_text(json.dumps(threshold_payload, indent=2))
    print(f"\nThreshold + best aggregator → {CACHE / 'threshold.json'}")

    # ── Markdown
    md = []
    md.append("# WarpDyn — Safe + Sensitive calibration\n")
    md.append("**Premise:** Real training videos = true negatives. Calibrate to **0% FPR on real**, then report how many generated videos still trip the alarm (likely hallu).\n")
    md.append(f"\n## Aggregator comparison (FPR = 0% on {len(reals)} real videos)\n")
    md.append("| Aggregator | AUROC | mean(real) | mean(gen) | T@safe | Gen caught | Catch rate |")
    md.append("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        md.append(f"| `{name}` | {r['auroc']:.4f} | {r['real_mean']:.4f} | {r['gen_mean']:.4f} | "
                  f"{r['threshold_safe']:.4f} | {r['gen_caught_safe']}/{len(gens)} | "
                  f"{r['gen_rate_safe']:.1%} |")
    md.append(f"\n## Multi-signal voting (per-signal threshold = max(real)+eps)\n")
    md.append(f"- **AND-vote** (both must agree): real flagged = `{real_and}/{len(reals)}`, gen caught = `{gen_and}/{len(gens)}`")
    md.append(f"- **OR-vote** (either suffices):  real flagged = `{real_or}/{len(reals)}`, gen caught = `{gen_or}/{len(gens)}`")
    md.append(f"\n## Recommended setup\n")
    md.append(f"- Aggregator: **`{best[0]}`**")
    md.append(f"- Threshold:  **`{best[1]['threshold_safe']:.4f}`**  (= max(real) + ε)")
    md.append(f"- Guarantee:  FPR = 0% on calibration set ({len(reals)} real videos)")
    md.append(f"- Gen catch:  **{best[1]['gen_caught_safe']}/{len(gens)} ({best[1]['gen_rate_safe']:.1%})**")
    (BENCH / "calibration_safe_sensitive.md").write_text("\n".join(md))
    print(f"Report → {BENCH / 'calibration_safe_sensitive.md'}")


if __name__ == "__main__":
    main()
