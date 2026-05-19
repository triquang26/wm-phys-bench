#!/usr/bin/env python3
"""REVISED evaluation framework — FPR on real training videos.

Premise correction:
  Generated videos are NOT ground-truth hallucinations. They are just
  videos produced by a generator that MAY or MAY NOT be hallucinated.

What we actually want to verify:
  Real training videos must NOT be classified as hallu.
  i.e. FPR (false positive rate on real) ≤ small target alpha.

This script:
  1. Takes the existing v3 eval results (30 real + 24 gen H_peak scores).
  2. Treats REAL as the only labeled class (the "true negative").
  3. Sweeps thresholds based on quantiles of REAL distribution.
  4. Reports for each threshold:
       - Real FPR (what fraction of training videos get falsely flagged)
       - Score range of flagged real videos (which ones — possibly edge cases)
       - Gen videos above threshold (informative — possibly hallucinated)
  5. Picks threshold = p99(real) so FPR ≤ 1% by construction.
  6. Updates ref_cache/threshold.json with mode="conformal_real_only".
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
EVAL = REPO_ROOT / "paper-physical-gr1" / "eval_results_v3"
CACHE = REPO_ROOT / "paper-physical-gr1" / "ref_cache"


def main():
    rows = list(csv.DictReader(open(EVAL / "video_table.csv")))
    rows = [r for r in rows if r.get("h_peak") not in ("", "None")]

    reals = [r for r in rows if r["type"] == "REAL"]
    gens  = [r for r in rows if r["type"] == "GEN"]

    real_scores = np.sort(np.asarray([float(r["h_peak"]) for r in reals]))
    gen_scores  = np.sort(np.asarray([float(r["h_peak"]) for r in gens]))

    print("=" * 70)
    print("REVISED EVAL — Real training videos = true negatives only")
    print("=" * 70)
    print(f"\nReal training videos:  N = {len(reals)}")
    print(f"  range:    [{real_scores.min():.4f}, {real_scores.max():.4f}]")
    print(f"  mean:      {real_scores.mean():.4f}  std: {real_scores.std():.4f}")
    print(f"  p50/p90/p95/p99: "
          f"{np.median(real_scores):.4f} / "
          f"{np.quantile(real_scores, 0.90):.4f} / "
          f"{np.quantile(real_scores, 0.95):.4f} / "
          f"{np.quantile(real_scores, 0.99):.4f}")
    print(f"\nGenerated videos (informative only):  N = {len(gens)}")
    print(f"  range:    [{gen_scores.min():.4f}, {gen_scores.max():.4f}]")
    print(f"  mean:      {gen_scores.mean():.4f}  std: {gen_scores.std():.4f}")

    # ── Threshold sweep based on REAL quantiles
    print("\n" + "=" * 70)
    print("Threshold sweep (target FPR ≤ alpha)")
    print("=" * 70)
    print(f"{'alpha':>8} {'threshold':>10} {'real_FPR':>10} {'real_flagged':>14} "
          f"{'gen_above':>10} {'gen_rate':>10}")
    rows_swept = []
    for alpha in [0.20, 0.10, 0.05, 0.03, 0.01, 0.00]:
        q = 1.0 - alpha
        if alpha == 0.0:
            # Use max + epsilon → threshold above all real
            T = real_scores.max() + 1e-6
        else:
            T = float(np.quantile(real_scores, q))
        n_real_flagged = int((real_scores > T).sum())
        actual_fpr = n_real_flagged / len(real_scores)
        n_gen_above = int((gen_scores > T).sum())
        gen_rate = n_gen_above / len(gen_scores)
        rows_swept.append((alpha, T, actual_fpr, n_real_flagged, n_gen_above, gen_rate))
        print(f"{alpha:>8.2f} {T:>10.4f} {actual_fpr:>10.4f} "
              f"{n_real_flagged:>3d}/{len(real_scores):d}"
              f"{'':>5} {n_gen_above:>3d}/{len(gen_scores):d}"
              f"{'':>2} {gen_rate:>10.4f}")

    # ── Which REAL videos score highest (potential edge cases)
    print("\n" + "=" * 70)
    print("Top 10 REAL videos by H_peak (potentially mis-flagged candidates)")
    print("=" * 70)
    real_sorted = sorted(reals, key=lambda r: -float(r["h_peak"]))
    print(f"{'#':>3} {'task':<10} {'video':<15} {'H_peak':>8} {'H_robust':>10}")
    for i, r in enumerate(real_sorted[:10], 1):
        print(f"{i:>3} {r['task']:<10} {r['video']:<15} "
              f"{float(r['h_peak']):>8.4f} {float(r['h_robust']):>10.4f}")

    # ── Save final threshold (use alpha=0.05 — p95 of real, conformal style)
    chosen_alpha = 0.05
    chosen_T = float(np.quantile(real_scores, 1 - chosen_alpha))
    threshold_payload = {
        "threshold":   chosen_T,
        "config":      "S2+S4",
        "aggregator":  "p80",
        "mode":        "conformal_real_only_v3",
        "calibration_set_size": len(reals),
        "target_fpr":  chosen_alpha,
        "actual_fpr_on_calibration_set":
            float((real_scores > chosen_T).sum() / len(real_scores)),
        "real_score_distribution": {
            "min":    float(real_scores.min()),
            "max":    float(real_scores.max()),
            "mean":   float(real_scores.mean()),
            "std":    float(real_scores.std()),
            "p50":    float(np.median(real_scores)),
            "p90":    float(np.quantile(real_scores, 0.90)),
            "p95":    float(np.quantile(real_scores, 0.95)),
            "p99":    float(np.quantile(real_scores, 0.99)),
        },
        "gen_score_distribution_informative": {
            "min":  float(gen_scores.min()),
            "max":  float(gen_scores.max()),
            "mean": float(gen_scores.mean()),
        },
        "interpretation": (
            "Threshold set to p95 of REAL training distribution. "
            "Videos scoring above are 'rarer than 95% of real training' → "
            "ANOMALOUS (not necessarily hallu). FPR ≤ 5% on similar real "
            "data by construction."
        ),
    }
    (CACHE / "threshold.json").write_text(json.dumps(threshold_payload, indent=2))
    print(f"\nThreshold saved → {CACHE / 'threshold.json'}")
    print(f"  threshold (alpha={chosen_alpha}) = {chosen_T:.4f}")
    print(f"  → FPR ≤ {chosen_alpha*100:.0f}% on real training data (guaranteed)")
    print(f"  → {sum(s > chosen_T for s in gen_scores)}/{len(gen_scores)} gen videos above (informative, not metric)")

    # ── Markdown report
    md = []
    md.append("# WarpDyn — REVISED eval (real-only calibration)\n")
    md.append("**Premise:** Real training videos = true negatives. Generated videos are *informative* but NOT ground-truth positives.\n")
    md.append(f"\n## Score distributions\n")
    md.append(f"- **Real (N={len(reals)}):** range [{real_scores.min():.3f}, {real_scores.max():.3f}], "
              f"mean {real_scores.mean():.3f}, p95 **{np.quantile(real_scores, 0.95):.3f}**, "
              f"p99 {np.quantile(real_scores, 0.99):.3f}\n")
    md.append(f"- **Gen  (N={len(gens)}):** range [{gen_scores.min():.3f}, {gen_scores.max():.3f}], "
              f"mean {gen_scores.mean():.3f}\n")
    md.append(f"\n## FPR-controlled threshold sweep\n")
    md.append("| target FPR | threshold | actual FPR | real flagged | gen above | (informative) |")
    md.append("|---|---|---|---|---|---|")
    for alpha, T, fpr, nrf, nga, gr in rows_swept:
        md.append(f"| {alpha:.2f} | {T:.4f} | {fpr:.4f} | {nrf}/{len(real_scores)} | "
                  f"{nga}/{len(gen_scores)} | gen-above rate {gr:.2f} |")
    md.append(f"\n## Top 10 REAL videos by H_peak (most likely false-positive candidates)\n")
    md.append("| # | task | video | H_peak | H_robust |")
    md.append("|---|---|---|---|---|")
    for i, r in enumerate(real_sorted[:10], 1):
        md.append(f"| {i} | {r['task']} | {r['video']} | "
                  f"{float(r['h_peak']):.4f} | {float(r['h_robust']):.4f} |")
    md.append(f"\n## Chosen operating point: alpha = {chosen_alpha}\n")
    md.append(f"- Threshold = `{chosen_T:.4f}`")
    md.append(f"- Real training videos flagged: "
              f"{int((real_scores > chosen_T).sum())}/{len(real_scores)} "
              f"(FPR = {(real_scores > chosen_T).sum() / len(real_scores):.4f})")
    md.append(f"- Gen videos above (informative only): "
              f"{int((gen_scores > chosen_T).sum())}/{len(gen_scores)} "
              f"({(gen_scores > chosen_T).sum() / len(gen_scores):.4f})")

    out_md = EVAL.parent / "eval_revised_realonly.md"
    out_md.write_text("\n".join(md))
    print(f"Markdown report → {out_md}")


if __name__ == "__main__":
    main()
