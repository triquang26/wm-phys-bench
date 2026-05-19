#!/usr/bin/env python3
"""Per-task ratio score: H_test / H_train_task — task-fair anomaly metric.

For each task T, the real training video's H scores are used as the per-task
baseline. ratio > 1.0 → more anomalous than training → likely hallu. By
construction FPR = 0% on real (each task's threshold = its own training H).

Reads per_task_dense_table.csv. Produces ratio table + markdown ranking.

When the table has k-NN/fused columns (from fusion eval), emits 3 ratios
(cycle, knn, fused) + complementarity matrix + separation gap.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
DEFAULT_EVAL_DIR = REPO_ROOT / "paper-physical-gr1" / "per_task_dense_eval"


def verdict(ratio: float) -> str:
    if ratio > 1.0:
        return "HALLU"
    if ratio > 0.95:
        return "borderline"
    return "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    ap.add_argument("--in_csv", default="per_task_dense_table.csv")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    in_csv = args.eval_dir / args.in_csv
    rows = list(csv.DictReader(open(in_csv)))
    if not rows:
        print(f"No rows in {in_csv}")
        return

    has_knn = "knn_peak" in rows[0] and rows[0].get("knn_peak", "") != ""
    has_fused = "fused_peak" in rows[0] and rows[0].get("fused_peak", "") != ""

    # Per-task H_train for each signal
    def col(r, c, default=0.0):
        v = r.get(c, "")
        try:
            return float(v) if v != "" else default
        except (TypeError, ValueError):
            return default

    h_train_cycle = {r["task"]: col(r, "cycle_peak") for r in rows if r["type"] == "REAL"}
    h_train_knn = {r["task"]: col(r, "knn_peak") for r in rows if r["type"] == "REAL"} if has_knn else {}
    h_train_fused = {r["task"]: col(r, "fused_peak") for r in rows if r["type"] == "REAL"} if has_fused else {}

    out_rows = []
    for r in rows:
        h_c = col(r, "cycle_peak")
        ht_c = h_train_cycle.get(r["task"], 1e-8)
        ratio_c = h_c / ht_c if ht_c > 1e-8 else 0.0

        out = {
            "type":         r["type"],
            "task":         r["task"],
            "video":        r["video"],
            "h_cycle":      h_c,
            "ht_cycle":     ht_c,
            "ratio_cycle":  ratio_c,
            "verdict_cycle": verdict(ratio_c),
        }

        if has_knn:
            h_k = col(r, "knn_peak")
            ht_k = h_train_knn.get(r["task"], 1e-8)
            ratio_k = h_k / ht_k if ht_k > 1e-8 else 0.0
            out["h_knn"] = h_k
            out["ht_knn"] = ht_k
            out["ratio_knn"] = ratio_k
            out["verdict_knn"] = verdict(ratio_k)
            out["knn_route"] = r.get("knn_route", "")

        if has_fused:
            h_f = col(r, "fused_peak")
            ht_f = h_train_fused.get(r["task"], 1e-8)
            ratio_f = h_f / ht_f if ht_f > 1e-8 else 0.0
            out["h_fused"] = h_f
            out["ht_fused"] = ht_f
            out["ratio_fused"] = ratio_f
            out["verdict_fused"] = verdict(ratio_f)

        out["label"] = r.get("label", "")
        out_rows.append(out)

    # Sort by fused if available else cycle
    sort_key = "ratio_fused" if has_fused else "ratio_cycle"
    out_rows.sort(key=lambda x: -x[sort_key])

    out_csv = args.eval_dir / f"per_task_ratio_table{args.out_suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"CSV → {out_csv}")

    # ── Markdown ─────────────────────────────────────────────────────────
    md = ["# WarpDyn — Per-task ratio ranking\n"]
    if has_fused:
        md.append("Fusion mode: **cycle + k-NN** via video-level Cauchy combine.\n")
    else:
        md.append("Cycle-only mode (k-NN signal disabled).\n")

    md.append("## Per-task training baselines\n")
    hdr = ["Task", "H_cycle"] + (["H_knn", "route"] if has_knn else []) + (["H_fused"] if has_fused else [])
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for t in sorted(h_train_cycle):
        cells = [t, f"{h_train_cycle[t]:.4f}"]
        if has_knn:
            knn_route = next((r["knn_route"] for r in out_rows
                              if r["task"] == t and r["type"] == "REAL"), "")
            cells += [f"{h_train_knn[t]:.4f}", knn_route]
        if has_fused:
            cells += [f"{h_train_fused[t]:.4f}"]
        md.append("| " + " | ".join(cells) + " |")

    # FPR check on real
    reals = [r for r in out_rows if r["type"] == "REAL"]
    md.append("\n## FPR check on real (must all be ≤ 1.0)\n")
    md.append("| Task | ratio_cycle" +
              (" | ratio_knn" if has_knn else "") +
              (" | ratio_fused" if has_fused else "") + " |")
    md.append("|---|---" + ("|---" if has_knn else "") + ("|---" if has_fused else "") + "|")
    for r in reals:
        cells = [r["task"], f"{r['ratio_cycle']:.3f}"]
        if has_knn:
            cells.append(f"{r['ratio_knn']:.3f}")
        if has_fused:
            cells.append(f"{r['ratio_fused']:.3f}")
        md.append("| " + " | ".join(cells) + " |")

    real_flagged_cycle = sum(1 for r in reals if r["ratio_cycle"] > 1.0)
    md.append(f"\nReal flagged by cycle: **{real_flagged_cycle}/{len(reals)}** (FPR=0% required)")
    if has_knn:
        real_flagged_knn = sum(1 for r in reals if r["ratio_knn"] > 1.0)
        md.append(f"Real flagged by knn:   **{real_flagged_knn}/{len(reals)}**")
    if has_fused:
        real_flagged_fused = sum(1 for r in reals if r["ratio_fused"] > 1.0)
        md.append(f"Real flagged by fused: **{real_flagged_fused}/{len(reals)}** ← decision metric")

    # ── Complementarity matrix (label-free) ─────────────────────────────
    gens = [r for r in out_rows if r["type"] == "GEN"]
    if has_knn and has_fused:
        from warp_score.fusion import complementarity_report, separation_gap, borderline_count
        comp = complementarity_report(out_rows)
        md.append("\n## Complementarity (label-free)\n")
        md.append(f"Among {comp['n_gens']} gens (catch = ratio > 1.0):")
        md.append("")
        md.append(f"- both cycle+knn caught: **{comp['both']}**")
        md.append(f"- cycle only:            **{comp['cycle_only']}**")
        md.append(f"- knn only:              **{comp['knn_only']}**")
        md.append(f"- neither caught:        **{comp['neither']}**")
        md.append(f"- fused total catch:     **{comp['fused_catch']}/{comp['n_gens']}**")
        md.append(f"- cycle-alone catch:     {comp['cycle_catch']}/{comp['n_gens']}")
        md.append(f"- knn-alone catch:       {comp['knn_catch']}/{comp['n_gens']}")

        gap_cycle = separation_gap(out_rows, "ratio_cycle")
        gap_knn = separation_gap(out_rows, "ratio_knn")
        gap_fused = separation_gap(out_rows, "ratio_fused")
        md.append("\n## Separation gap  (min ratio over HALLU gens − max ratio over reals; higher = cleaner)\n")
        md.append(f"- cycle: {gap_cycle:+.3f}")
        md.append(f"- knn:   {gap_knn:+.3f}")
        md.append(f"- fused: {gap_fused:+.3f}")

        bord_cycle = borderline_count(out_rows, "ratio_cycle")
        bord_fused = borderline_count(out_rows, "ratio_fused")
        md.append(f"\n## Borderline shrinkage (rows in 0.95–1.05 zone)\n")
        md.append(f"- cycle: {bord_cycle}")
        md.append(f"- fused: {bord_fused}")

    # ── Verdict summary ─────────────────────────────────────────────────
    n_h_cycle = sum(1 for r in gens if r["ratio_cycle"] > 1.0)
    md.append("\n## Verdict summary (gens, HALLU = ratio > 1.0)\n")
    md.append(f"- cycle: {n_h_cycle}/{len(gens)}")
    if has_knn:
        n_h_knn = sum(1 for r in gens if r["ratio_knn"] > 1.0)
        md.append(f"- knn:   {n_h_knn}/{len(gens)}")
    if has_fused:
        n_h_fused = sum(1 for r in gens if r["ratio_fused"] > 1.0)
        md.append(f"- fused: {n_h_fused}/{len(gens)}  ← decision metric")

    # ── Full gen ranking ────────────────────────────────────────────────
    md.append(f"\n## Full gen ranking (sorted by {sort_key})\n")
    hdr = ["Rank", "Task", "Vid"]
    if has_fused:
        hdr += ["ratio_fused", "ratio_cycle", "ratio_knn", "verdict_fused"]
    elif has_knn:
        hdr += ["ratio_cycle", "ratio_knn", "verdict_cycle"]
    else:
        hdr += ["ratio_cycle", "verdict_cycle"]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "|".join(["---"] * len(hdr)) + "|")
    emoji = {"HALLU": "🔴", "borderline": "⚠", "clean": "✓"}
    for i, r in enumerate(gens, 1):
        cells = [str(i), r["task"], r["video"]]
        if has_fused:
            cells += [
                f"**{r['ratio_fused']:.3f}**",
                f"{r['ratio_cycle']:.3f}",
                f"{r['ratio_knn']:.3f}",
                f"{emoji.get(r['verdict_fused'], '?')} {r['verdict_fused']}",
            ]
        elif has_knn:
            cells += [
                f"{r['ratio_cycle']:.3f}",
                f"{r['ratio_knn']:.3f}",
                f"{emoji.get(r['verdict_cycle'], '?')} {r['verdict_cycle']}",
            ]
        else:
            cells += [
                f"**{r['ratio_cycle']:.3f}**",
                f"{emoji.get(r['verdict_cycle'], '?')} {r['verdict_cycle']}",
            ]
        md.append("| " + " | ".join(cells) + " |")

    out_md = args.eval_dir / f"per_task_ratio_ranking{args.out_suffix}.md"
    out_md.write_text("\n".join(md))
    print(f"Markdown → {out_md}")

    # ── Stdout summary ──────────────────────────────────────────────────
    print(f"\nReal flagged (cycle): {real_flagged_cycle}/{len(reals)}")
    if has_knn:
        print(f"Real flagged (knn):   {sum(1 for r in reals if r['ratio_knn'] > 1.0)}/{len(reals)}")
    if has_fused:
        print(f"Real flagged (fused): {sum(1 for r in reals if r['ratio_fused'] > 1.0)}/{len(reals)}")
    print(f"\nGen HALLU (cycle): {n_h_cycle}/{len(gens)}")
    if has_knn:
        print(f"Gen HALLU (knn):   {sum(1 for r in gens if r['ratio_knn'] > 1.0)}/{len(gens)}")
    if has_fused:
        print(f"Gen HALLU (fused): {sum(1 for r in gens if r['ratio_fused'] > 1.0)}/{len(gens)}")


if __name__ == "__main__":
    main()
