#!/usr/bin/env python3
"""DROID per-task ratio computation for the multi-view pipeline.

Reads `paper-physical-droid/per_task_dense_eval/per_task_dense_table.csv`
produced by `scripts/eval_droid_multiview.py` and computes, for each video:

  * per-view ratios:    ratio_<view>_cycle, ratio_<view>_knn, ratio_<view>_fused
  * multiview ratio:    ratio_multiview = multiview_fused_peak / H_train_multiview
  * verdicts (HALLU / borderline / clean) for every signal column

Outputs:
  paper-physical-droid/per_task_dense_eval/per_task_ratio_table.csv
  paper-physical-droid/per_task_dense_eval/per_task_ratio_ranking.md

The markdown includes:
  * per-task training baselines (per-view + multiview)
  * FPR check on real (must all be ≤ 1.0)
  * view-wise gen catch matrix (one row per gen, one col per view + multiview)
  * cross-view + cycle/knn complementarity (full 4-modality combinations)
  * full gen ranking by ratio_multiview
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPO_ROOT = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
DEFAULT_EVAL_DIR = REPO_ROOT / "paper-physical-droid" / "per_task_dense_eval"
DEFAULT_VIEWS = ["exterior_1", "exterior_2", "wrist"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _f(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v in (None, "", "nan"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def verdict(ratio: float) -> str:
    if math.isnan(ratio):
        return "unknown"
    if ratio > 1.0:
        return "HALLU"
    if ratio > 0.95:
        return "borderline"
    return "clean"


def safe_ratio(num: float, denom: float) -> float:
    if denom is None or denom <= 1e-8:
        return float("nan")
    return num / denom


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute DROID multi-view per-task ratios + markdown report."
    )
    ap.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR,
                    help=f"directory holding per_task_dense_table.csv "
                         f"(default: {DEFAULT_EVAL_DIR})")
    ap.add_argument("--in_csv", default="per_task_dense_table.csv",
                    help="input filename inside --eval_dir")
    ap.add_argument("--views", nargs="+", default=DEFAULT_VIEWS,
                    help=f"view tags to score (default: {' '.join(DEFAULT_VIEWS)})")
    ap.add_argument("--out_suffix", default="",
                    help="append suffix to output filenames")
    args = ap.parse_args()

    in_csv = args.eval_dir / args.in_csv
    if not in_csv.exists():
        raise FileNotFoundError(in_csv)

    rows = list(csv.DictReader(open(in_csv)))
    if not rows:
        print(f"no rows in {in_csv}")
        return

    views = list(args.views)
    sample = rows[0]
    have_views = [v for v in views if f"fused_{v}_peak" in sample]
    if have_views != views:
        print(f"[warn] columns present for views {have_views}, requested {views}")

    # ── Per-task training (REAL) baselines ──────────────────────────────
    h_train: dict[str, dict[str, float]] = {}   # task -> column_name -> value
    for r in rows:
        if r["type"] != "REAL":
            continue
        task = r["task"]
        h_train[task] = {}
        for v in views:
            h_train[task][f"cycle_{v}_peak"] = _f(r, f"cycle_{v}_peak")
            h_train[task][f"knn_{v}_peak"]   = _f(r, f"knn_{v}_peak")
            h_train[task][f"fused_{v}_peak"] = _f(r, f"fused_{v}_peak")
        h_train[task]["multiview_fused_peak"] = _f(r, "multiview_fused_peak")

    # ── Build output rows ───────────────────────────────────────────────
    out_rows: list[dict] = []
    for r in rows:
        task = r["task"]
        ref = h_train.get(task, {})
        out: dict = {
            "type": r["type"],
            "task": task,
            "video": r["video"],
            "label": r.get("label", ""),
        }
        for v in views:
            for sig in ["cycle", "knn", "fused"]:
                col = f"{sig}_{v}_peak"
                h = _f(r, col)
                ht = ref.get(col, 0.0)
                ratio = safe_ratio(h, ht)
                out[f"h_{sig}_{v}"] = h
                out[f"ht_{sig}_{v}"] = ht
                out[f"ratio_{sig}_{v}"] = ratio
                out[f"verdict_{sig}_{v}"] = verdict(ratio)
            out[f"knn_route_{v}"] = r.get(f"knn_{v}_route", "")

        # multiview
        h_mv = _f(r, "multiview_fused_peak")
        ht_mv = ref.get("multiview_fused_peak", 0.0)
        ratio_mv = safe_ratio(h_mv, ht_mv)
        out["h_multiview"] = h_mv
        out["ht_multiview"] = ht_mv
        out["ratio_multiview"] = ratio_mv
        out["verdict_multiview"] = verdict(ratio_mv)
        out["missing_demux_views"] = r.get("missing_demux_views", "")

        out_rows.append(out)

    # Sort by multiview ratio (NaNs last) then by task
    def _sort_key(o):
        rmv = o["ratio_multiview"]
        is_nan = isinstance(rmv, float) and math.isnan(rmv)
        return (1 if is_nan else 0, -(rmv if not is_nan else 0.0), o["task"], o["video"])

    out_rows.sort(key=_sort_key)

    out_csv = args.eval_dir / f"per_task_ratio_table{args.out_suffix}.csv"
    if out_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print(f"CSV → {out_csv}")

    # ── Markdown report ─────────────────────────────────────────────────
    md: list[str] = ["# WarpDyn DROID — Multi-view per-task ratio ranking\n"]
    md.append(
        "Each of the 3 DROID camera views (`exterior_1`, `exterior_2`, `wrist`) "
        "is processed independently by the single-view WarpDyn pipeline "
        "(multi-lag cycle null + DINOv2 k-NN LOO null + within-view Cauchy "
        "fusion). The per-view fused H scores are then cross-view fused via a "
        "second Cauchy combine to produce **multiview_fused_peak**, the "
        "primary decision metric.\n"
    )

    # 1. Per-task training baselines
    md.append("## Per-task training baselines\n")
    hdr = ["Task"]
    for v in views:
        hdr += [f"H_cycle_{v}", f"H_knn_{v}", f"H_fused_{v}"]
    hdr += ["H_multiview"]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for task in sorted(h_train):
        cells = [task]
        for v in views:
            cells.append(f"{h_train[task].get(f'cycle_{v}_peak', 0):.4f}")
            cells.append(f"{h_train[task].get(f'knn_{v}_peak', 0):.4f}")
            cells.append(f"{h_train[task].get(f'fused_{v}_peak', 0):.4f}")
        cells.append(f"{h_train[task].get('multiview_fused_peak', 0):.4f}")
        md.append("| " + " | ".join(cells) + " |")

    # 2. FPR check on real (must all be ≤ 1.0 by construction)
    md.append("\n## FPR check on real (must all be ≤ 1.0)\n")
    fpr_hdr = ["Task"]
    for v in views:
        fpr_hdr += [f"ratio_cycle_{v}", f"ratio_knn_{v}", f"ratio_fused_{v}"]
    fpr_hdr += ["ratio_multiview"]
    md.append("| " + " | ".join(fpr_hdr) + " |")
    md.append("|" + "|".join(["---"] * len(fpr_hdr)) + "|")
    reals = [r for r in out_rows if r["type"] == "REAL"]
    real_flagged: dict[str, int] = {k: 0 for k in fpr_hdr[1:]}
    for r in reals:
        cells = [r["task"]]
        for v in views:
            for sig in ["cycle", "knn", "fused"]:
                key = f"ratio_{sig}_{v}"
                val = r[key]
                cells.append(f"{val:.3f}" if not (isinstance(val, float) and math.isnan(val)) else "nan")
                if isinstance(val, float) and not math.isnan(val) and val > 1.0:
                    real_flagged[key] += 1
        mv = r["ratio_multiview"]
        cells.append(f"{mv:.3f}" if not (isinstance(mv, float) and math.isnan(mv)) else "nan")
        if isinstance(mv, float) and not math.isnan(mv) and mv > 1.0:
            real_flagged["ratio_multiview"] += 1
        md.append("| " + " | ".join(cells) + " |")

    md.append("\n**Real flag counts (must be 0/N):**\n")
    for k in fpr_hdr[1:]:
        md.append(f"- {k}: {real_flagged[k]}/{len(reals)}")

    # 3. View-wise gen catch matrix
    gens = [r for r in out_rows if r["type"] == "GEN"]
    md.append("\n## View-wise gen catch matrix (HALLU = ratio > 1.0)\n")
    md.append("Columns mark which signal/view caught each gen.\n")
    mat_hdr = ["Task", "Vid"] + [f"fused_{v}" for v in views] + ["multiview"]
    md.append("| " + " | ".join(mat_hdr) + " |")
    md.append("|" + "|".join(["---"] * len(mat_hdr)) + "|")

    def _mark(r: dict, key: str) -> str:
        v = r[key]
        if isinstance(v, float) and math.isnan(v):
            return "—"
        return "HALLU" if v > 1.0 else ("borderline" if v > 0.95 else "·")

    for r in gens:
        cells = [r["task"], r["video"]]
        for v in views:
            cells.append(_mark(r, f"ratio_fused_{v}"))
        cells.append(_mark(r, "ratio_multiview"))
        md.append("| " + " | ".join(cells) + " |")

    # 4. Cross-view complementarity counts
    md.append("\n## Cross-view complementarity (gens caught per view-set)\n")
    catch_sets: dict[frozenset, int] = {}
    for r in gens:
        flags = frozenset(
            v for v in views
            if not (isinstance(r[f"ratio_fused_{v}"], float)
                    and math.isnan(r[f"ratio_fused_{v}"]))
            and r[f"ratio_fused_{v}"] > 1.0
        )
        catch_sets[flags] = catch_sets.get(flags, 0) + 1
    md.append("| Catching views (fused_<v>>1.0) | # gens |")
    md.append("|---|---|")
    for fs, ct in sorted(catch_sets.items(), key=lambda kv: (-len(kv[0]), sorted(kv[0]))):
        label = "{" + ", ".join(sorted(fs)) + "}" if fs else "∅ (none)"
        md.append(f"| {label} | {ct} |")

    # 5. Cycle/kNN/View 4-modality complementarity counts
    md.append("\n## Cycle / k-NN / view complementarity (per-modality catch sets)\n")
    md.append("Modality = `<signal>_<view>` (cycle or knn per view). Count gens by "
              "exactly which modalities caught them.\n")
    mod_catch_sets: dict[frozenset, int] = {}
    for r in gens:
        mods = []
        for v in views:
            for sig in ["cycle", "knn"]:
                val = r[f"ratio_{sig}_{v}"]
                if not (isinstance(val, float) and math.isnan(val)) and val > 1.0:
                    mods.append(f"{sig}_{v}")
        mod_catch_sets[frozenset(mods)] = mod_catch_sets.get(frozenset(mods), 0) + 1
    md.append("| Catching modalities | # gens |")
    md.append("|---|---|")
    for fs, ct in sorted(mod_catch_sets.items(), key=lambda kv: (-len(kv[0]), sorted(kv[0]))):
        label = "{" + ", ".join(sorted(fs)) + "}" if fs else "∅ (none)"
        md.append(f"| {label} | {ct} |")

    # 6. Verdict summary
    md.append("\n## Verdict summary (gens)\n")
    md.append("| Signal | HALLU/N |")
    md.append("|---|---|")
    for v in views:
        for sig in ["cycle", "knn", "fused"]:
            n_hal = sum(
                1 for r in gens
                if not (isinstance(r[f"ratio_{sig}_{v}"], float)
                        and math.isnan(r[f"ratio_{sig}_{v}"]))
                and r[f"ratio_{sig}_{v}"] > 1.0
            )
            md.append(f"| {sig}_{v} | {n_hal}/{len(gens)} |")
    n_mv = sum(
        1 for r in gens
        if not (isinstance(r["ratio_multiview"], float) and math.isnan(r["ratio_multiview"]))
        and r["ratio_multiview"] > 1.0
    )
    md.append(f"| **multiview (decision)** | **{n_mv}/{len(gens)}** |")

    # 7. Full gen ranking
    md.append("\n## Full gen ranking (sorted by ratio_multiview)\n")
    emoji = {"HALLU": "🔴", "borderline": "⚠", "clean": "✓", "unknown": "?"}
    rk_hdr = ["Rank", "Task", "Vid", "ratio_multiview"]
    for v in views:
        rk_hdr.append(f"ratio_fused_{v}")
    rk_hdr.append("verdict_multiview")
    md.append("| " + " | ".join(rk_hdr) + " |")
    md.append("|" + "|".join(["---"] * len(rk_hdr)) + "|")
    for i, r in enumerate(gens, 1):
        rmv = r["ratio_multiview"]
        rmv_s = f"**{rmv:.3f}**" if not (isinstance(rmv, float) and math.isnan(rmv)) else "nan"
        cells = [str(i), r["task"], r["video"], rmv_s]
        for v in views:
            val = r[f"ratio_fused_{v}"]
            cells.append(f"{val:.3f}" if not (isinstance(val, float) and math.isnan(val)) else "nan")
        verd = r["verdict_multiview"]
        cells.append(f"{emoji.get(verd, '?')} {verd}")
        md.append("| " + " | ".join(cells) + " |")

    out_md = args.eval_dir / f"per_task_ratio_ranking{args.out_suffix}.md"
    out_md.write_text("\n".join(md))
    print(f"Markdown → {out_md}")

    # ── Stdout summary ──────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"Reals: {len(reals)}   Gens: {len(gens)}")
    print(f"Real flagged (multiview): {real_flagged['ratio_multiview']}/{len(reals)}")
    print(f"Gen HALLU (multiview):    {n_mv}/{len(gens)}")


if __name__ == "__main__":
    main()
