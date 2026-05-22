"""Aggregate 23-task doanh eval results + render comparison plots."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
DEMO = REPO / ".claude/worktrees/feat-knn-pool-gr1/outputs/doanh_eval_demo"
BENCH = REPO / "paper-doanh-eval"
EVAL_TASKS = json.loads((BENCH / "eval_tasks.json").read_text())

rows = []
for ts, task in sorted(EVAL_TASKS.items(), key=lambda x: int(x[0])):
    folder = f"{ts}_{task['task_full']}"[:200]
    t_json = DEMO / folder / "timing.json"
    if not t_json.exists():
        continue
    t = json.loads(t_json.read_text())
    o = t["online"]
    bn = t.get("baseline_normalizer", {})
    rows.append({
        "ts": int(ts),
        "task_full": task["task_full"],
        "eval_subfolder": task.get("eval_subfolder", ""),
        "ratio_cycle": o["ratio_cycle"],
        "ratio_knn": o["ratio_knn"],
        "ratio_fused": o["ratio_fused"],
        "score_norm": o.get("score_norm", float("nan")),
        "sigma": bn.get("sigma", float("nan")),
        "alpha": bn.get("alpha", float("nan")),
    })

if not rows:
    print("No completed tasks found")
    exit(0)

# Write summary CSV
out_csv = BENCH / "per_task_dense_eval" / "doanh_low_vs_high_ratio.csv"
out_csv.parent.mkdir(exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"→ {out_csv}")

# Sort by ratio_fused descending
rows_sorted = sorted(rows, key=lambda r: -r["ratio_fused"])

# Plot 1: Per-task ratio bars (cycle / knn / fused)
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(rows_sorted))
w = 0.27
labels = [f"{r['ts']}: {r['task_full'][:35]}" for r in rows_sorted]
ax.bar(x - w, [r["ratio_cycle"] for r in rows_sorted], w, label="ratio_cycle", color="steelblue", alpha=0.85)
ax.bar(x,     [r["ratio_knn"]   for r in rows_sorted], w, label="ratio_knn",   color="darkorange", alpha=0.85)
ax.bar(x + w, [r["ratio_fused"] for r in rows_sorted], w, label="ratio_fused", color="crimson", alpha=0.9)
ax.axhline(1.0, color="black", ls="--", lw=1, label="HALLU threshold")
ax.axhline(0.95, color="gray", ls=":", lw=1, label="borderline lower")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=8)
ax.set_ylabel("ratio (= H_low / H_train_high)")
ax.set_title(f"doanh high → low quality detection (n={len(rows_sorted)} tasks, sorted by ratio_fused)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out_bar = BENCH / "per_task_dense_eval" / "doanh_ratio_bars.png"
fig.savefig(out_bar, dpi=130)
plt.close(fig)
print(f"→ {out_bar}")

# Plot 1b: Normalized score bars (sigmoid-normalized) — sorted by score_norm
rows_sorted_norm = sorted(rows, key=lambda r: -r["score_norm"])
fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(rows_sorted_norm))
labels_n = [f"{r['ts']}: {r['task_full'][:35]}" for r in rows_sorted_norm]
colors = ["crimson" if r["score_norm"] > 0.5 else "steelblue" for r in rows_sorted_norm]
ax.bar(x, [r["score_norm"] for r in rows_sorted_norm], color=colors, alpha=0.85, edgecolor="white")
ax.axhline(0.5, color="black", ls="--", lw=1.5, label="baseline (ratio = 1.0)")
ax.axhline(0.73, color="orange", ls=":", lw=1, label="+1σ_baseline (weak HALLU)")
ax.axhline(0.88, color="red", ls=":", lw=1, label="+2σ_baseline (strong HALLU)")
ax.axhline(0.27, color="green", ls=":", lw=1, label="-1σ_baseline (clean)")
ax.set_xticks(x)
ax.set_xticklabels(labels_n, rotation=80, ha="right", fontsize=8)
ax.set_ylabel("score_norm = sigmoid(α · (ratio_fused − 1))")
ax.set_title(f"doanh — sigmoid-normalized fused score (n={len(rows_sorted_norm)} tasks)")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out_norm = BENCH / "per_task_dense_eval" / "doanh_score_norm_bars.png"
fig.savefig(out_norm, dpi=130)
plt.close(fig)
print(f"→ {out_norm}")

# Plot 1c: ratio_fused vs score_norm scatter — sanity check
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter([r["ratio_fused"] for r in rows], [r["score_norm"] for r in rows],
                 c=[r["sigma"] for r in rows], cmap="viridis", s=80, edgecolor="white", linewidth=0.5)
plt.colorbar(sc, ax=ax, label="bootstrap σ_baseline (per task)")
ax.axhline(0.5, color="black", ls="--", lw=1)
ax.axvline(1.0, color="black", ls="--", lw=1)
for r in rows:
    ax.annotate(str(r["ts"]), (r["ratio_fused"], r["score_norm"]),
                fontsize=7, alpha=0.7, ha="center", va="center")
ax.set_xlabel("ratio_fused = H_test / H_train")
ax.set_ylabel("score_norm = sigmoid(α · (ratio - 1))")
ax.set_title("ratio → normalized score mapping (color = task's bootstrap σ)")
ax.grid(alpha=0.3)
fig.tight_layout()
out_scatter = BENCH / "per_task_dense_eval" / "doanh_ratio_vs_score_norm.png"
fig.savefig(out_scatter, dpi=130)
plt.close(fig)
print(f"→ {out_scatter}")

# Plot 2: Distribution histogram
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, key, color in zip(axes, ["ratio_cycle", "ratio_knn", "ratio_fused"],
                           ["steelblue", "darkorange", "crimson"]):
    vals = [r[key] for r in rows]
    ax.hist(vals, bins=15, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="black", ls="--", lw=1, label="HALLU thr")
    ax.axvline(np.mean(vals), color="green", ls=":", lw=1, label=f"mean={np.mean(vals):.2f}")
    ax.set_xlabel(key)
    ax.set_ylabel("count")
    ax.set_title(f"{key} (n={len(vals)})")
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle("Distribution of ratios across 23 doanh tasks (low quality scored against high quality ref)")
fig.tight_layout()
out_hist = BENCH / "per_task_dense_eval" / "doanh_ratio_histograms.png"
fig.savefig(out_hist, dpi=130)
plt.close(fig)
print(f"→ {out_hist}")

# Print summary
import statistics
print(f"\n=== Summary ({len(rows)} tasks) ===")
for k in ("ratio_cycle", "ratio_knn", "ratio_fused"):
    vals = [r[k] for r in rows]
    n_hallu = sum(1 for v in vals if v > 1.0)
    print(f"  {k}: mean={statistics.mean(vals):.3f}, median={statistics.median(vals):.3f}, HALLU={n_hallu}/{len(vals)}")
sigmas = [r["sigma"] for r in rows]
scores = [r["score_norm"] for r in rows]
n_hallu_n = sum(1 for s in scores if s > 0.73)
n_border_n = sum(1 for s in scores if 0.5 < s <= 0.73)
print(f"  score_norm: mean={statistics.mean(scores):.3f}, median={statistics.median(scores):.3f}, "
      f"HALLU(>0.73)={n_hallu_n}/{len(scores)}, border(0.5-0.73)={n_border_n}/{len(scores)}")
print(f"  σ_baseline: mean={statistics.mean(sigmas):.4f}, range=[{min(sigmas):.4f}, {max(sigmas):.4f}]")

# Markdown table
md = ["# doanh high (training) → low (query) eval — 23 tasks\n"]
md.append("Pipeline: doanh `high/<task>.mp4` → SAM3 refs (50 frames) → cycle null + kNN LOO null.")
md.append("Then score doanh `low/<task>.mp4` as query. ratio = H_test_low / H_train_high.\n")
md.append("→ ratio > 1.0 means low quality version is MORE anomalous than high quality (expected).\n")
md.append("Sigmoid normalization: per-task σ_baseline from bootstrap (200 iter) of H_train_fused.")
md.append("`score_norm = sigmoid((ratio - 1) / σ_baseline)`, mapped to [0,1] where 0.5 = baseline.\n")
md.append("## Per-task scores (sorted by score_norm desc)\n")
md.append("| Rank | task_short | Eval folder | task | ratio_cycle | ratio_knn | **ratio_fused** | σ_base | **score_norm** | verdict |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(rows_sorted_norm, 1):
    rc, rk, rf = r["ratio_cycle"], r["ratio_knn"], r["ratio_fused"]
    sn, sg = r["score_norm"], r["sigma"]
    v = "🔴 HALLU" if sn > 0.73 else ("⚠ borderline" if sn > 0.5 else "✓ clean")
    md.append(f"| {i} | {r['ts']} | {r['eval_subfolder']} | {r['task_full'][:50]} | {rc:.3f} | {rk:.3f} | **{rf:.3f}** | {sg:.4f} | **{sn:.3f}** | {v} |")
md.append("\n## Decision thresholds on score_norm\n")
md.append("- `> 0.88`: strongly anomalous (≥ +2σ from baseline)")
md.append("- `0.73 – 0.88`: weakly anomalous (≥ +1σ)")
md.append("- `0.50`: at baseline (ratio = 1.0)")
md.append("- `0.27 – 0.50`: weakly clean")
md.append("- `< 0.27`: strongly clean (≤ −1σ)")
md.append("")
md.append("\n## Summary (score_norm-based)\n")
n_hallu = sum(1 for r in rows if r["score_norm"] > 0.73)
n_border = sum(1 for r in rows if 0.5 < r["score_norm"] <= 0.73)
n_clean = sum(1 for r in rows if r["score_norm"] <= 0.5)
md.append(f"- 🔴 HALLU (>0.73): **{n_hallu}/{len(rows)}**")
md.append(f"- ⚠ borderline (0.5-0.73): **{n_border}/{len(rows)}**")
md.append(f"- ✓ clean (≤0.5): **{n_clean}/{len(rows)}**")
sigmas = [r["sigma"] for r in rows]
md.append(f"- σ_baseline across tasks: mean={np.mean(sigmas):.4f}, range=[{min(sigmas):.4f}, {max(sigmas):.4f}]")

out_md = BENCH / "per_task_dense_eval" / "doanh_summary.md"
out_md.write_text("\n".join(md))
print(f"→ {out_md}")
