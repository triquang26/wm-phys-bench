"""Post-hoc analysis of all signals from summary.csv.

Computes AUROC for each raw signal and multiple fusion combinations without
re-running detection. Requires:
  - summary.csv   (with raw_ivar_maha, raw_peak_maha, raw_ivar_px, raw_evidence)
  - labels.csv    (task, frame, split, label)
  - calibration.npz  (TaskCalibration distributions for p-value computation)

Usage:
    python analyze_signals.py [--summary summary.csv] [--labels labels.csv]
                               [--calib calibration.npz]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import rankdata


# ── Helpers ──────────────────────────────────────────────────────────────────

def empirical_p_high(value: float, dist_sorted: np.ndarray) -> float:
    n = len(dist_sorted)
    if n == 0:
        return 1.0
    k = int(n - np.searchsorted(dist_sorted, value, side="left"))
    return (1 + min(k, n - 1)) / (n + 1)


def cauchy_fuse(p_values: list[float]) -> float:
    eps = 1e-12
    ps = np.clip(np.array(p_values), eps, 1.0 - eps)
    k = len(ps)
    ws = np.full(k, 1.0 / k)
    T = float(np.sum(ws * np.tan(np.pi * (0.5 - ps))))
    return float(np.clip(0.5 - np.arctan(T) / np.pi, eps, 1.0))


def max_fuse(p_values: list[float]) -> float:
    return float(min(p_values))


# ── Load data ─────────────────────────────────────────────────────────────────

def load_labels(path: Path) -> dict[tuple, int]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["task"], row.get("split", ""), row["frame"])
            out[key] = int(row["label"])
    return out


def load_summary(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_calib(path: Path) -> dict:
    """Load calibration distributions keyed by task name."""
    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        meta = json.load(f)
    npz = np.load(path)
    calib = {}
    for task_name, task_meta in meta["tasks"].items():
        slug = task_meta["slug"]
        tc = {"task": task_name}
        for sig in ["ivar_maha", "evidence", "peak_maha", "ivar"]:
            key = f"{slug}__{sig}"
            if key in npz.files:
                tc[f"{sig}_dist"] = npz[key]
        calib[task_name] = tc
    # Global
    global_tc = {}
    for sig in ["ivar_maha", "evidence", "peak_maha", "ivar"]:
        key = f"__global__{sig}"
        if key in npz.files:
            global_tc[f"{sig}_dist"] = npz[key]
    calib["__global__"] = global_tc
    return calib


# ── AUROC helpers ─────────────────────────────────────────────────────────────

def auroc(y, scores) -> float:
    y = np.array(y)
    s = np.array(scores)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def ap(y, scores) -> float:
    y = np.array(y)
    s = np.array(scores)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    summary_path = Path(args.summary)
    labels_path = Path(args.labels)
    calib_path = Path(args.calib)

    labels = load_labels(labels_path)
    rows = load_summary(summary_path)
    calib = load_calib(calib_path)

    # Align labels with summary rows
    matched: list[tuple[tuple, dict, int]] = []
    for row in rows:
        key = (row["task"], row.get("split", ""), row["frame"])
        if key in labels:
            matched.append((key, row, labels[key]))

    print(f"Matched {len(matched)}/{len(rows)} rows (labels: {len(labels)})")
    print(f"  pos={sum(l for _, _, l in matched)}, neg={sum(1-l for _, _, l in matched)}")

    if not matched:
        sys.exit("No overlap between labels and summary!")

    # Collect per-row data
    keys = [m[0] for m in matched]
    y = [m[2] for m in matched]

    def get_raw(field, fallback=None):
        vals = []
        for _, row, _ in matched:
            v = row.get(field, "")
            vals.append(float(v) if v != "" else fallback)
        return vals

    raw_ivar_maha = get_raw("raw_ivar_maha", 0.0)
    raw_evidence  = get_raw("raw_evidence",  0.0)
    raw_ivar_px   = get_raw("raw_ivar_px",   0.5)
    raw_peak_maha = get_raw("raw_peak_maha", 0.0)
    h_score_orig  = get_raw("H_score",       0.5)

    # Per-task p-value computation from calibration
    def compute_p_values(raw_vals, signal: str) -> list[float]:
        """Compute per-task empirical p-values for a signal."""
        ps = []
        for (task, split, frame), val in zip(keys, raw_vals):
            tc = calib.get(task, calib.get("__global__", {}))
            dist_key = f"{signal}_dist"
            dist = tc.get(dist_key, calib.get("__global__", {}).get(dist_key))
            if dist is None or len(dist) == 0:
                ps.append(1.0)
            else:
                ps.append(empirical_p_high(val, dist))
        return ps

    def compute_global_p_values(raw_vals, signal: str) -> list[float]:
        """Compute global empirical p-values (all tasks pooled) for a signal."""
        dist = calib.get("__global__", {}).get(f"{signal}_dist")
        if dist is None or len(dist) == 0:
            return [1.0] * len(raw_vals)
        return [empirical_p_high(v, dist) for v in raw_vals]

    p_ivar_maha = compute_p_values(raw_ivar_maha, "ivar_maha")
    p_evidence  = compute_p_values(raw_evidence,  "evidence")
    p_peak_maha = compute_p_values(raw_peak_maha, "peak_maha")

    p_global_ivar_maha = compute_global_p_values(raw_ivar_maha, "ivar_maha")
    p_global_peak_maha = compute_global_p_values(raw_peak_maha, "peak_maha")

    # Fusion combinations
    h_ivar_maha_only       = [1.0 - p for p in p_ivar_maha]
    h_peak_maha_only       = [1.0 - p for p in p_peak_maha]
    h_cauchy_ivar_evidence = [1.0 - cauchy_fuse([pi, pe]) for pi, pe in zip(p_ivar_maha, p_evidence)]
    h_cauchy_ivar_peak     = [1.0 - cauchy_fuse([pi, pp]) for pi, pp in zip(p_ivar_maha, p_peak_maha)]
    h_cauchy_all3          = [1.0 - cauchy_fuse([pi, pe, pp]) for pi, pe, pp in
                               zip(p_ivar_maha, p_evidence, p_peak_maha)]
    h_max_ivar_peak        = [max(a, b) for a, b in zip(h_ivar_maha_only, h_peak_maha_only)]

    # Rank-sum combinations — rank-normalize then add
    N = len(y)
    rank_ivar_maha = rankdata(raw_ivar_maha) / N
    rank_ivar_px   = rankdata(raw_ivar_px)   / N
    rank_evidence  = rankdata(raw_evidence)  / N
    rank_peak_maha = rankdata(raw_peak_maha) / N

    h_rank_sum_ivar_px      = (rank_ivar_maha + rank_ivar_px).tolist()
    h_rank_sum_ivar_neg_px  = (rank_ivar_maha - rank_ivar_px).tolist()
    h_rank_sum_ivar_peak    = (rank_ivar_maha + rank_peak_maha).tolist()
    h_rank_sum_all3         = (rank_ivar_maha + rank_peak_maha + rank_ivar_px).tolist()
    h_rank_w2ivar_peak      = (2.0 * rank_ivar_maha + rank_peak_maha).tolist()
    h_rank_ivar_w2peak      = (rank_ivar_maha + 2.0 * rank_peak_maha).tolist()

    # Global p-value normalization (cross-task pooled calibration)
    h_global_p_ivar  = [1.0 - p for p in p_global_ivar_maha]
    h_global_p_peak  = [1.0 - p for p in p_global_peak_maha]
    h_global_cauchy  = [1.0 - cauchy_fuse([pi, pp]) for pi, pp in zip(p_global_ivar_maha, p_global_peak_maha)]

    # ── Task-adaptive signal routing ──────────────────────────────────────────
    # Route to peak_maha when test distribution is below null (domain shift indicator)
    # Route to ivar_maha otherwise
    task_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (task, split, frame) in enumerate(keys):
        task_to_indices[task].append(i)

    task_to_ivar_vals: dict[str, list[float]] = defaultdict(list)
    task_to_peak_vals: dict[str, list[float]] = defaultdict(list)
    for (task, split, frame), vi, vp in zip(keys, raw_ivar_maha, raw_peak_maha):
        task_to_ivar_vals[task].append(vi)
        task_to_peak_vals[task].append(vp)

    def get_null_mean(task: str, signal: str) -> float | None:
        tc = calib.get(task, calib.get("__global__", {}))
        dist = tc.get(f"{signal}_dist")
        if dist is None or len(dist) == 0:
            return None
        return float(dist.mean())

    # Determine per-task routing: ivar_maha or peak_maha
    # CV-based routing: high CV(ivar_maha) → ivar has good relative spread → use ivar
    # Low CV → ivar is flat / uninformative for this task → use peak_maha
    CV_THRESHOLD = 0.50
    task_routing: dict[str, str] = {}
    print("\n=== Per-task signal routing diagnostics ===")
    for task in sorted(task_to_indices.keys()):
        null_mean_ivar = get_null_mean(task, "ivar_maha")
        test_mean_ivar = float(np.mean(task_to_ivar_vals[task]))
        test_std_ivar  = float(np.std(task_to_ivar_vals[task]))
        null_mean_peak = get_null_mean(task, "peak_maha")
        test_mean_peak = float(np.mean(task_to_peak_vals[task]))
        ratio = (test_mean_ivar / null_mean_ivar) if null_mean_ivar else 1.0
        cv_ivar = (test_std_ivar / test_mean_ivar) if test_mean_ivar > 1e-8 else 0.0
        # CV-based routing: if coefficient of variation of ivar_maha is HIGH → use ivar
        # (large relative spread suggests discriminative power regardless of domain shift)
        use_ivar = cv_ivar >= CV_THRESHOLD
        task_routing[task] = "ivar_maha" if use_ivar else "peak_maha"
        tag = "→ ivar_maha [high CV]" if use_ivar else "→ peak_maha [low CV]"
        print(f"  {task[:55]:55s}  ivar_ratio={ratio:.2f}  CV={cv_ivar:.3f}  {tag}")

    h_adaptive_raw = []
    for (task, split, frame), vi, vp in zip(keys, raw_ivar_maha, raw_peak_maha):
        h_adaptive_raw.append(vp if task_routing[task] == "peak_maha" else vi)

    # Per-task rank normalize → then route
    def per_task_rank(vals: list[float]) -> list[float]:
        out = [0.0] * len(vals)
        for task, idxs in task_to_indices.items():
            sub = [vals[i] for i in idxs]
            ranks = (rankdata(sub) / len(sub)).tolist()
            for j, i in enumerate(idxs):
                out[i] = ranks[j]
        return out

    pt_rank_ivar = per_task_rank(raw_ivar_maha)
    pt_rank_peak = per_task_rank(raw_peak_maha)

    h_adaptive_pt_rank = []
    for (task, split, frame), ri, rp in zip(keys, pt_rank_ivar, pt_rank_peak):
        h_adaptive_pt_rank.append(rp if task_routing[task] == "peak_maha" else ri)

    # 3-way routing: very high CV → rank_sum(ivar+peak); mid CV → ivar; low CV → peak
    # Normalise rank sum by /2 to stay in [0,1] for cross-task comparability
    CV_HIGH = 0.70
    h_adaptive_3way = []
    for (task, split, frame), ri, rp in zip(keys, pt_rank_ivar, pt_rank_peak):
        cv = float(np.std(task_to_ivar_vals[task])) / max(float(np.mean(task_to_ivar_vals[task])), 1e-8)
        if cv >= CV_HIGH:
            h_adaptive_3way.append((ri + rp) / 2.0)
        elif cv >= CV_THRESHOLD:
            h_adaptive_3way.append(ri)
        else:
            h_adaptive_3way.append(rp)

    # Max of within-task ranks (both directions contribute)
    h_pt_rank_max = [max(ri, rp) for ri, rp in zip(pt_rank_ivar, pt_rank_peak)]

    # Oracle: per-task, per-signal AUROC → choose best signal per task
    # (upper bound — uses labels, not usable at inference time)
    task_oracle_signal: dict[str, str] = {}
    for task, idxs in task_to_indices.items():
        yt = [y[i] for i in idxs]
        vi = [raw_ivar_maha[i] for i in idxs]
        vp = [raw_peak_maha[i] for i in idxs]
        ai = auroc(yt, vi)
        ap_ = auroc(yt, vp)
        if np.isnan(ai):
            ai = 0.5
        if np.isnan(ap_):
            ap_ = 0.5
        task_oracle_signal[task] = "peak_maha" if ap_ > ai else "ivar_maha"

    h_oracle = []
    for (task, split, frame), vi, vp in zip(keys, raw_ivar_maha, raw_peak_maha):
        h_oracle.append(vp if task_oracle_signal[task] == "peak_maha" else vi)

    h_oracle_pt = []
    for (task, split, frame), ri, rp in zip(keys, pt_rank_ivar, pt_rank_peak):
        h_oracle_pt.append(rp if task_oracle_signal[task] == "peak_maha" else ri)

    # Calibration-based two-sided test: score = how far from null mean (either direction)
    def two_sided_h(raw_vals: list[float], signal: str) -> list[float]:
        """1 - min(p_high, p_low) — fires for anomalies in either direction."""
        out = []
        for (task, split, frame), val in zip(keys, raw_vals):
            tc = calib.get(task, calib.get("__global__", {}))
            dist = tc.get(f"{signal}_dist", calib.get("__global__", {}).get(f"{signal}_dist"))
            if dist is None or len(dist) == 0:
                out.append(0.0)
                continue
            p_high = empirical_p_high(val, dist)
            p_low = 1.0 - p_high
            p_two = 2 * min(p_high, p_low)
            out.append(1.0 - p_two)
        return out

    h_two_sided_ivar = two_sided_h(raw_ivar_maha, "ivar_maha")
    h_two_sided_peak = two_sided_h(raw_peak_maha, "peak_maha")
    h_two_sided_cauchy = [1.0 - cauchy_fuse([1.0 - a, 1.0 - b])
                          for a, b in zip(h_two_sided_ivar, h_two_sided_peak)]

    signals = {
        "raw_ivar_maha":                    raw_ivar_maha,
        "raw_ivar_px":                      raw_ivar_px,
        "raw_peak_maha":                    raw_peak_maha,
        "raw_evidence":                     raw_evidence,
        "H_score_orig (Cauchy ivar+ev)":    h_score_orig,
        "H: 1-p(ivar_maha)":               h_ivar_maha_only,
        "H: 1-p(peak_maha)":               h_peak_maha_only,
        "H: Cauchy(ivar+evidence)":         h_cauchy_ivar_evidence,
        "H: Cauchy(ivar+peak_maha)":        h_cauchy_ivar_peak,
        "H: Cauchy(ivar+ev+peak)":          h_cauchy_all3,
        "H: max(1-p_ivar, 1-p_peak)":       h_max_ivar_peak,
        "H: global_p ivar_maha":            h_global_p_ivar,
        "H: global_p peak_maha":            h_global_p_peak,
        "H: global Cauchy(ivar+peak)":      h_global_cauchy,
        "H: two-sided(ivar_maha)":          h_two_sided_ivar,
        "H: two-sided(peak_maha)":          h_two_sided_peak,
        "rank: ivar_maha + ivar_px":        h_rank_sum_ivar_px,
        "rank: ivar_maha - ivar_px":        h_rank_sum_ivar_neg_px,
        "rank: ivar_maha + peak_maha":      h_rank_sum_ivar_peak,
        "rank: ivar + peak + ivar_px":      h_rank_sum_all3,
        "rank: 2*ivar + peak":              h_rank_w2ivar_peak,
        "rank: ivar + 2*peak":              h_rank_ivar_w2peak,
        "pt_rank: max(ivar, peak)":         h_pt_rank_max,
        "adaptive: raw (calib-routed)":     h_adaptive_raw,
        "adaptive: pt_rank (calib-routed)": h_adaptive_pt_rank,
        "adaptive: pt_rank 3way":           h_adaptive_3way,
        "ORACLE: raw (labels)":             h_oracle,
        "ORACLE: pt_rank (labels)":         h_oracle_pt,
    }

    print("\n=== Overall AUROC / AP ===")
    results = []
    for name, scores in signals.items():
        a = auroc(y, scores)
        p = ap(y, scores)
        results.append((name, a, p))
    results.sort(key=lambda x: -x[1])
    for name, a, p_val in results:
        tag = " ← BEST" if a == max(r[1] for r in results if not np.isnan(r[1])) else ""
        tag2 = " [ORACLE]" if "ORACLE" in name else ""
        print(f"  {name:42s}  AUROC={a:.4f}  AP={p_val:.4f}{tag}{tag2}")

    # Per-task breakdown for the top 4 non-oracle signals + oracle
    non_oracle = [r for r in results if "ORACLE" not in r[0]]
    oracle_results = [r for r in results if "ORACLE" in r[0]]
    top4 = [r[0] for r in non_oracle[:4]]
    show = top4 + [r[0] for r in oracle_results[:1]]

    print("\n=== Per-task AUROC for top signals ===")
    task_data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for i, (task, split, frame) in enumerate(keys):
        task_data[task]["y"].append(y[i])
        for name in show:
            task_data[task][name].append(signals[name][i])

    for task in sorted(task_data.keys()):
        d = task_data[task]
        yt = np.array(d["y"])
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        parts = [f"n={len(yt)}({yt.sum()}+)"]
        for name in show:
            a = auroc(yt, d[name])
            # Short label
            if "ORACLE" in name:
                lbl = "ORACLE"
            elif "adaptive" in name:
                lbl = f"adapt({task_routing.get(task, '?')[:4]})"
            else:
                lbl = name.split("(")[-1].rstrip(")") if "(" in name else name.split("_", 1)[-1][:15]
            parts.append(f"{lbl}={a:.3f}")
        print(f"  {task[:55]:55s}  " + "  ".join(parts))

    # Best non-oracle signal recommendation
    best = non_oracle[0]
    print(f"\n=> Best signal (non-oracle): '{best[0]}'  AUROC={best[1]:.4f}  AP={best[2]:.4f}")
    if oracle_results:
        orb = oracle_results[0]
        print(f"=> Oracle upper bound:       '{orb[0]}'  AUROC={orb[1]:.4f}  AP={orb[2]:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--summary",
                   default="results/summary.csv")
    p.add_argument("--labels",
                   default="/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/"
                           "feature_matching_eval_hallucination/test/results/labels.csv")
    p.add_argument("--calib",
                   default="results/calibration.npz")
    main(p.parse_args())
