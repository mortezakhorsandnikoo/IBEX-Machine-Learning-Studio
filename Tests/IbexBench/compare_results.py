"""Merges scikit-learn results with results_ibex.csv and writes paper-ready markdown tables.

Usage:  python compare_results.py <dir with results_sklearn_*.csv> results_ibex.csv  [out.md]
Needs:  pandas
"""
import sys, os
import pandas as pd

sk_dir = sys.argv[1] if len(sys.argv) > 1 else "."
ibex_path = sys.argv[2] if len(sys.argv) > 2 else "results_ibex.csv"
out_path = sys.argv[3] if len(sys.argv) > 3 else "comparison_tables.md"

ib = pd.read_csv(ibex_path)
lines = []

def pivot_cell(sk, ibx, task, dataset, algo, skcol):
    s = sk[(sk.dataset == dataset) & (sk.algorithm == algo)][skcol]
    skv = float(s.iloc[0]) if len(s) else float("nan")
    rows = ibx[(ibx.task == task) & (ibx.dataset == dataset) & (ibx.algorithm == algo)]
    return skv, rows

def fmt(x):
    return "n/a" if pd.isna(x) else f"{x:.3f}"

def supervised(task, sk_file, skcol, title, unit):
    sk = pd.read_csv(os.path.join(sk_dir, sk_file))
    datasets = list(dict.fromkeys(sk.dataset))
    algos = list(dict.fromkeys(list(sk.algorithm) + list(ib[ib.task == task].algorithm)))
    lines.append(f"\n### {title}\n")
    lines.append("| Algorithm | Dataset | scikit-learn | IBEX (config) | abs. difference | IBEX status |")
    lines.append("|---|---|---:|---|---:|---|")
    diffs = []
    for a in algos:
        for d in datasets:
            skv, rows = pivot_cell(sk, ib, task, d, a, skcol)
            if rows.empty:
                continue
            cells, dcells, stat = [], [], []
            for _, r in rows.iterrows():
                cells.append(f"{fmt(r.score_mean)} ({r.config})" if len(rows) > 1 else fmt(r.score_mean))
                dv = abs(r.score_mean - skv) if pd.notna(r.score_mean) and pd.notna(skv) else float("nan")
                dcells.append(fmt(dv)); stat.append("OK" if str(r.status) == "OK" else str(r.status)[:40])
                if pd.notna(dv): diffs.append((a, d, r.config, dv))
            lines.append(f"| {a} | {d} | {fmt(skv)} | {'; '.join(cells)} | {'; '.join(dcells)} | {'; '.join(stat)} |")
    if diffs:
        worst = max(diffs, key=lambda t: t[3])
        lines.append(f"\nLargest absolute difference ({unit}): {worst[3]:.3f} for {worst[0]} on {worst[1]} ({worst[2]}).")

supervised("regression", "results_sklearn_regression.csv", "R2_mean", "Regression: mean cross-validated R2", "R2")
supervised("classification", "results_sklearn_classification.csv", "Acc_mean", "Classification: mean cross-validated accuracy", "accuracy")

# clustering
sk = pd.read_csv(os.path.join(sk_dir, "results_sklearn_clustering.csv"))
lines.append("\n### Clustering\n")
lines.append("| Dataset | Algorithm | sk silhouette | IBEX silhouette | sk ARI | IBEX ARI | sk clusters | IBEX clusters |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
for _, r in sk.iterrows():
    m = ib[(ib.task == "clustering") & (ib.dataset == r.dataset) & (ib.algorithm == r.algorithm)]
    if m.empty:
        continue
    i = m.iloc[0]
    lines.append(f"| {r.dataset} | {r.algorithm} | {fmt(r.silhouette)} | {fmt(i.silhouette)} | {fmt(r.ARI)} | {fmt(i.ari)} | {int(r.n_clusters)} | {int(i.n_clusters)} |")
only = ib[(ib.task == "clustering") & (~ib.algorithm.isin(sk.algorithm))]
for _, i in only.iterrows():
    lines.append(f"| {i.dataset} | {i.algorithm} | no equivalent | {fmt(i.silhouette)} | no equivalent | {fmt(i.ari)} | no equivalent | {int(i.n_clusters)} |")

# coverage + time
lines.append("\n### Functional coverage\n")
ok = ib[ib.status == "OK"]
lines.append(f"{len(ok)} of {len(ib)} runs completed (train, cross-validate, predict).")
for t, g in ib.groupby("task"):
    algos = g.algorithm.nunique(); okalgos = g[g.status == "OK"].groupby("algorithm").size()
    lines.append(f"- {t}: {len(g[g.status == 'OK'])}/{len(g)} runs OK; {algos} distinct algorithms.")
bad = ib[ib.status != "OK"]
if len(bad):
    lines.append("\nFailed runs:\n")
    for _, r in bad.iterrows():
        lines.append(f"- {r.task} / {r.dataset} / {r.algorithm} ({r.config}): {r.status}")

lines.append("\n### IBEX run time (seconds for 5-fold CV + final fit)\n")
t = ib[ib.status == "OK"].groupby(["task", "algorithm"]).seconds.sum().round(2)
lines.append(t.to_frame("seconds").to_markdown())

open(out_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("wrote", out_path)
