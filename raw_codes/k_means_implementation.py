"""
Step 04 — Final K-means fit on the PhilGEPS PC space.

Reads ``output_source/03/Clustering/philgeps_clustering_pc_scores.csv`` (PC1, PC2, PC3 from
step 03) and the K chosen in step 05 (``output_source/05/KSelection/k_selection_summary.json``,
key ``chosen_k``). One full K-means is fit on all rows and the labels are persisted so step 06
can interpret each cluster.

Outputs:

    output_source/04/KMeans/
        philgeps_kmeans_assignments.csv     (row_index, cluster_id, PC1, PC2, PC3)
    output_source/04/Backtrack/
        philgeps_cluster_backtrack.csv      (row_index, cluster_id, PC1..PC3, theme + base cols)
    output_source/04/per_cluster/
        cluster_{cid}.csv                   (one CSV per cluster: rows from the backtrack frame)
    results/04/PCA_Cluster/
        pca_space_pc123_3d_kmeans_numeric.png  (``C0``, ``C1``, …; step 04 writes first pass; step 06 refreshes same subsample/jitter, still numeric legend only)
        pca_space_pc123_3d_kmeans_semantic.png (step 06; descriptive cluster labels in legend)
        pca_space_pc123_3d_kmeans_interactive.html / *_interactive_rows.json  (step 06; Plotly + wide-merge click fields)
    results/04/Summaries/
        cluster_counts.json
        run_meta.json
        kmeans_implementation_readme.txt
    logs/04/Log entries/04_kmeans_implementation_philgeps_activity.txt
    logs/04/Terminal Logs/04_kmeans_implementation_philgeps_terminal.txt

Run after ``05_evaluating_kmeans_philgeps.py`` (or pass ``--k`` to override). Step 06 will read
the assignments and backtrack written here.

Usage:
    python 04_kmeans_implementation_philgeps.py
    python 04_kmeans_implementation_philgeps.py --k 5             # override the K from step 05
    python 04_kmeans_implementation_philgeps.py --pc3d-max-points 0  # plot every row (slow)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import cupy as np
import cudf as pd
from cuml.cluster import KMeans

from philgeps_kmeans_common import (
    KMEANS_N_INIT,
    OUT_04_ASSIGNMENTS_CSV,
    OUT_04_BACKTRACK_CSV,
    OUT_04_CLUSTER_COUNTS_JSON,
    OUT_04_NUMERIC_PNG,
    OUT_04_README,
    OUT_04_RUN_META_JSON,
    OUT_K_SELECTION_SUMMARY,
    PATH_LOG_ENTRIES_04,
    PATH_LOG_TERMINAL_04,
    PATH_LOGS_04,
    PATH_OUT_04_BACKTRACK,
    PATH_OUT_04_KMEANS,
    PATH_OUT_04_PER_CLUSTER,
    PATH_RES_04_PCA_CLUSTER,
    PATH_RES_04_SUMMARIES,
    PC3D_MAX_POINTS_DEFAULT,
    PC3D_PLOT_JITTER_FRAC_DEFAULT,
    RANDOM_SEED,
    ensure_dirs,
    ensure_log_tree,
    load_backtrack_frame,
    load_chosen_k,
    load_pc_scores,
    load_pca_ratios3,
    open_activity_log,
    save_labeled_pc_scatter_3d,
    tee_stdio_to_file,
)


def _ensure_tree() -> None:
    ensure_dirs(
        PATH_OUT_04_KMEANS,
        PATH_OUT_04_BACKTRACK,
        PATH_OUT_04_PER_CLUSTER,
        PATH_RES_04_PCA_CLUSTER,
        PATH_RES_04_SUMMARIES,
    )
    ensure_log_tree(PATH_LOGS_04)


def _write_readme(*, k: int, n: int, jitter: float, max_points: int) -> None:
    body = f"""PhilGEPS step 04 — Final K-means fit on PC scores

Inputs:
  output_source/03/Clustering/philgeps_clustering_pc_scores.csv      (PC1..PC3 from step 03)
  output_source/05/KSelection/k_selection_summary.json                (chosen_k from step 05)
  output_source/02/Min-Max Scaling/philgeps_min_max_scaled.csv        (theme + base columns for backtrack)

Process:
  K = {k}; one fit on the full PC matrix (n={n:,}):
    KMeans(n_clusters={k}, random_state={RANDOM_SEED}, n_init={KMEANS_N_INIT}, algorithm='lloyd')

Row alignment:
  Step 03 writes PC scores in the same row order as the scaled numerics CSV (default RangeIndex).
  Step 04 joins backtrack columns by row position (no shuffling between steps). The
  ``row_index`` column persisted with the assignments is the RangeIndex from step 03.

Outputs:
  output_source/04/KMeans/philgeps_kmeans_assignments.csv  — row_index, cluster_id, PC1..PC3
  output_source/04/Backtrack/philgeps_cluster_backtrack.csv
                                                          — row_index, cluster_id, PC1..PC3,
                                                            POLICY_THEME_SCORE_COLUMNS,
                                                            POLICY_PCA_BASE_COLUMNS
  output_source/04/per_cluster/cluster_{{cid}}.csv         — one CSV per cluster (rows from backtrack)
  results/04/PCA_Cluster/pca_space_pc123_3d_kmeans_numeric.png
                                                          — numeric legend (C0, C1, …); step 04 writes first pass;
                                                            step 06 refreshes for aligned subsample/jitter (**still C0…Ck**, not semantic text).
  results/04/PCA_Cluster/pca_space_pc123_3d_kmeans_semantic.png
                                                          — step 06 only; same scatter settings as numeric but semantic legend.
  results/04/PCA_Cluster/pca_space_pc123_3d_kmeans_interactive.html (+ *_interactive_rows.json)
                                                          — written by step 06 (Plotly; requires plotly); click payload
                                                            from the wide merge (see step 06 readme).

Step 06 reads these outputs to assign semantic names and write
``pca_space_pc123_3d_kmeans_semantic.png``, refresh ``pca_space_pc123_3d_kmeans_numeric.png`` (numeric legend),
and interactive HTML when plotly is installed.
"""
    with open(OUT_04_README, "w", encoding="utf-8", newline="\n") as f:
        f.write(body.strip() + "\n")


def _write_per_cluster_csvs(df_bt: pd.DataFrame, *, log: Any) -> dict[int, str]:
    paths: dict[int, str] = {}
    for cid, sub in df_bt.groupby("cluster_id", sort=True):
        cid_int = int(cid)
        out = os.path.join(PATH_OUT_04_PER_CLUSTER, f"cluster_{cid_int}.csv")
        sub.to_csv(out, index=False)
        paths[cid_int] = out
        log(f"Wrote per-cluster CSV cluster_id={cid_int} (rows={len(sub):,}) -> {out}")
    return paths


def run_step04(
    *,
    k_override: int | None,
    pc3d_max_points: int,
    pc3d_plot_jitter_frac: float,
) -> None:
    _ensure_tree()
    activity_path = os.path.join(
        PATH_LOG_ENTRIES_04, "04_kmeans_implementation_philgeps_activity.txt",
    )
    log = open_activity_log(activity_path)

    if k_override is None:
        log(f"Reading chosen K from step 05 summary: {OUT_K_SELECTION_SUMMARY}")
        summary = load_chosen_k(OUT_K_SELECTION_SUMMARY)
        k = int(summary["chosen_k"])
        chosen_method = str(summary.get("chosen_k_method", "silhouette"))
        log(f"chosen_k={k} (method={chosen_method})")
    else:
        if k_override < 2:
            msg = f"--k must be >= 2 (got {k_override})"
            log(f"ERROR: {msg}")
            print(msg, file=sys.stderr)
            sys.exit(2)
        k = int(k_override)
        chosen_method = "manual_override"
        log(f"chosen_k={k} (manual override; ignoring step 05 summary)")

    log("Loading PC scores from step 03")
    X = load_pc_scores()
    n = int(X.shape[0])
    log(f"PC scores: shape={X.shape}")
    if k >= n:
        msg = f"K={k} cannot exceed n={n}"
        log(f"ERROR: {msg}")
        print(msg, file=sys.stderr)
        sys.exit(1)

    log(f"Fitting KMeans(K={k}) on full PC matrix")
    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=KMEANS_N_INIT,
        algorithm="lloyd",
    )
    labels = km.fit_predict(X)
    inertia = float(km.inertia_)
    centers = km.cluster_centers_.astype(float)
    log(f"Fit done. inertia={inertia:.4f}; centers shape={centers.shape}")

    counts = pd.Series(labels).value_counts().sort_index()
    log("Cluster sizes: " + ", ".join(f"C{int(c)}={int(v)}" for c, v in counts.items()))

    df_assign = pd.DataFrame({
        "row_index": np.arange(n, dtype=np.int64),
        "cluster_id": labels.astype(np.int64),
        "PC1": X[:, 0],
        "PC2": X[:, 1],
        "PC3": X[:, 2],
    })
    df_assign.to_csv(OUT_04_ASSIGNMENTS_CSV, index=False)
    log(f"Wrote assignments -> {OUT_04_ASSIGNMENTS_CSV} (shape={df_assign.shape})")

    log("Loading backtrack columns from step 02 scaled CSV")
    df_back = load_backtrack_frame()
    if len(df_back) != n:
        msg = (
            f"Row count mismatch: scaled CSV has {len(df_back):,} rows but PC scores have {n:,}. "
            "Step 03 must be regenerated against the same step-02 output."
        )
        log(f"ERROR: {msg}")
        print(msg, file=sys.stderr)
        sys.exit(1)
    df_back = df_back.reset_index(drop=True)
    df_back.insert(0, "row_index", np.arange(n, dtype=np.int64))
    df_back.insert(1, "cluster_id", labels.astype(np.int64))
    df_back.insert(2, "PC1", X[:, 0])
    df_back.insert(3, "PC2", X[:, 1])
    df_back.insert(4, "PC3", X[:, 2])
    df_back.to_csv(OUT_04_BACKTRACK_CSV, index=False)
    log(f"Wrote backtrack -> {OUT_04_BACKTRACK_CSV} (shape={df_back.shape})")

    per_cluster_paths = _write_per_cluster_csvs(df_back, log=log)

    log("Loading PCA variance ratios for plot axes")
    ratios3 = load_pca_ratios3()
    save_labeled_pc_scatter_3d(
        X,
        labels,
        OUT_04_NUMERIC_PNG,
        ratios3=ratios3,
        title_suffix=f"K={k}",
        max_points=pc3d_max_points,
        plot_jitter_frac=pc3d_plot_jitter_frac,
    )
    log(f"Saved numeric 3D K-means scatter -> {OUT_04_NUMERIC_PNG}")

    counts_payload: dict[str, Any] = {
        "k": k,
        "n_total": n,
        "cluster_counts": {str(int(c)): int(v) for c, v in counts.items()},
        "cluster_share": {str(int(c)): float(v) / float(n) for c, v in counts.items()},
        "inertia_full": inertia,
    }
    with open(OUT_04_CLUSTER_COUNTS_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(counts_payload, f, indent=2)
    log(f"Wrote cluster counts -> {OUT_04_CLUSTER_COUNTS_JSON}")

    run_meta: dict[str, Any] = {
        "k": k,
        "k_source": chosen_method,
        "n_total": n,
        "random_seed": RANDOM_SEED,
        "n_init": KMEANS_N_INIT,
        "algorithm": "lloyd",
        "centers_pc": centers.tolist(),
        "inertia_full": inertia,
        "pc3d_max_points": int(pc3d_max_points),
        "pc3d_plot_jitter_frac": float(pc3d_plot_jitter_frac),
        "paths": {
            "assignments_csv": OUT_04_ASSIGNMENTS_CSV,
            "backtrack_csv": OUT_04_BACKTRACK_CSV,
            "per_cluster_csv_dir": PATH_OUT_04_PER_CLUSTER,
            "per_cluster_csvs": {str(c): p for c, p in per_cluster_paths.items()},
            "numeric_3d_png": OUT_04_NUMERIC_PNG,
            "cluster_counts_json": OUT_04_CLUSTER_COUNTS_JSON,
            "k_selection_summary_json": OUT_K_SELECTION_SUMMARY,
            "logs_dir": PATH_LOGS_04,
        },
    }
    with open(OUT_04_RUN_META_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(run_meta, f, indent=2)
    log(f"Wrote run meta -> {OUT_04_RUN_META_JSON}")

    _write_readme(
        k=k, n=n, jitter=float(pc3d_plot_jitter_frac), max_points=int(pc3d_max_points),
    )
    log(f"Wrote readme -> {OUT_04_README}")

    print(
        "PhilGEPS step 04 done. "
        f"K={k}; assignments: {OUT_04_ASSIGNMENTS_CSV}; backtrack: {OUT_04_BACKTRACK_CSV}; "
        f"per-cluster: {PATH_OUT_04_PER_CLUSTER}; results: {PATH_RES_04_PCA_CLUSTER}; "
        f"summaries: {PATH_RES_04_SUMMARIES}; logs: {PATH_LOGS_04}. "
        "Run 06_cluster_interpretation_philgeps.py next.",
        flush=True,
    )
    log(f"Step 04 complete. K={k}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhilGEPS step 04 — Final K-means fit on the PC space (uses K from step 05).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override K (default: read chosen_k from step 05 summary).",
    )
    parser.add_argument(
        "--pc3d-max-points",
        type=int,
        default=PC3D_MAX_POINTS_DEFAULT,
        metavar="N",
        help=f"Max points in numeric 3D scatter (0 = all rows; default {PC3D_MAX_POINTS_DEFAULT}).",
    )
    parser.add_argument(
        "--pc3d-plot-jitter-frac",
        type=float,
        default=PC3D_PLOT_JITTER_FRAC_DEFAULT,
        metavar="F",
        help=(
            "Plot-only Gaussian jitter (matches step 03 solid PNG; default "
            f"{PC3D_PLOT_JITTER_FRAC_DEFAULT}; 0 = no jitter)."
        ),
    )
    args = parser.parse_args()
    jf = float(args.pc3d_plot_jitter_frac)
    if not math.isfinite(jf) or jf < 0.0:
        print("--pc3d-plot-jitter-frac must be a finite number >= 0.", file=sys.stderr)
        sys.exit(2)

    _ensure_tree()
    term_log = os.path.join(
        PATH_LOG_TERMINAL_04, "04_kmeans_implementation_philgeps_terminal.txt",
    )
    with tee_stdio_to_file(term_log):
        run_step04(
            k_override=args.k,
            pc3d_max_points=int(args.pc3d_max_points),
            pc3d_plot_jitter_frac=jf,
        )


if __name__ == "__main__":
    main()