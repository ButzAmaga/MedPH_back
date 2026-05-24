"""
Step 03 — PhilGEPS PCA dimensionality (theme track + diagnostics).

Reads step-02 Layer C numerics (``philgeps_min_max_scaled.csv``). Default PCA input is
``POLICY_PCA_BASE_COLUMNS`` (underlying feature names in loadings). Use ``--matrix theme`` for theme
aggregates only, or ``--matrix scaled`` for all numerics. PCA then writes:

    output_source/03/Clustering/
        philgeps_clustering_features.csv   (columns used as PCA input)
        philgeps_clustering_pc_scores.csv    (PC1, PC2, PC3)
    results/03/Clustering/
        pca_theme_clustering.json            (variance ratios, mean, components for 3 PCs)
        cumulative_variance_pca.png
        pca_loadings_pc123.csv / .png
        pca_space_pc123_3d.png             (3D scatter; default KNN density + optional jitter; --pc3d-color)
        pca_space_pc123_3d_solid.png       (same view, always single-color steelblue; no colorbar / density)
        pca_space_pc123_3d_interactive.html (Plotly; click -> full FS row via lazy-loaded JSON subsample)
        pca_space_pc123_3d_interactive_rows.json (subsample row dicts only; not embedded in HTML)
        pca_pc123_dominance_audit.json     (feature correlation + PC1–3 loading participation)
        clustering_features_readme.txt

Run after ``02_data_preprocessing_philgeps.py``.

Usage:
    python 03_PCA_Dimensionality_philgeps.py                    # base numerics (default; loadings use feature names)
    python 03_PCA_Dimensionality_philgeps.py --matrix theme    # theme clustering subset (theme-named columns)
    python 03_PCA_Dimensionality_philgeps.py --matrix scaled   # all scaled numerics
    python 03_PCA_Dimensionality_philgeps.py --matrix scaled --rows 50000 --max-pc 9
    python 03_PCA_Dimensionality_philgeps.py --pc3d-max-points 0   # 3D plot uses every row (may be slow)
    python 03_PCA_Dimensionality_philgeps.py --no-standardize      # legacy: PCA on min–max without z-scoring
    python 03_PCA_Dimensionality_philgeps.py --matrix scaled --scaled-include-policy-themes
    python 03_PCA_Dimensionality_philgeps.py --pc3d-color solid          # legacy single-color 3D PNG
    python 03_PCA_Dimensionality_philgeps.py --pc3d-plot-jitter-frac 0  # no jitter (exact subsample coords on PNG)
    python 03_PCA_Dimensionality_philgeps.py --no-pc3d-interactive-html # skip Plotly HTML

v5 parity (Phase A). By default, PCA is fit on ``StandardScaler`` outputs (per-column z-scores) so no
feature dominates covariance purely from spread. Near-constant columns are dropped before scaling; final
``feature_names`` and ``scaler_mean`` / ``scaler_scale`` are recorded in ``pca_theme_clustering.json``.
With ``--matrix scaled``, ``POLICY_THEME_SCORE_COLUMNS`` are omitted from PCA unless
``--scaled-include-policy-themes`` is set (legacy v5 drops theme proxies from the PCA block). Loadings
and PC scores are defined on that PCA input (z-scored or not). The 3D figure follows legacy v5 styling
(axis %, title, marker size, ``view_init``). By default the PNG uses **KNN local density** coloring
(viridis) so mass and tails stay visible under heavy overplot; use ``--pc3d-color solid`` for steelblue
only. Optional plot jitter (``--pc3d-plot-jitter-frac``) breaks exact duplicates on the PNG. Cumulative
variance percentages still depend on ``p`` and ``--matrix`` (they will not match high‑D v5 numbers unless
you use the same wide matrix).

Plan — this step vs legacy v5 ``03_kmeans_implementation_philgeps.py`` (3D scatter):

    Purpose. This script fits PCA on ``philgeps_min_max_scaled.csv`` for a chosen column set
    (``--matrix`` base | theme | scaled) and exports PC scores, loadings, cumulative variance, and an
    unlabeled 3D scatter (PC1–PC3). The v5 k-means script fits PCA on different geometry: StandardScaler
    on a wide matrix (many dummies + numerics; theme proxies omitted from that PCA block), with many
    components to hit a cumulative variance target; k-means runs in that reduced space. Its 3D plot is the
    first three axes of that high-D basis—not the same problem as a small-column Layer C PCA.

    Why variances and clouds differ. (1) Input space: Layer C min–max subsets here vs standardized wide
    design there. (2) Dimensionality: e.g. p=4 (base) or tens of numerics (scaled) here vs hundreds of
    features there—so per-PC shares and cumulative PC1–3 can be much smaller in v5 (~few percent)
    than here (~90%+ for base). (3) Interpretation: loadings/scatter here tie directly to the selected
    columns; v5’s PC1–3 mix many standardized columns.

    What we align for figure style. Unlabeled 3D only; subsample via ``--pc3d-max-points`` (default
    matches v5-style cap); default **density** coloring by KNN in plot space; optional jitter via
    ``--pc3d-plot-jitter-frac``; axis labels can show % variance per PC; title can follow the
    ``PhilGEPS — PCA 3D scatter (n_plot=…, 3-PC variance=…)`` pattern. Axis limits should follow this
    PCA’s cloud—do not reuse v5’s fixed PC1/PC2 limits (tuned to their scale).

    What we do not claim. Identity of PC1–PC3 or explained variance vs v5 unless inputs and scaler policy
    are intentionally harmonized.

    Optional later work. A dedicated “geometry v5” mode or side-by-side run of v5 for numerical parity;
    document ``--matrix`` and ``--pc3d-max-points`` for each published figure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Callable, Literal, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3d projection
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from philgeps_theme_scores import (
    POLICY_PCA_BASE_COLUMNS,
    POLICY_THEME_CLUSTERING_COLUMNS,
    POLICY_THEME_SCORE_COLUMNS,
)

from philgeps_paths import MEDFLOW_ROOT

PATH_SCALED_INPUT = os.path.join(
    MEDFLOW_ROOT, "output_source", "02", "Min-Max Scaling", "philgeps_min_max_scaled.csv",
)

PATH_OUTPUT_03 = os.path.join(MEDFLOW_ROOT, "output_source", "03")
PATH_OUT_CLUSTER = os.path.join(PATH_OUTPUT_03, "Clustering")
PATH_RESULTS_03 = os.path.join(MEDFLOW_ROOT, "results", "03")
PATH_RES_CLUSTER = os.path.join(PATH_RESULTS_03, "Clustering")

PATH_LOGS_03 = os.path.join(MEDFLOW_ROOT, "logs", "03")
PATH_LOG_TERMINAL = os.path.join(PATH_LOGS_03, "Terminal Logs")
PATH_LOG_ENTRIES = os.path.join(PATH_LOGS_03, "Log entries")

OUT_CLUSTER_FEATURES_CSV = os.path.join(PATH_OUT_CLUSTER, "philgeps_clustering_features.csv")
OUT_CLUSTER_PC_CSV = os.path.join(PATH_OUT_CLUSTER, "philgeps_clustering_pc_scores.csv")
OUT_PCA_JSON = os.path.join(PATH_RES_CLUSTER, "pca_theme_clustering.json")
OUT_CUMVAR_PNG = os.path.join(PATH_RES_CLUSTER, "cumulative_variance_pca.png")
OUT_LOADINGS_CSV = os.path.join(PATH_RES_CLUSTER, "pca_loadings_pc123.csv")
OUT_LOADINGS_PNG = os.path.join(PATH_RES_CLUSTER, "pca_loadings_pc123.png")
OUT_PC3D_PNG = os.path.join(PATH_RES_CLUSTER, "pca_space_pc123_3d.png")
OUT_PC3D_PNG_SOLID = os.path.join(PATH_RES_CLUSTER, "pca_space_pc123_3d_solid.png")
OUT_PC3D_HTML = os.path.join(PATH_RES_CLUSTER, "pca_space_pc123_3d_interactive.html")
OUT_PC3D_INTERACTIVE_ROWS_JSON = os.path.join(PATH_RES_CLUSTER, "pca_space_pc123_3d_interactive_rows.json")
PATH_FEATURES_SELECTED = os.path.join(
    MEDFLOW_ROOT, "output_source", "02", "Feature Selection", "philgeps_features_selected.csv",
)
REGION_BACKTRACK_COL_PROC = "Region"
REGION_BACKTRACK_COL_AW = "Region of Awardee"
OUT_PCA_DOMINANCE_AUDIT_JSON = os.path.join(PATH_RES_CLUSTER, "pca_pc123_dominance_audit.json")
OUT_README = os.path.join(PATH_RES_CLUSTER, "clustering_features_readme.txt")

RANDOM_SEED = 42
# 3D unlabeled scatter (aligned with legacy v5 step 03 style)
SCATTER_3D_S_UNLABELED = 22
SCATTER_3D_AXIS_MARGIN = 0.02
SCATTER_3D_AXIS_PAD_MIN_FRAC = 0.002
SCATTER_3D_Z_LIMIT_P_LO = 0.5
SCATTER_3D_Z_LIMIT_P_HI = 99.5
SCATTER_3D_Z_MARGIN_SCALE = 0.32
SCATTER_3D_PC1_LIM: tuple[float, float] | None = None
SCATTER_3D_PC2_LIM: tuple[float, float] | None = None
SCATTER_3D_LIM_OUTSET_FRAC = 0.055
SCATTER_3D_Z_OUTSET_REL = 0.55
VAR_EPS = 1e-18
# Plot-only: Gaussian jitter on 3D scatter. Scale per axis = F × max(peak-to-peak, std) on the subsample
# (std-only was ~0.5% of axis span here — invisible; ptp matches what you see on the axes).
# PC scores CSV stay exact.
PC3D_PLOT_JITTER_FRAC_DEFAULT = 0.025
# KNN neighbors for 3D PNG density coloring (plot coords only; mitigates uniform blob / fake grids from overplot).
PC3D_DENSITY_K_NEIGHBORS = 24


# ---------------------------------------------------------------------------
# Logging (mirror step 02)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def tee_stdio_to_file(path: str) -> Any:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as logf:

        class _Tee:
            def __init__(self, *streams: TextIO) -> None:
                self._streams = streams

            def write(self, data: str) -> None:
                for s in self._streams:
                    s.write(data)
                    s.flush()

            def flush(self) -> None:
                for s in self._streams:
                    s.flush()

        old = sys.stdout
        sys.stdout = _Tee(old, logf)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old


def open_activity_log(activity_path: str) -> Callable[[str], None]:
    os.makedirs(os.path.dirname(activity_path) or ".", exist_ok=True)
    if os.path.isfile(activity_path):
        os.remove(activity_path)

    def _log(msg: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        with open(activity_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"[{ts}] {msg}\n")

    return _log


def _pc3_knn_log_local_density(X: np.ndarray, *, k_neighbors: int) -> np.ndarray:
    """log(1 + 1/mean_dist) to k NN in plot space; higher = tighter / denser neighborhoods."""
    Xf = np.asarray(X, dtype=np.float64)
    n = int(Xf.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    k_use = min(int(k_neighbors), n - 1)
    if k_use < 1:
        return np.zeros(n, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=k_use + 1, algorithm="ball_tree")
    nn.fit(Xf)
    dists, _ = nn.kneighbors(Xf)
    reach = np.mean(dists[:, 1:], axis=1)
    return np.log1p(1.0 / np.maximum(reach, 1e-12))


def _ensure_tree() -> None:
    for p in (
        PATH_OUTPUT_03,
        PATH_OUT_CLUSTER,
        PATH_RESULTS_03,
        PATH_RES_CLUSTER,
        PATH_LOGS_03,
        PATH_LOG_TERMINAL,
        PATH_LOG_ENTRIES,
    ):
        os.makedirs(p, exist_ok=True)


def _write_dominance_audit_json(
    path: str,
    *,
    X: np.ndarray,
    feature_names: list[str],
    components3: np.ndarray,
    standardized: bool,
) -> None:
    """Pearson correlations of PCA inputs + squared-loading mass in PC1–3 per feature."""
    Xf = np.asarray(X, dtype=np.float64)
    C = np.corrcoef(Xf, rowvar=False)
    loading_sq_sum = np.sum(np.asarray(components3, dtype=np.float64) ** 2, axis=0)
    total = float(np.sum(loading_sq_sum))
    if total <= 0.0:
        participation = {f: 0.0 for f in feature_names}
    else:
        participation = {
            feature_names[j]: float(loading_sq_sum[j] / total)
            for j in range(len(feature_names))
        }
    top_feat = max(participation.items(), key=lambda kv: kv[1])[0]
    payload: dict[str, Any] = {
        "note": (
            "participation_share normalizes sum_k loading_{k,j}^2 (k=PC1..3) across features. "
            "High share means that feature's coordinates figure strongly in the 3D loading basis; "
            "with StandardScaler, correlation and PCA geometry use comparable scaling."
        ),
        "standardized_before_pca": standardized,
        "pc123_squared_loading_sum_per_feature": {
            feature_names[j]: float(loading_sq_sum[j]) for j in range(len(feature_names))
        },
        "pc123_participation_share": participation,
        "feature_with_largest_pc123_participation": top_feat,
        "feature_pearson_correlation": {
            "columns": feature_names,
            "matrix": [[float(C[i, j]) for j in range(C.shape[1])] for i in range(C.shape[0])],
        },
    }
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)


def _drop_near_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    keep: list[str] = []
    for c in df.columns:
        s = df[c]
        v = float(s.var(ddof=0))
        if s.nunique(dropna=False) <= 1 or v < VAR_EPS or not math.isfinite(v):
            dropped.append(str(c))
        else:
            keep.append(str(c))
    if dropped:
        msg = ", ".join(dropped[:25])
        if len(dropped) > 25:
            msg += ", ..."
        print(
            f"Dropping {len(dropped)} near-constant column(s) before PCA: {msg}",
            flush=True,
        )
    if not keep:
        raise ValueError("All columns are near-constant after filtering; cannot run PCA.")
    return df.loc[:, keep].copy(), dropped


def _scatter_subsample_row_indices(n_rows: int, *, max_points: int, seed: int) -> np.ndarray:
    """Integer row positions ``iloc`` into the PC matrix (length n_plot)."""
    n = int(n_rows)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def _pc3_subsample_jittered(
    pc3: np.ndarray,
    *,
    max_points: int,
    plot_jitter_frac: float,
    log_jitter: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(Xdraw, iloc_idx, n_plot)`` — same subsample + jitter as static PNGs."""
    n = int(pc3.shape[0])
    iloc_idx = _scatter_subsample_row_indices(n, max_points=max_points, seed=RANDOM_SEED)
    Xp = np.asarray(pc3[iloc_idx], dtype=np.float64)
    n_plot = int(len(iloc_idx))
    if plot_jitter_frac > 0.0 and Xp.size:
        span = np.ptp(Xp, axis=0)
        sdev = np.std(Xp, axis=0, ddof=0)
        basis = np.maximum(span, sdev)
        basis = np.maximum(basis, 1e-12)
        rng = np.random.default_rng(RANDOM_SEED + 999)
        sigma = plot_jitter_frac * basis
        Xdraw = Xp + rng.normal(size=Xp.shape).astype(np.float64) * sigma
        if log_jitter:
            print(
                "3D PNG: plot-only jitter Gaussian, scale s = "
                f"{plot_jitter_frac} * max(ptp, std) per PC on subsample (not applied to "
                f"{OUT_CLUSTER_PC_CSV}); s (PC1-PC3) ~ [{float(sigma[0]):.3f}, {float(sigma[1]):.3f}, {float(sigma[2]):.3f}].",
                flush=True,
            )
    else:
        Xdraw = Xp
    return Xdraw, iloc_idx, n_plot


def _set_axes_labels_3d_ratios(ax: Any, ratios3: list[float]) -> None:
    """Match v5 ``_set_axes_labels_3d`` (no labelpad — avoids fighting mplot3d layout)."""
    ax.set_xlabel(f"PC1 ({ratios3[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratios3[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ratios3[2] * 100:.1f}%)")


def _outset_lim_pair(lo: float, hi: float, frac: float) -> tuple[float, float]:
    if frac <= 0.0 or not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return lo, hi
    span = hi - lo
    d = span * frac * 0.5
    return float(lo - d), float(hi + d)


def _set_3d_axis_limits_tight(
    ax: Any,
    Xp: np.ndarray,
    *,
    extra_xyz: np.ndarray | None = None,
    margin: float | None = None,
) -> None:
    m = SCATTER_3D_AXIS_MARGIN if margin is None else margin
    cloud = np.asarray(Xp, dtype=float).reshape(-1, 3)
    if cloud.shape[0] == 0:
        return

    raw_lo = cloud.min(axis=0)
    raw_hi = cloud.max(axis=0)
    if extra_xyz is not None and extra_xyz.size:
        ex = np.asarray(extra_xyz, dtype=float).reshape(-1, 3)
        raw_lo = np.minimum(raw_lo, ex.min(axis=0))
        raw_hi = np.maximum(raw_hi, ex.max(axis=0))

    lo = raw_lo.copy()
    hi = raw_hi.copy()
    z_lo_p = float(np.percentile(cloud[:, 2], SCATTER_3D_Z_LIMIT_P_LO))
    z_hi_p = float(np.percentile(cloud[:, 2], SCATTER_3D_Z_LIMIT_P_HI))
    lo[2] = z_lo_p
    hi[2] = z_hi_p
    if extra_xyz is not None and extra_xyz.size:
        ez = np.asarray(extra_xyz, dtype=float).reshape(-1, 3)[:, 2]
        lo[2] = float(min(lo[2], float(ez.min())))
        hi[2] = float(max(hi[2], float(ez.max())))

    z_span = float(np.maximum(hi[2] - lo[2], 1e-9))
    z_pad = float(
        np.maximum(z_span * m, z_span * SCATTER_3D_AXIS_PAD_MIN_FRAC) * SCATTER_3D_Z_MARGIN_SCALE,
    )
    z_lim = (float(lo[2] - z_pad), float(hi[2] + z_pad))

    if SCATTER_3D_PC1_LIM is not None:
        a, b = SCATTER_3D_PC1_LIM
        x_lim = (float(min(a, b)), float(max(a, b)))
    else:
        span_x = float(np.maximum(hi[0] - lo[0], 1e-9))
        pad_x = float(np.maximum(span_x * m, span_x * SCATTER_3D_AXIS_PAD_MIN_FRAC))
        x_lim = (float(lo[0] - pad_x), float(hi[0] + pad_x))

    if SCATTER_3D_PC2_LIM is not None:
        a, b = SCATTER_3D_PC2_LIM
        y_lim = (float(min(a, b)), float(max(a, b)))
    else:
        span_y = float(np.maximum(hi[1] - lo[1], 1e-9))
        pad_y = float(np.maximum(span_y * m, span_y * SCATTER_3D_AXIS_PAD_MIN_FRAC))
        y_lim = (float(lo[1] - pad_y), float(hi[1] + pad_y))

    ox0, ox1 = _outset_lim_pair(x_lim[0], x_lim[1], SCATTER_3D_LIM_OUTSET_FRAC)
    oy0, oy1 = _outset_lim_pair(y_lim[0], y_lim[1], SCATTER_3D_LIM_OUTSET_FRAC)
    oz0, oz1 = _outset_lim_pair(
        z_lim[0], z_lim[1],
        SCATTER_3D_LIM_OUTSET_FRAC * SCATTER_3D_Z_OUTSET_REL,
    )

    ax.set_xlim(ox0, ox1)
    ax.set_ylim(oy0, oy1)
    ax.set_zlim(oz0, oz1)
    if hasattr(ax, "set_box_aspect"):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        z0, z1 = ax.get_zlim()
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))


def _save_pc_space_3d_unlabeled_v5(
    pc3: np.ndarray,
    path: str,
    *,
    ratios3: list[float],
    max_points: int,
    plot_jitter_frac: float,
    color_mode: Literal["solid", "density"],
    log_jitter: bool = True,
    log_density: bool = True,
) -> None:
    """Unlabeled 3D PCA scatter — save path aligned with v5 ``plot_pca_3d_unlabeled`` (no tight bbox)."""
    Xdraw, _iloc_idx, n_plot = _pc3_subsample_jittered(
        pc3,
        max_points=max_points,
        plot_jitter_frac=plot_jitter_frac,
        log_jitter=log_jitter,
    )
    fig_w = 11.0 if (color_mode == "density" and Xdraw.size) else 10.0
    fig = plt.figure(figsize=(fig_w, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_alpha = 0.38 if plot_jitter_frac > 0.0 else 0.35
    sc = None
    if color_mode == "density" and Xdraw.size:
        dens = _pc3_knn_log_local_density(Xdraw, k_neighbors=PC3D_DENSITY_K_NEIGHBORS)
        spread = float(np.ptp(dens))
        if spread > 0.0:
            lo, hi = np.percentile(dens, [3.0, 97.0])
            if hi <= lo:
                lo, hi = float(np.min(dens)), float(np.max(dens))
        else:
            lo, hi = 0.0, 1.0
        sc = ax.scatter(
            Xdraw[:, 0],
            Xdraw[:, 1],
            Xdraw[:, 2],
            s=SCATTER_3D_S_UNLABELED,
            c=dens,
            cmap="viridis",
            vmin=float(lo),
            vmax=float(hi),
            alpha=0.78,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        if log_density:
            print(
                f"3D PNG: color = KNN local density (k={PC3D_DENSITY_K_NEIGHBORS}, viridis; PNG only).",
                flush=True,
            )
    else:
        sc = ax.scatter(
            Xdraw[:, 0],
            Xdraw[:, 1],
            Xdraw[:, 2],
            s=SCATTER_3D_S_UNLABELED,
            alpha=plot_alpha,
            c="steelblue",
            linewidths=0,
            edgecolors="none",
        )
    _set_3d_axis_limits_tight(ax, Xdraw)
    _set_axes_labels_3d_ratios(ax, ratios3)
    cum3 = float(sum(ratios3))
    ax.set_title(
        f"PhilGEPS — PCA 3D scatter (n_plot={n_plot:,}, "
        f"3-PC variance={cum3:.3f})",
    )
    ax.view_init(elev=20, azim=30)
    # v5 uses plain tight_layout + full-figure save. bbox_inches="tight" shrinks the canvas and
    # routinely clips mplot3d z-axis labels; do not add it here.
    plt.tight_layout()
    if color_mode == "density" and sc is not None and Xdraw.size:
        fig.subplots_adjust(right=0.88)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.62, fraction=0.03, pad=0.06)
        cbar.set_label("log local density (PNG)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _region_backtrack_arrays(df_work: pd.DataFrame, *, path_fs: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(Region, Region of Awardee)`` str arrays aligned to ``df_work`` rows, or None."""
    if not os.path.isfile(path_fs):
        print(
            f"Interactive 3D: missing {path_fs}; hover Region will show (unknown). "
            "Run step 02 Feature Selection output.",
            flush=True,
        )
        return None
    try:
        fs = pd.read_csv(
            path_fs,
            usecols=lambda c: c in {REGION_BACKTRACK_COL_PROC, REGION_BACKTRACK_COL_AW},
        )
    except ValueError:
        fs = pd.read_csv(path_fs)
        for col in (REGION_BACKTRACK_COL_PROC, REGION_BACKTRACK_COL_AW):
            if col not in fs.columns:
                print(f"Interactive 3D: column {col!r} missing in features_selected; hover partial.", flush=True)
    if len(fs) != len(df_work):
        print(
            f"Interactive 3D: features_selected rows ({len(fs):,}) != PCA rows ({len(df_work):,}); "
            "Region backtrack disabled (e.g. --matrix scaled with --rows).",
            flush=True,
        )
        return None
    if not df_work.index.is_unique:
        print("Interactive 3D: non-unique dataframe index; cannot align Region. Skipping.", flush=True)
        return None
    try:
        sub = fs.loc[df_work.index]
    except KeyError:
        print(
            "Interactive 3D: index labels do not match features_selected (row order?). Region skipped.",
            flush=True,
        )
        return None
    rp = (
        sub[REGION_BACKTRACK_COL_PROC].fillna("").astype(str).to_numpy()
        if REGION_BACKTRACK_COL_PROC in sub.columns
        else np.repeat("", len(sub))
    )
    ra = (
        sub[REGION_BACKTRACK_COL_AW].fillna("").astype(str).to_numpy()
        if REGION_BACKTRACK_COL_AW in sub.columns
        else np.repeat("", len(sub))
    )
    return rp, ra


def _pandas_series_to_json_record(row: pd.Series) -> dict[str, Any]:
    """JSON-serializable dict for one FS row (no NaN / numpy scalars)."""
    out: dict[str, Any] = {}
    for col in row.index:
        name = str(col)
        v = row[col]
        if pd.isna(v):
            out[name] = None
            continue
        if isinstance(v, pd.Timestamp):
            out[name] = v.isoformat()
            continue
        if isinstance(v, (np.bool_, bool)):
            out[name] = bool(v)
            continue
        if isinstance(v, (np.integer, int)):
            out[name] = int(v)
            continue
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            out[name] = fv if math.isfinite(fv) else None
            continue
        out[name] = str(v)
    return out


def _write_pca_interactive_row_lookup_json(
    *,
    path_fs: str,
    df_work: pd.DataFrame,
    iloc_idx: np.ndarray,
    out_path: str,
) -> None:
    """Write a compact JSON map row_index -> full ``philgeps_features_selected`` row for plotted points only."""
    fail: dict[str, Any] = {
        "ok": False,
        "source_csv": os.path.basename(path_fs) if path_fs else "",
        "message": "",
        "by_row_index": {},
    }
    if not os.path.isfile(path_fs):
        fail["message"] = f"Missing features_selected CSV: {path_fs}"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(fail, f, ensure_ascii=False, separators=(",", ":"))
        print(f"Interactive rows JSON (empty): {out_path} - {fail['message']}", flush=True)
        return
    fs = pd.read_csv(path_fs)
    if len(fs) != len(df_work):
        fail["message"] = (
            f"Row count mismatch: features_selected={len(fs):,} vs PCA input={len(df_work):,} "
            "(e.g. --matrix scaled --rows)."
        )
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(fail, f, ensure_ascii=False, separators=(",", ":"))
        print(f"Interactive rows JSON (empty): {out_path} - {fail['message']}", flush=True)
        return
    by_key: dict[str, Any] = {}
    for j in range(len(iloc_idx)):
        pos = int(iloc_idx[j])
        rid = df_work.index[pos]
        key = str(rid)
        by_key[key] = _pandas_series_to_json_record(fs.iloc[pos])
    payload: dict[str, Any] = {
        "ok": True,
        "source_csv": os.path.basename(path_fs),
        "n_points": len(by_key),
        "note": (
            "Subsample only. Keys are the dataframe row index (same as philgeps_clustering_* "
            "CSV row order when using default RangeIndex)."
        ),
        "by_row_index": by_key,
    }
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Wrote interactive row lookup JSON -> {out_path} ({len(by_key):,} rows, {sz_mb:.2f} MB)", flush=True)


def _save_pc_space_3d_interactive_html(
    pc3: np.ndarray,
    path_html: str,
    *,
    ratios3: list[float],
    max_points: int,
    plot_jitter_frac: float,
    df_work: pd.DataFrame,
    path_fs: str,
    path_row_json: str,
) -> None:
    """Plotly HTML: lazy-load subsample JSON; click shows full FS row; hover shows index + regions."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("Interactive 3D: plotly not installed; pip install plotly. Skipped HTML.", flush=True)
        return

    Xdraw, iloc_idx, n_plot = _pc3_subsample_jittered(
        pc3,
        max_points=max_points,
        plot_jitter_frac=plot_jitter_frac,
        log_jitter=False,
    )
    _write_pca_interactive_row_lookup_json(
        path_fs=path_fs,
        df_work=df_work,
        iloc_idx=iloc_idx,
        out_path=path_row_json,
    )

    rows = _region_backtrack_arrays(df_work, path_fs=path_fs)
    if rows is None:
        rproc = np.asarray(["(unknown)"] * n_plot, dtype=object)
        rawardee = np.asarray(["(unknown)"] * n_plot, dtype=object)
    else:
        r_all, a_all = rows
        rproc = r_all[iloc_idx]
        rawardee = a_all[iloc_idx]
    cum3 = float(sum(ratios3))
    row_id = df_work.index.take(iloc_idx).to_numpy()

    customdata = np.column_stack(
        [
            row_id.astype(object),
            rproc.astype(object),
            rawardee.astype(object),
        ],
    )
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=Xdraw[:, 0],
                y=Xdraw[:, 1],
                z=Xdraw[:, 2],
                mode="markers",
                marker=dict(size=3, color="steelblue", opacity=0.38, line=dict(width=0)),
                customdata=customdata,
                hovertemplate=(
                    "<b>Click</b> for full Feature Selection row (loads JSON once).<br>"
                    "<b>Row index</b>=%{customdata[0]}<br>"
                    "<b>Region</b>=%{customdata[1]}<br>"
                    "<b>Region (Awardee)</b>=%{customdata[2]}<br>"
                    "PC1=%{x:.4f}<br>PC2=%{y:.4f}<br>PC3=%{z:.4f}<extra></extra>"
                ),
            ),
        ],
    )
    fig.update_layout(
        title=(
            f"PhilGEPS — PCA 3D interactive (n_plot={n_plot:,}, 3-PC variance={cum3:.3f}; click = full row)"
        ),
        scene=dict(
            xaxis_title=f"PC1 ({ratios3[0] * 100:.1f}%)",
            yaxis_title=f"PC2 ({ratios3[1] * 100:.1f}%)",
            zaxis_title=f"PC3 ({ratios3[2] * 100:.1f}%)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    plot_div = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id="pca-3d-plot",
        config={"displayModeBar": True},
    )
    row_json_basename = os.path.basename(path_row_json)
    script = f"""
<script>
(function () {{
  var ROWDATA_URL = {json.dumps(row_json_basename)};
  var rowPayload = null;
  var rowLoadPromise = null;
  function ensureRowPayload() {{
    if (rowPayload) return Promise.resolve(rowPayload);
    if (rowLoadPromise) return rowLoadPromise;
    rowLoadPromise = fetch(ROWDATA_URL)
      .then(function (r) {{ return r.json(); }})
      .then(function (j) {{ rowPayload = j; return j; }})
      .catch(function (e) {{
        document.getElementById("pca-click-details").textContent =
          "Could not load " + ROWDATA_URL + ": " + e + "\\n\\nOpen this folder over HTTP, e.g.\\n  python -m http.server 8765\\nthen visit http://localhost:8765/pca_space_pc123_3d_interactive.html";
        throw e;
      }});
    return rowLoadPromise;
  }}
  window.addEventListener("load", function () {{
    setTimeout(function () {{
      var gd = document.getElementById("pca-3d-plot");
      if (!gd || typeof gd.on !== "function") return;
      gd.on("plotly_click", function (ev) {{
        var pt = ev.points[0];
        var rowKey = String(pt.customdata[0]);
        document.getElementById("pca-click-hint").textContent =
          "Row index " + rowKey + " - loading / looking up...";
        ensureRowPayload()
          .then(function (data) {{
            var detail = document.getElementById("pca-click-details");
            if (!data.ok) {{
              detail.textContent = (data.message || "Row lookup unavailable") + "\\nRow index: " + rowKey;
              return;
            }}
            var rec = data.by_row_index[rowKey];
            if (!rec) {{
              detail.textContent =
                "No subsample entry for row index " +
                rowKey +
                ". Regenerate step 03 so HTML and *_interactive_rows.json match.";
              return;
            }}
            detail.textContent = JSON.stringify(rec, null, 2);
            document.getElementById("pca-click-hint").textContent =
              "Full row from " +
              (data.source_csv || "features_selected") +
              " (row index " +
              rowKey +
              ")";
          }})
          .catch(function () {{}});
      }});
    }}, 0);
  }});
}})();
</script>
"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>PhilGEPS PCA 3D interactive</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 12px; }}
    #pca-click-panel {{
      margin-top: 16px;
      max-height: 40vh;
      overflow: auto;
      border: 1px solid #ccc;
      border-radius: 6px;
      padding: 10px 12px;
      background: #fafafa;
    }}
    #pca-click-details {{ white-space: pre-wrap; font-size: 12px; margin: 0; }}
  </style>
</head>
<body>
{plot_div}
<div id="pca-click-panel">
  <h3 style="margin:0 0 8px 0">Feature Selection row (full record)</h3>
  <p id="pca-click-hint" style="margin:0 0 8px 0;font-size:13px;color:#444">
    Click any point. Row text is loaded once from <code>{row_json_basename}</code> (plot subsample only; full CSV is not embedded).
    If nothing loads, serve this directory over HTTP (browsers block <code>file://</code> fetches).
  </p>
  <pre id="pca-click-details"></pre>
</div>
{script}
</body>
</html>
"""
    with open(path_html, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(html)
    print(f"Saved interactive 3D PCA -> {path_html}", flush=True)


def _write_readme(
    *,
    input_used: str,
    n_samples: int,
    n_features: int,
    feature_names: list[str],
    standardized_before_pca: bool,
    pc3d_plot_jitter_frac: float,
    pc3d_color_mode: str,
) -> None:
    z_note = (
        "PCA was fit on StandardScaler outputs (zero mean, unit variance per column). "
        "Loadings/PCs are with respect to those z-scores, not raw min–max magnitudes.\n"
        if standardized_before_pca
        else "PCA was fit on min–max Layer C values as-is (no StandardScaler).\n"
    )
    color_part = (
        f"KNN density colormap (k={PC3D_DENSITY_K_NEIGHBORS}, PNG only)"
        if pc3d_color_mode == "density"
        else "solid steelblue (--pc3d-color solid)"
    )
    jitter_part = (
        "no plot jitter"
        if pc3d_plot_jitter_frac <= 0.0
        else f"plot jitter {pc3d_plot_jitter_frac} * max(ptp, std) on subsample"
    )
    jitter_line = (
        f"- {OUT_PC3D_PNG}  (v5-style 3D; {color_part}; {jitter_part}; see JSON pc3d_png_* keys)\n"
        f"- {OUT_PC3D_PNG_SOLID}  (same subsample and jitter rules; always single-color steelblue, no density colorbar)\n"
        f"- {OUT_PC3D_HTML}  (Plotly: click loads {os.path.basename(OUT_PC3D_INTERACTIVE_ROWS_JSON)} once; full FS row JSON; use HTTP server if file:// blocks fetch)\n"
    )
    body = f"""PhilGEPS step 03 — PCA / dimensionality (after step 02)

Input: {input_used}
PCA input shape: {n_samples} rows x {n_features} features (after dropping near-constant columns, if any)
Feature columns: {", ".join(feature_names)}
{z_note}
Outputs (same row order as PCA input; join by position):
- {OUT_CLUSTER_FEATURES_CSV}  (pre-z-score min–max values; columns match PCA feature_names in JSON)
- {OUT_CLUSTER_PC_CSV}  (PC1, PC2, PC3)
- {OUT_PCA_JSON}  (includes standardized_before_pca, scaler_mean/scale when used, feature_names)
- {OUT_PCA_DOMINANCE_AUDIT_JSON}  (Pearson correlations + PC1–3 loading participation by feature)
- {OUT_CUMVAR_PNG}, {OUT_LOADINGS_CSV}, {OUT_LOADINGS_PNG}
{jitter_line}

Legacy v5 ``03_kmeans_implementation_philgeps.py`` uses a wider StandardScaler matrix for clustering; this step uses Layer C subsets—geometry differs unless you harmonize inputs.
"""
    with open(OUT_README, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(body.strip() + "\n")


def _save_loadings_table_png(
    loadings_display: pd.DataFrame,
    path: str,
    *,
    title: str,
    subtitle: str | None = None,
) -> None:
    # Rows = features (long names read vertically on the left); cols = PC-1 … PC-n.
    plot_df = loadings_display.T
    n_feat, _n_pc = plot_df.shape
    longest = max((len(str(i)) for i in plot_df.index), default=10)
    fig_w = max(9.0, 5.2 + 0.09 * longest)
    fig_h = max(4.2, 0.55 * n_feat + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".6f",
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.6,
        linecolor="#cccccc",
        cbar_kws={"shrink": 0.85, "label": "Loading"},
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(plot_df.index, rotation=0, ha="right")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    fig.suptitle(
        title if not subtitle else title + "\n" + subtitle,
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_step03(
    *,
    matrix: str,
    max_pc: int,
    rows: int | None,
    pc3d_max_points: int,
    pc3d_plot_jitter_frac: float,
    pc3d_color_mode: Literal["solid", "density"],
    pc3d_write_interactive_html: bool,
    standardize: bool,
    scaled_include_policy_themes: bool,
) -> None:
    _ensure_tree()
    activity_path = os.path.join(PATH_LOG_ENTRIES, "03_PCA_Dimensionality_philgeps_activity.txt")
    log = open_activity_log(activity_path)

    log(
        "Step 03 — PCA: "
        f"matrix={matrix}, max_pc={max_pc}, rows={rows!s}, pc3d_max_points={pc3d_max_points}, "
        f"pc3d_plot_jitter_frac={pc3d_plot_jitter_frac}, pc3d_color={pc3d_color_mode}, "
        f"interactive_html={pc3d_write_interactive_html}, standardize={standardize}, "
        f"scaled_include_policy_themes={scaled_include_policy_themes}",
    )

    if not os.path.isfile(PATH_SCALED_INPUT):
        msg = f"Missing {PATH_SCALED_INPUT}. Run 02_data_preprocessing_philgeps.py first."
        log(f"ERROR: {msg}")
        print(msg, file=sys.stderr)
        sys.exit(1)

    log(f"Reading scaled numerics: {PATH_SCALED_INPUT}")
    df_full = pd.read_csv(PATH_SCALED_INPUT)
    log(f"Loaded full scaled table shape={df_full.shape}")
    if matrix == "theme":
        missing = [c for c in POLICY_THEME_CLUSTERING_COLUMNS if c not in df_full.columns]
        if missing:
            raise KeyError(
                "Theme clustering columns missing from scaled numerics: " + str(missing),
            )
        df = df_full.loc[:, list(POLICY_THEME_CLUSTERING_COLUMNS)].copy()
        input_label = f"subset of {PATH_SCALED_INPUT} ({', '.join(POLICY_THEME_CLUSTERING_COLUMNS)})"
    elif matrix == "base":
        missing = [c for c in POLICY_PCA_BASE_COLUMNS if c not in df_full.columns]
        if missing:
            raise KeyError(
                "POLICY_PCA_BASE_COLUMNS missing from scaled numerics: " + str(missing),
            )
        df = df_full.loc[:, list(POLICY_PCA_BASE_COLUMNS)].copy()
        input_label = f"subset of {PATH_SCALED_INPUT} ({', '.join(POLICY_PCA_BASE_COLUMNS)})"
    else:
        numeric = df_full.select_dtypes(include=[np.number]).columns.tolist()
        df = df_full[numeric].copy()
        if rows is not None and len(df) > rows:
            df = df.sample(n=rows, random_state=42)
        input_label = f"numeric columns from {PATH_SCALED_INPUT}"
        if not scaled_include_policy_themes:
            to_drop = [c for c in POLICY_THEME_SCORE_COLUMNS if c in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)
                omit_msg = (
                    "Omitting policy theme score columns from --matrix scaled (v5 geometry): "
                    + ", ".join(to_drop)
                )
                print(omit_msg, flush=True)
                log(omit_msg)
                input_label += (
                    "; policy theme scores omitted (use --scaled-include-policy-themes to keep)"
                )
        if rows is not None:
            log(f"--matrix scaled: using row subsample n={rows} (actual df rows={len(df):,})")

    log(f"PCA column selection ({matrix}): {input_label}")
    df_work, dropped_const = _drop_near_constant_columns(df)
    if dropped_const:
        log(f"Dropped {len(dropped_const)} near-constant column(s) before PCA")
    feature_names_for_pca = list(df_work.columns)

    n_samples, n_features = df_work.shape
    n_comp = min(max_pc, n_features, n_samples)
    log(f"PCA input: n_samples={n_samples:,}, n_features={n_features}, n_comp={n_comp} (capped)")
    if n_comp < 3:
        msg = f"Need at least 3 PCs; got n_comp={n_comp} (features={n_features}, samples={n_samples})."
        log(f"ERROR: {msg}")
        print(msg, file=sys.stderr)
        sys.exit(1)

    df_work.to_csv(OUT_CLUSTER_FEATURES_CSV, index=False)
    log(f"Wrote PCA input features -> {OUT_CLUSTER_FEATURES_CSV} (shape={df_work.shape})")
    print(f"Wrote PCA input features -> {OUT_CLUSTER_FEATURES_CSV} (shape={df_work.shape})")

    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_work.to_numpy(dtype=np.float64))
    else:
        X = df_work.to_numpy(dtype=np.float64)

    pca = PCA(n_components=n_comp)
    pca.fit(X)
    Z = pca.transform(X)
    pc3 = Z[:, :3]
    pd.DataFrame(pc3, columns=["PC1", "PC2", "PC3"]).to_csv(OUT_CLUSTER_PC_CSV, index=False)
    log(f"Wrote PC scores -> {OUT_CLUSTER_PC_CSV}")
    print(f"Wrote PC scores -> {OUT_CLUSTER_PC_CSV}")

    ratios = pca.explained_variance_ratio_
    cumulative123 = float(np.sum(ratios[:3]))
    ratios3 = [float(x) for x in ratios[:3].tolist()]
    payload: dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "feature_names": feature_names_for_pca,
        "columns_dropped_near_constant": dropped_const,
        "standardized_before_pca": bool(standardize),
        "matrix_mode": matrix,
        "scaled_policy_themes_included": bool(scaled_include_policy_themes) if matrix == "scaled" else None,
        "explained_variance_ratio": ratios3,
        "cumulative_explained_variance_pc123": cumulative123,
        "pc3d_png_plot_jitter_frac": float(pc3d_plot_jitter_frac),
        "pc3d_png_color_mode": pc3d_color_mode,
        "pc3d_density_k_neighbors": int(PC3D_DENSITY_K_NEIGHBORS) if pc3d_color_mode == "density" else None,
        "pc3d_png_unlabeled_solid": OUT_PC3D_PNG_SOLID,
        "pc3d_interactive_html": OUT_PC3D_HTML,
        "pc3d_interactive_rows_json": OUT_PC3D_INTERACTIVE_ROWS_JSON,
        "pc3d_interactive_region_source": PATH_FEATURES_SELECTED,
        "components": [[float(x) for x in row] for row in pca.components_[:3].tolist()],
    }
    if scaler is not None:
        payload["scaler_mean"] = [float(x) for x in scaler.mean_.tolist()]
        payload["scaler_scale"] = [float(x) for x in scaler.scale_.tolist()]
    else:
        payload["scaler_mean"] = None
        payload["scaler_scale"] = None

    with open(OUT_PCA_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
    log(
        f"Wrote {OUT_PCA_JSON} (cumulative PC1–PC3 variance={cumulative123:.6f}; "
        f"explained_ratio_PC123={[round(x, 6) for x in ratios3]})",
    )
    print(f"Wrote {OUT_PCA_JSON} (cumulative PC1–PC3 variance={cumulative123:.6f})")

    _write_dominance_audit_json(
        OUT_PCA_DOMINANCE_AUDIT_JSON,
        X=X,
        feature_names=feature_names_for_pca,
        components3=pca.components_[:3, :],
        standardized=bool(standardize),
    )
    log(f"Wrote dominance audit -> {OUT_PCA_DOMINANCE_AUDIT_JSON}")
    print(f"Wrote {OUT_PCA_DOMINANCE_AUDIT_JSON}")

    # Loadings table (first 3 PCs); interpret w.r.t. PCA input X (z-scored if standardize)
    loadings_df = pd.DataFrame(
        pca.components_[:3, :],
        index=[f"PC-{i + 1}" for i in range(3)],
        columns=feature_names_for_pca,
    )
    loadings_rounded = loadings_df.round(6)
    print("\nPC loadings (components_):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(loadings_rounded.to_string())
    log("Printed PC loadings (PC1–PC3) to terminal")
    loadings_rounded.to_csv(OUT_LOADINGS_CSV)
    loadings_title = {
        "base": "PCA loadings (PC1–PC3) — base numeric features",
        "theme": "PCA loadings (PC1–PC3) — policy theme aggregates",
        "scaled": "PCA loadings (PC1–PC3) — all scaled numeric columns",
    }[matrix]
    loadings_sub = (
        "Columns z-scored before PCA (StandardScaler)."
        if standardize
        else "PCA on min–max values (--no-standardize)."
    )
    _save_loadings_table_png(
        loadings_rounded,
        OUT_LOADINGS_PNG,
        title=loadings_title,
        subtitle=loadings_sub,
    )
    log(f"Saved loadings table + heatmap -> {OUT_LOADINGS_CSV}, {OUT_LOADINGS_PNG}")
    print(f"Saved loadings -> {OUT_LOADINGS_CSV}, {OUT_LOADINGS_PNG}")

    # Cumulative variance plot
    cumulative = np.cumsum(ratios)
    pcs = np.arange(1, len(cumulative) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcs, cumulative, "b-o", linewidth=2, markersize=6)
    ax.axhline(cumulative123, color="gray", linestyle="--", alpha=0.6)
    if len(cumulative) >= 3:
        ax.scatter([3], [cumulative123], color="red", s=80, zorder=5)
        ax.annotate(
            f"PC 1–3: {cumulative123:.2%}",
            xy=(3, cumulative123),
            xytext=(3 + 0.15 * n_comp, cumulative123 - 0.05),
            fontsize=10,
        )
    ax.set_xlabel("PC")
    ax.set_ylabel("cumulative variance explained")
    ax.set_xticks(pcs)
    ax.set_ylim(max(0.0, float(cumulative.min()) - 0.05), min(1.0, float(cumulative.max()) + 0.05))
    ax.grid(True, alpha=0.3)
    if matrix == "theme":
        plot_caption = "Theme clustering features"
    elif matrix == "base":
        plot_caption = "Base numerics (theme back-track)"
    else:
        plot_caption = "Layer C scaled numerics"
    ax.set_title(f"Cumulative explained variance ({plot_caption})")
    fig.tight_layout()
    fig.savefig(OUT_CUMVAR_PNG, dpi=150)
    plt.close(fig)
    log(f"Saved cumulative variance plot -> {OUT_CUMVAR_PNG}")
    print(f"Saved plot -> {OUT_CUMVAR_PNG}")

    _save_pc_space_3d_unlabeled_v5(
        pc3,
        OUT_PC3D_PNG,
        ratios3=ratios3,
        max_points=pc3d_max_points,
        plot_jitter_frac=pc3d_plot_jitter_frac,
        color_mode=pc3d_color_mode,
        log_jitter=True,
        log_density=True,
    )
    log(f"Saved 3D PCA scatter -> {OUT_PC3D_PNG}")
    print(f"Saved 3D PCA scatter -> {OUT_PC3D_PNG}")

    if pc3d_color_mode == "density":
        _save_pc_space_3d_unlabeled_v5(
            pc3,
            OUT_PC3D_PNG_SOLID,
            ratios3=ratios3,
            max_points=pc3d_max_points,
            plot_jitter_frac=pc3d_plot_jitter_frac,
            color_mode="solid",
            log_jitter=False,
            log_density=False,
        )
    else:
        shutil.copy2(OUT_PC3D_PNG, OUT_PC3D_PNG_SOLID)
    log(f"Saved 3D PCA scatter (solid) -> {OUT_PC3D_PNG_SOLID}")
    print(f"Saved 3D PCA scatter (solid) -> {OUT_PC3D_PNG_SOLID}")

    if pc3d_write_interactive_html:
        _save_pc_space_3d_interactive_html(
            pc3,
            OUT_PC3D_HTML,
            ratios3=ratios3,
            max_points=pc3d_max_points,
            plot_jitter_frac=pc3d_plot_jitter_frac,
            df_work=df_work,
            path_fs=PATH_FEATURES_SELECTED,
            path_row_json=OUT_PC3D_INTERACTIVE_ROWS_JSON,
        )
        log(f"Wrote interactive 3D HTML + row JSON -> {OUT_PC3D_HTML}, {OUT_PC3D_INTERACTIVE_ROWS_JSON}")
    else:
        log("Interactive 3D HTML: skipped (--no-pc3d-interactive-html).")
        print("Interactive 3D HTML: skipped (--no-pc3d-interactive-html).", flush=True)

    _write_readme(
        input_used=input_label,
        n_samples=n_samples,
        n_features=n_features,
        feature_names=feature_names_for_pca,
        standardized_before_pca=standardize,
        pc3d_plot_jitter_frac=pc3d_plot_jitter_frac,
        pc3d_color_mode=pc3d_color_mode,
    )
    log(f"Wrote readme -> {OUT_README}")
    print(f"Wrote readme -> {OUT_README}")

    print(
        "PhilGEPS step 03 done. "
        f"PCA input: {OUT_CLUSTER_FEATURES_CSV}; PC scores: {OUT_CLUSTER_PC_CSV}; "
        f"results: {PATH_RES_CLUSTER}; logs: {PATH_LOGS_03}. "
        "Run k-means step 04 on PC scores when ready.",
        flush=True,
    )
    log(
        f"Step 03 complete. outputs: {OUT_CLUSTER_FEATURES_CSV}, {OUT_CLUSTER_PC_CSV}, "
        f"results_dir={PATH_RES_CLUSTER}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PhilGEPS step 03 — PCA on step-02 scaled numerics")
    parser.add_argument(
        "--matrix",
        choices=("theme", "base", "scaled"),
        default="base",
        help="base: POLICY_PCA_BASE_COLUMNS (default; feature names in loadings). theme: theme aggregates. scaled: all.",
    )
    parser.add_argument(
        "--max-pc",
        type=int,
        default=9,
        metavar="K",
        help="Number of PCs to fit (capped by n_features); used for scree plot.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional row subsample (--matrix scaled only).",
    )
    parser.add_argument(
        "--pc3d-max-points",
        type=int,
        default=30_000,
        metavar="N",
        help="Max points in 3D PC scatter (0 = all rows; default 30000, matches v5 plot subsample).",
    )
    parser.add_argument(
        "--pc3d-plot-jitter-frac",
        type=float,
        default=PC3D_PLOT_JITTER_FRAC_DEFAULT,
        metavar="F",
        help=(
            "Plot-only Gaussian jitter for 3D PNG: per-PC s = F * max(peak-to-peak, std) on the drawn subsample "
            f"(default {PC3D_PLOT_JITTER_FRAC_DEFAULT}; 0 = no jitter; does not modify philgeps_clustering_pc_scores.csv)."
        ),
    )
    parser.add_argument(
        "--pc3d-color",
        choices=("density", "solid"),
        default="density",
        help=(
            "3D PNG point colors: density = KNN log-density viridis + colorbar (default); "
            "solid = steelblue only (legacy v5 look)."
        ),
    )
    parser.add_argument(
        "--no-pc3d-interactive-html",
        action="store_true",
        help="Do not write pca_space_pc123_3d_interactive.html (Plotly hover backtrack).",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Fit PCA on min–max columns without StandardScaler (legacy; restores pre-v5-parity scaling).",
    )
    parser.add_argument(
        "--scaled-include-policy-themes",
        action="store_true",
        help="With --matrix scaled, keep POLICY_THEME_SCORE_COLUMNS in PCA (default: omit for v5-style geometry).",
    )
    args = parser.parse_args()
    if args.matrix == "theme" and args.rows is not None:
        print("Warning: --rows applies to --matrix scaled only; ignored for theme/base.", file=sys.stderr)
    jf = float(args.pc3d_plot_jitter_frac)
    if not math.isfinite(jf) or jf < 0.0:
        print("--pc3d-plot-jitter-frac must be a finite number >= 0.", file=sys.stderr)
        sys.exit(2)

    _ensure_tree()
    term_log = os.path.join(PATH_LOG_TERMINAL, "03_PCA_Dimensionality_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        run_step03(
            matrix=args.matrix,
            max_pc=args.max_pc,
            rows=args.rows if args.matrix == "scaled" else None,
            pc3d_max_points=args.pc3d_max_points,
            pc3d_plot_jitter_frac=jf,
            pc3d_color_mode=args.pc3d_color,
            pc3d_write_interactive_html=not args.no_pc3d_interactive_html,
            standardize=not args.no_standardize,
            scaled_include_policy_themes=args.scaled_include_policy_themes,
        )


if __name__ == "__main__":
    main()