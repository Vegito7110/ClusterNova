#!/usr/bin/env python3
"""scripts/plot_results.py
Reads results/bench.csv and produces all report-ready figures.

Figures generated
─────────────────
1.  speedup_vs_threads.png  – OMP speedup over sequential per image size
2.  cuda_vs_omp.png         – CUDA vs best-OMP wall time (bar chart)
3.  psnr_vs_K.png           – PSNR as K increases for each mode
4.  icv_vs_K.png            – ICV as K increases for each mode
5.  convergence.png         – (if fitness_history.csv exists) ICV per generation
6.  scalability_heatmap.png – speedup heatmap: K × image_size

Usage
─────
    pip install pandas matplotlib seaborn
    python scripts/plot_results.py [--csv results/bench.csv] [--out results/plots/]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  default="results/bench.csv")
    p.add_argument("--out",  default="results/plots")
    return p.parse_args()

# ── Helper ─────────────────────────────────────────────────────────────────

def savefig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Plot functions ─────────────────────────────────────────────────────────

def plot_speedup_vs_threads(df, out_dir):
    """OMP speedup = t_seq / t_omp(T threads)  for each (image, K)."""
    seq  = df[df["mode"] == "gka_seq"][["image", "K", "wall_ms"]].rename(columns={"wall_ms": "seq_ms"})
    omp  = df[df["mode"] == "gka_omp"].copy()
    merged = omp.merge(seq, on=["image", "K"])
    merged["speedup"] = merged["seq_ms"] / merged["wall_ms"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for img, grp in merged.groupby("image"):
        g = grp.groupby("threads")["speedup"].mean().reset_index()
        ax.plot(g["threads"], g["speedup"], marker="o", label=img)

    ideal_t = sorted(merged["threads"].unique())
    ax.plot(ideal_t, ideal_t, "k--", alpha=0.4, label="Ideal linear")
    ax.set_xlabel("Thread count")
    ax.set_ylabel("Speedup over gka_seq")
    ax.set_title("OpenMP Speedup vs Thread Count")
    ax.legend(title="Image")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    savefig(fig, out_dir, "speedup_vs_threads.png")


def plot_cuda_vs_omp(df, out_dir):
    """Side-by-side bar: best OMP vs CUDA wall time per (image, K)."""
    best_omp = (
        df[df["mode"] == "gka_omp"]
        .groupby(["image", "K"])["wall_ms"].min()
        .reset_index()
        .rename(columns={"wall_ms": "omp_ms"})
    )
    cuda = (
        df[df["mode"] == "gka_cuda"][["image", "K", "wall_ms"]]
        .rename(columns={"wall_ms": "cuda_ms"})
    )
    merged = best_omp.merge(cuda, on=["image", "K"])
    merged["label"] = merged["image"].str.replace(".png","") + "_K" + merged["K"].astype(str)

    x = np.arange(len(merged))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(merged)), 5))
    ax.bar(x - w/2, merged["omp_ms"],  w, label="Best OMP")
    ax.bar(x + w/2, merged["cuda_ms"], w, label="CUDA")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"], rotation=45, ha="right")
    ax.set_ylabel("Wall time (ms)")
    ax.set_title("CUDA vs Best-OMP Wall Time")
    ax.legend()
    savefig(fig, out_dir, "cuda_vs_omp.png")


def plot_quality_vs_K(df, out_dir):
    """PSNR and ICV vs K for each mode (averaged over image sizes)."""
    modes = ["kmeans", "gka_seq", "gka_omp", "gka_cuda"]
    palette = sns.color_palette("deep", len(modes))

    for metric, ylabel, fname in [
        ("PSNR", "PSNR (dB)",   "psnr_vs_K.png"),
        ("ICV",  "ICV (mean sq dist)", "icv_vs_K.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for mode, col in zip(modes, palette):
            sub = df[df["mode"] == mode].groupby("K")[metric].mean().reset_index()
            if sub.empty:
                continue
            ax.plot(sub["K"], sub[metric], marker="o", label=mode, color=col)
        ax.set_xlabel("K  (number of clusters / colours)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} vs K")
        ax.legend()
        savefig(fig, out_dir, fname)


def plot_scalability_heatmap(df, out_dir):
    """Heatmap of OMP speedup over sequential, axes: K × image width."""
    seq  = df[df["mode"] == "gka_seq"][["image", "K", "wall_ms"]].rename(columns={"wall_ms": "seq_ms"})
    omp  = df[(df["mode"] == "gka_omp")].copy()
    merged = omp.merge(seq, on=["image", "K"])
    merged["speedup"] = merged["seq_ms"] / merged["wall_ms"]
    best = merged.groupby(["width", "K"])["speedup"].max().reset_index()

    if best.empty:
        return
    pivot = best.pivot(index="K", columns="width", values="speedup")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Peak OMP Speedup (K × Image Width)")
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("K")
    savefig(fig, out_dir, "scalability_heatmap.png")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Derived columns
    if "width" in df.columns:
        df["pixels"] = df["width"] * df["height"]

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Modes found: {df['mode'].unique().tolist()}")
    print(f"K values:    {sorted(df['K'].unique().tolist())}")

    plot_speedup_vs_threads(df, args.out)
    plot_cuda_vs_omp(df, args.out)
    plot_quality_vs_K(df, args.out)
    plot_scalability_heatmap(df, args.out)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
