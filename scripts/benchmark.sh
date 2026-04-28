#!/usr/bin/env bash
# scripts/benchmark.sh
# ─────────────────────────────────────────────
# Full benchmark sweep: vary image size, K, and thread count.
# Runs all four modes and writes every result to results/bench.csv.
#
# Pre-requisites
# ──────────────
#   • Build directory at  ../build/  relative to this script
#   • Images at  ../data/512.png  ../data/1024.png  etc.
#   • (Optional) Place any colour-rich PNG you like in data/ and
#     update IMAGE_SIZES below.
#
# Usage
# ─────
#   chmod +x scripts/benchmark.sh
#   ./scripts/benchmark.sh
# ─────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/.."
BIN="${ROOT}/build/genetic_kmeans"
DATA="${ROOT}/data"
RESULTS="${ROOT}/results"
CSV="${RESULTS}/bench.csv"

mkdir -p "${RESULTS}"

# ── Parameters to sweep ──────────────────────
IMAGE_SIZES=(512 1024 2048 4096)
K_VALUES=(8 16 32 64)
THREAD_COUNTS=(1 2 4 8 16)   # for gka_omp
POPULATION=20
GENERATIONS=50
MUTATION=0.10
SIGMA=15.0
REFINE=2
SEED=42

echo "─────────────────────────────────────────"
echo " Genetic K-Means Benchmark"
echo " Output CSV : ${CSV}"
echo "─────────────────────────────────────────"

for SIZE in "${IMAGE_SIZES[@]}"; do
    IMG="${DATA}/${SIZE}.png"
    if [[ ! -f "${IMG}" ]]; then
        echo "[SKIP] ${IMG} not found – place a ${SIZE}×${SIZE} PNG there."
        continue
    fi

    for K in "${K_VALUES[@]}"; do

        echo ""
        echo ">>> image=${SIZE}.png  K=${K}"

        # ── Standard K-means (baseline) ──
        echo "    [kmeans]"
        "${BIN}" \
            --mode kmeans \
            --image "${IMG}" \
            --K "${K}" \
            --out "${RESULTS}" \
            --csv "${CSV}" \
            --seed "${SEED}"

        # ── Sequential GKA ──
        echo "    [gka_seq]"
        "${BIN}" \
            --mode gka_seq \
            --image "${IMG}" \
            --K "${K}" \
            --P "${POPULATION}" \
            --G "${GENERATIONS}" \
            --mut "${MUTATION}" \
            --sigma "${SIGMA}" \
            --refine "${REFINE}" \
            --out "${RESULTS}" \
            --csv "${CSV}" \
            --seed "${SEED}"

        # ── OpenMP GKA – sweep thread count ──
        for T in "${THREAD_COUNTS[@]}"; do
            echo "    [gka_omp threads=${T}]"
            "${BIN}" \
                --mode gka_omp \
                --image "${IMG}" \
                --K "${K}" \
                --P "${POPULATION}" \
                --G "${GENERATIONS}" \
                --mut "${MUTATION}" \
                --sigma "${SIGMA}" \
                --refine "${REFINE}" \
                --threads "${T}" \
                --out "${RESULTS}" \
                --csv "${CSV}" \
                --seed "${SEED}"
        done

        # ── CUDA GKA ──
        echo "    [gka_cuda]"
        "${BIN}" \
            --mode gka_cuda \
            --image "${IMG}" \
            --K "${K}" \
            --P "${POPULATION}" \
            --G "${GENERATIONS}" \
            --mut "${MUTATION}" \
            --sigma "${SIGMA}" \
            --refine "${REFINE}" \
            --out "${RESULTS}" \
            --csv "${CSV}" \
            --seed "${SEED}"

    done
done

echo ""
echo "─────────────────────────────────────────"
echo " Benchmark complete.  Results in ${CSV}"
echo "─────────────────────────────────────────"
