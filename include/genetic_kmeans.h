#pragma once
#include "kmeans.h"
#include <vector>

// ─────────────────────────────────────────────
//  Algorithm parameters
// ─────────────────────────────────────────────

struct GKAParams {
    int          K;                  // clusters (= colour palette size)
    int          population;         // chromosomes per generation (P)
    int          generations;        // number of generations (G)
    double       mutation_rate;      // per-centroid mutation probability
    float        mutation_sigma;     // Gaussian noise std-dev (pixel units)
    int          local_refine_steps; // K-means steps applied after crossover
    unsigned int seed;               // global RNG seed
};

// ─────────────────────────────────────────────
//  Result type
// ─────────────────────────────────────────────

struct GKAResult {
    std::vector<Pixel>   best_centroids;
    std::vector<int>     best_labels;
    double               best_fitness;    // final ICV
    double               wall_time_ms;
    std::vector<double>  fitness_history; // best ICV per generation
};

// ─────────────────────────────────────────────
//  Three solver variants
// ─────────────────────────────────────────────

/** Sequential GKA – reference implementation */
GKAResult run_gka_sequential(const std::vector<Pixel>& pixels,
                              const GKAParams& params);

/** OpenMP GKA – population fitness loop parallelised */
GKAResult run_gka_openmp(const std::vector<Pixel>& pixels,
                          const GKAParams& params,
                          int num_threads = 0); // 0 = use OMP_NUM_THREADS

/** CUDA GKA – assignment / fitness step offloaded to GPU.
 *  Falls back to sequential if no CUDA device is found at runtime. */
GKAResult run_gka_cuda(const std::vector<Pixel>& pixels,
                        const GKAParams& params);
