#pragma once
#include <vector>
#include <string>

// ─────────────────────────────────────────────
//  Core data types
// ─────────────────────────────────────────────

struct Pixel {
    float r, g, b;
};

struct KMeansResult {
    std::vector<Pixel>  centroids;
    std::vector<int>    labels;
    double              intra_cluster_variance; // ICV – lower is better
    int                 iterations;
    double              wall_time_ms;
};

// ─────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────

/**
 * Standard K-means with K-means++ initialisation.
 * pixels    : flat list of RGB pixels in [0,255]
 * K         : number of clusters / colours
 * max_iter  : convergence cap
 * seed      : RNG seed for reproducibility
 */
KMeansResult run_kmeans(const std::vector<Pixel>& pixels,
                        int K,
                        int max_iter = 100,
                        unsigned int seed = 42);

/**
 * Peak Signal-to-Noise Ratio between original and
 * quantised pixel arrays (higher = better quality).
 */
double compute_psnr(const std::vector<Pixel>& original,
                    const std::vector<Pixel>& compressed);

/**
 * Mean squared distance of every pixel to its
 * assigned centroid (the GKA fitness function).
 */
double intra_cluster_variance(const std::vector<Pixel>&  pixels,
                               const std::vector<int>&    labels,
                               const std::vector<Pixel>&  centroids,
                               int K);
