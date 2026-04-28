/*  genetic_kmeans.cpp  –  Sequential Genetic K-means Algorithm (GKA)
 *
 *  Based on: Bandyopadhyay & Maulik, "An Efficient Technique for
 *  Clustering with Genetic Algorithms" (2002).
 *
 *  Chromosome  = vector of K Pixel centroids
 *  Fitness     = ICV (intra-cluster variance) – minimised
 *
 *  Operators
 *  ─────────
 *  Initialisation : K random pixels sampled from the image
 *  Selection      : binary tournament (size 2), elitism (top 1 kept)
 *  Crossover      : uniform (each centroid from parent A or B with p=0.5)
 *  Mutation       : Gaussian perturbation of each centroid (rate per centroid)
 *  Local refinement : 1-2 K-means steps applied to every child after crossover
 *  Repair         : empty clusters re-seeded from a random pixel
 */

#include "genetic_kmeans.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

// ─────────────────────────────────────────────
//  Internal types
// ─────────────────────────────────────────────

using Chromosome  = std::vector<Pixel>;
using Population  = std::vector<Chromosome>;
using LabelMatrix = std::vector<std::vector<int>>;

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

static inline float dist_sq(const Pixel& a, const Pixel& b) {
    float dr = a.r - b.r, dg = a.g - b.g, db = a.b - b.b;
    return dr*dr + dg*dg + db*db;
}

/**
 * Assign every pixel to its nearest centroid.
 * Returns raw sum-of-squared-distances (divide by N for ICV).
 */
static double assign_pixels(const std::vector<Pixel>& pixels,
                             const Chromosome&          centroids,
                             std::vector<int>&           labels) {
    int N = static_cast<int>(pixels.size());
    int K = static_cast<int>(centroids.size());
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
        float best_d = std::numeric_limits<float>::max();
        int   best_k = 0;
        for (int k = 0; k < K; ++k) {
            float d = dist_sq(pixels[i], centroids[k]);
            if (d < best_d) { best_d = d; best_k = k; }
        }
        labels[i] = best_k;
        total += static_cast<double>(best_d);
    }
    return total / static_cast<double>(N); // ICV
}

/** Recompute centroids from current label assignments. */
static void update_centroids(const std::vector<Pixel>& pixels,
                              const std::vector<int>&    labels,
                              Chromosome&                centroids,
                              std::mt19937&              rng) {
    int K = static_cast<int>(centroids.size());
    int N = static_cast<int>(pixels.size());
    std::vector<Pixel> acc(K, {0.f, 0.f, 0.f});
    std::vector<int>   cnt(K, 0);

    for (int i = 0; i < N; ++i) {
        int c = labels[i];
        acc[c].r += pixels[i].r;
        acc[c].g += pixels[i].g;
        acc[c].b += pixels[i].b;
        cnt[c]++;
    }

    std::uniform_int_distribution<int> rnd_pixel(0, N - 1);
    for (int k = 0; k < K; ++k) {
        if (cnt[k] > 0) {
            centroids[k] = { acc[k].r / cnt[k],
                             acc[k].g / cnt[k],
                             acc[k].b / cnt[k] };
        } else {
            // Empty-cluster repair: replace with a random pixel
            centroids[k] = pixels[rnd_pixel(rng)];
        }
    }
}

/**
 * Apply `steps` rounds of K-means update to `chrom` in-place.
 * Returns the ICV after the last assignment step.
 */
static double local_refine(const std::vector<Pixel>& pixels,
                            Chromosome&               chrom,
                            std::vector<int>&          labels,
                            int                        steps,
                            std::mt19937&              rng) {
    double icv = 0.0;
    for (int s = 0; s < steps; ++s) {
        icv = assign_pixels(pixels, chrom, labels);
        update_centroids(pixels, labels, chrom, rng);
    }
    // One final assignment so labels match the updated centroids
    icv = assign_pixels(pixels, chrom, labels);
    return icv;
}

// ─────────────────────────────────────────────
//  Genetic operators
// ─────────────────────────────────────────────

static Chromosome init_chromosome(const std::vector<Pixel>& pixels,
                                   int K,
                                   std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(pixels.size()) - 1);
    Chromosome chrom(K);
    for (int k = 0; k < K; ++k)
        chrom[k] = pixels[dist(rng)];
    return chrom;
}

/** Uniform crossover: each gene (centroid) taken from parent A or B. */
static Chromosome crossover(const Chromosome& a,
                             const Chromosome& b,
                             std::mt19937&     rng) {
    int K = static_cast<int>(a.size());
    Chromosome child(K);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int k = 0; k < K; ++k)
        child[k] = coin(rng) ? a[k] : b[k];
    return child;
}

/** Gaussian centroid perturbation. */
static void mutate(Chromosome&   chrom,
                   double        rate,
                   float         sigma,
                   std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.f, 1.f);
    std::normal_distribution<float>       noise(0.f, sigma);
    for (auto& c : chrom) {
        if (prob(rng) < static_cast<float>(rate)) {
            c.r = std::clamp(c.r + noise(rng), 0.f, 255.f);
            c.g = std::clamp(c.g + noise(rng), 0.f, 255.f);
            c.b = std::clamp(c.b + noise(rng), 0.f, 255.f);
        }
    }
}

/** Binary tournament selection – returns index of winner. */
static int tournament(const std::vector<double>& fitness,
                       std::mt19937&              rng) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(fitness.size()) - 1);
    int a = dist(rng), b = dist(rng);
    return fitness[a] < fitness[b] ? a : b;
}

// ─────────────────────────────────────────────
//  Sequential GKA solver
// ─────────────────────────────────────────────

GKAResult run_gka_sequential(const std::vector<Pixel>& pixels,
                              const GKAParams&          params) {
    if (pixels.empty())
        throw std::invalid_argument("Pixel list is empty.");
    if (params.K < 1 || params.population < 2)
        throw std::invalid_argument("K >= 1 and population >= 2 required.");

    auto t_start = std::chrono::high_resolution_clock::now();

    const int N = static_cast<int>(pixels.size());
    const int K = params.K;
    const int P = params.population;
    const int G = params.generations;

    std::mt19937 rng(params.seed);

    // ── Initialise population ────────────────
    Population         pop(P);
    std::vector<double> fitness(P);
    LabelMatrix        labels(P, std::vector<int>(N));

    for (int i = 0; i < P; ++i) {
        pop[i]     = init_chromosome(pixels, K, rng);
        fitness[i] = local_refine(pixels, pop[i], labels[i],
                                  params.local_refine_steps, rng);
    }

    int best_idx = static_cast<int>(
        std::min_element(fitness.begin(), fitness.end()) - fitness.begin());

    GKAResult result;
    result.fitness_history.reserve(G);

    // ── Evolution loop ───────────────────────
    for (int gen = 0; gen < G; ++gen) {
        Population         new_pop(P);
        std::vector<double> new_fit(P);
        LabelMatrix        new_lbl(P, std::vector<int>(N));

        // Elitism: carry forward the current best unchanged
        new_pop[0] = pop[best_idx];
        new_fit[0] = fitness[best_idx];
        new_lbl[0] = labels[best_idx];

        for (int i = 1; i < P; ++i) {
            int p1 = tournament(fitness, rng);
            int p2 = tournament(fitness, rng);
            Chromosome child = crossover(pop[p1], pop[p2], rng);
            mutate(child, params.mutation_rate, params.mutation_sigma, rng);
            new_fit[i] = local_refine(pixels, child, new_lbl[i],
                                      params.local_refine_steps, rng);
            new_pop[i] = std::move(child);
        }

        pop     = std::move(new_pop);
        fitness = std::move(new_fit);
        labels  = std::move(new_lbl);

        best_idx = static_cast<int>(
            std::min_element(fitness.begin(), fitness.end()) - fitness.begin());

        result.fitness_history.push_back(fitness[best_idx]);
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    result.best_centroids = pop[best_idx];
    result.best_labels    = labels[best_idx];
    result.best_fitness   = fitness[best_idx];
    result.wall_time_ms   =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return result;
}
