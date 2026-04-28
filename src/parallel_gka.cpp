/*  parallel_gka.cpp  –  OpenMP-parallel Genetic K-means Algorithm
 *
 *  Parallelism strategy
 *  ────────────────────
 *  The population fitness-evaluation loop (the inner `for i` over P chromosomes)
 *  is embarrassingly parallel: each chromosome is independent.
 *  We parallelise it with  #pragma omp parallel for.
 *
 *  Thread-safety
 *  ─────────────
 *  Each thread gets its own mt19937 seeded from the master seed + thread id,
 *  so results are reproducible for a given (seed, num_threads) pair.
 *
 *  Genetic operators (selection, crossover, mutation) run serially on the
 *  master thread after the parallel fitness sweep.  This keeps the code
 *  correct without needing a parallel RNG library.
 */

#include "genetic_kmeans.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ─────────────────────────────────────────────
//  Internal helpers  (duplicated here so this
//  TU is self-contained and avoids linking to
//  genetic_kmeans.cpp internals)
// ─────────────────────────────────────────────

using Chromosome  = std::vector<Pixel>;
using Population  = std::vector<Chromosome>;
using LabelMatrix = std::vector<std::vector<int>>;

static inline float dist_sq(const Pixel& a, const Pixel& b) {
    float dr = a.r-b.r, dg = a.g-b.g, db = a.b-b.b;
    return dr*dr + dg*dg + db*db;
}

static double assign_pixels(const std::vector<Pixel>& pixels,
                             const Chromosome& centroids,
                             std::vector<int>& labels) {
    int N = (int)pixels.size(), K = (int)centroids.size();
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
        float best = std::numeric_limits<float>::max(); int bk = 0;
        for (int k = 0; k < K; ++k) {
            float d = dist_sq(pixels[i], centroids[k]);
            if (d < best) { best = d; bk = k; }
        }
        labels[i] = bk;
        total += best;
    }
    return total / N;
}

static void update_centroids(const std::vector<Pixel>& pixels,
                              const std::vector<int>& labels,
                              Chromosome& centroids,
                              std::mt19937& rng) {
    int K = (int)centroids.size(), N = (int)pixels.size();
    std::vector<Pixel> acc(K,{0,0,0});
    std::vector<int>   cnt(K,0);
    for (int i = 0; i < N; ++i) {
        acc[labels[i]].r += pixels[i].r;
        acc[labels[i]].g += pixels[i].g;
        acc[labels[i]].b += pixels[i].b;
        cnt[labels[i]]++;
    }
    std::uniform_int_distribution<int> rp(0, N-1);
    for (int k = 0; k < K; ++k) {
        if (cnt[k] > 0)
            centroids[k] = {acc[k].r/cnt[k], acc[k].g/cnt[k], acc[k].b/cnt[k]};
        else
            centroids[k] = pixels[rp(rng)];
    }
}

static double local_refine(const std::vector<Pixel>& pixels,
                            Chromosome& chrom,
                            std::vector<int>& labels,
                            int steps,
                            std::mt19937& rng) {
    double icv = 0;
    for (int s = 0; s < steps; ++s) {
        icv = assign_pixels(pixels, chrom, labels);
        update_centroids(pixels, labels, chrom, rng);
    }
    icv = assign_pixels(pixels, chrom, labels);
    return icv;
}

static Chromosome init_chromosome(const std::vector<Pixel>& pixels, int K, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(0,(int)pixels.size()-1);
    Chromosome c(K);
    for (auto& p : c) p = pixels[d(rng)];
    return c;
}

static Chromosome crossover(const Chromosome& a, const Chromosome& b, std::mt19937& rng) {
    int K = (int)a.size();
    Chromosome child(K);
    std::uniform_int_distribution<int> coin(0,1);
    for (int k = 0; k < K; ++k) child[k] = coin(rng) ? a[k] : b[k];
    return child;
}

static void mutate(Chromosome& c, double rate, float sigma, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0,1);
    std::normal_distribution<float>       noise(0, sigma);
    for (auto& p : c) {
        if (prob(rng) < (float)rate) {
            p.r = std::clamp(p.r+noise(rng), 0.f, 255.f);
            p.g = std::clamp(p.g+noise(rng), 0.f, 255.f);
            p.b = std::clamp(p.b+noise(rng), 0.f, 255.f);
        }
    }
}

static int tournament(const std::vector<double>& f, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(0,(int)f.size()-1);
    int a = d(rng), b = d(rng);
    return f[a] < f[b] ? a : b;
}

// ─────────────────────────────────────────────
//  OpenMP GKA solver
// ─────────────────────────────────────────────

GKAResult run_gka_openmp(const std::vector<Pixel>& pixels,
                          const GKAParams& params,
                          int num_threads) {
    if (pixels.empty()) throw std::invalid_argument("Empty pixel list.");
    if (params.K < 1 || params.population < 2)
        throw std::invalid_argument("K>=1 and population>=2 required.");

#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    int T = omp_get_max_threads();
#else
    int T = 1;
    (void)num_threads;
#endif

    auto t_start = std::chrono::high_resolution_clock::now();

    const int N = (int)pixels.size();
    const int K = params.K;
    const int P = params.population;
    const int G = params.generations;

    // One RNG per thread for thread-safe generation
    std::vector<std::mt19937> rngs(T);
    for (int t = 0; t < T; ++t)
        rngs[t].seed(params.seed + (unsigned)t * 6364136223846793005ULL);

    // Master RNG for selection / crossover / mutation (serial sections)
    std::mt19937 master_rng(params.seed ^ 0xDEADBEEF);

    // ── Initialise population (parallel) ────
    Population          pop(P);
    std::vector<double> fitness(P);
    LabelMatrix         labels(P, std::vector<int>(N));

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < P; ++i) {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        pop[i]     = init_chromosome(pixels, K, rngs[tid]);
        fitness[i] = local_refine(pixels, pop[i], labels[i],
                                  params.local_refine_steps, rngs[tid]);
    }

    int best_idx = (int)(std::min_element(fitness.begin(),fitness.end()) - fitness.begin());

    GKAResult result;
    result.fitness_history.reserve(G);

    // ── Evolution loop ───────────────────────
    for (int gen = 0; gen < G; ++gen) {

        // ① Serial: generate children via selection + crossover + mutation
        Population         children(P);
        children[0] = pop[best_idx]; // elitism

        for (int i = 1; i < P; ++i) {
            int p1 = tournament(fitness, master_rng);
            int p2 = tournament(fitness, master_rng);
            children[i] = crossover(pop[p1], pop[p2], master_rng);
            mutate(children[i], params.mutation_rate, params.mutation_sigma, master_rng);
        }

        // ② Parallel: evaluate fitness of each child
        std::vector<double> new_fit(P);
        LabelMatrix         new_lbl(P, std::vector<int>(N));

        new_fit[0] = fitness[best_idx];
        new_lbl[0] = labels[best_idx];

#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 1; i < P; ++i) {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            new_fit[i] = local_refine(pixels, children[i], new_lbl[i],
                                      params.local_refine_steps, rngs[tid]);
        }

        pop     = std::move(children);
        fitness = std::move(new_fit);
        labels  = std::move(new_lbl);

        best_idx = (int)(std::min_element(fitness.begin(),fitness.end()) - fitness.begin());
        result.fitness_history.push_back(fitness[best_idx]);
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    result.best_centroids = pop[best_idx];
    result.best_labels    = labels[best_idx];
    result.best_fitness   = fitness[best_idx];
    result.wall_time_ms   = std::chrono::duration<double,std::milli>(t_end - t_start).count();
    return result;
}
