#include "kmeans.h"
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

static inline float dist_sq(const Pixel& a, const Pixel& b) {
    float dr = a.r - b.r, dg = a.g - b.g, db = a.b - b.b;
    return dr*dr + dg*dg + db*db;
}

double intra_cluster_variance(const std::vector<Pixel>&  pixels,
                               const std::vector<int>&    labels,
                               const std::vector<Pixel>&  centroids,
                               int K) {
    double total = 0.0;
    for (size_t i = 0; i < pixels.size(); ++i)
        total += dist_sq(pixels[i], centroids[labels[i]]);
    return total / static_cast<double>(pixels.size());
}

double compute_psnr(const std::vector<Pixel>& original,
                    const std::vector<Pixel>& compressed) {
    double mse = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double dr = original[i].r - compressed[i].r;
        double dg = original[i].g - compressed[i].g;
        double db = original[i].b - compressed[i].b;
        mse += (dr*dr + dg*dg + db*db) / 3.0;
    }
    mse /= static_cast<double>(original.size());
    if (mse < 1e-10) return 100.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// ─────────────────────────────────────────────
//  K-means++ seeding
// ─────────────────────────────────────────────

static std::vector<Pixel> kmeanspp_init(const std::vector<Pixel>& pixels,
                                         int K,
                                         std::mt19937& rng) {
    int N = static_cast<int>(pixels.size());
    std::uniform_int_distribution<int> uni(0, N - 1);
    std::vector<Pixel> centres;
    centres.reserve(K);
    centres.push_back(pixels[uni(rng)]);

    std::vector<float> d2(N, std::numeric_limits<float>::max());

    for (int k = 1; k < K; ++k) {
        // Update min-distances to closest existing centre
        for (int i = 0; i < N; ++i)
            d2[i] = std::min(d2[i], dist_sq(pixels[i], centres.back()));

        std::discrete_distribution<int> weighted(d2.begin(), d2.end());
        centres.push_back(pixels[weighted(rng)]);
    }
    return centres;
}

// ─────────────────────────────────────────────
//  K-means main loop
// ─────────────────────────────────────────────

KMeansResult run_kmeans(const std::vector<Pixel>& pixels,
                        int K,
                        int max_iter,
                        unsigned int seed) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int N = static_cast<int>(pixels.size());
    std::mt19937 rng(seed);

    std::vector<Pixel> centroids = kmeanspp_init(pixels, K, rng);
    std::vector<int>   labels(N, 0);
    int iter = 0;
    bool changed = true;

    while (changed && iter < max_iter) {
        changed = false;

        // ── Assignment step ──────────────────
        for (int i = 0; i < N; ++i) {
            float best = std::numeric_limits<float>::max();
            int   best_k = 0;
            for (int k = 0; k < K; ++k) {
                float d = dist_sq(pixels[i], centroids[k]);
                if (d < best) { best = d; best_k = k; }
            }
            if (labels[i] != best_k) { labels[i] = best_k; changed = true; }
        }

        // ── Update step ──────────────────────
        std::vector<Pixel> acc(K, {0.f, 0.f, 0.f});
        std::vector<int>   cnt(K, 0);
        for (int i = 0; i < N; ++i) {
            int c = labels[i];
            acc[c].r += pixels[i].r;
            acc[c].g += pixels[i].g;
            acc[c].b += pixels[i].b;
            cnt[c]++;
        }
        for (int k = 0; k < K; ++k) {
            if (cnt[k] > 0) {
                centroids[k] = { acc[k].r / cnt[k],
                                 acc[k].g / cnt[k],
                                 acc[k].b / cnt[k] };
            }
            // If a cluster is empty we simply leave its centroid unchanged.
            // In practice K-means++ seeding makes this extremely rare.
        }
        ++iter;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double icv = intra_cluster_variance(pixels, labels, centroids, K);
    double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return { centroids, labels, icv, iter, ms };
}
