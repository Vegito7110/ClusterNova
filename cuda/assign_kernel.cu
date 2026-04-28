/*  assign_kernel.cu  –  CUDA point-assignment kernel
 *
 *  One thread per pixel.  Each thread finds the nearest centroid and writes
 *  back the label and squared distance.  The host then sums distances to
 *  compute ICV and updates centroids on the CPU.
 *
 *  This file also contains:
 *    • cuda_assign_pixels()   – host wrapper called from run_gka_cuda()
 *    • GKACudaContext         – RAII wrapper that keeps device memory alive
 *      across the whole evolution loop so we don't reallocate every call.
 */

#include "genetic_kmeans.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <limits>

// ─────────────────────────────────────────────
//  Error-checking macro
// ─────────────────────────────────────────────

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess)                                           \
            throw std::runtime_error(std::string("CUDA error: ")         \
                + cudaGetErrorString(err)                                 \
                + " at " __FILE__ ":" + std::to_string(__LINE__));       \
    } while (0)

// ─────────────────────────────────────────────
//  GPU kernel
// ─────────────────────────────────────────────

/**
 * assign_kernel
 *  pixels    : [N × 3]  float  (r,g,b interleaved)
 *  centroids : [K × 3]  float  (in constant / global memory)
 *  labels    : [N]      int    output
 *  dists     : [N]      float  squared-distance output (for ICV)
 *  N, K      : counts
 *
 *  Centroid data is small (K ≤ 256, 3 floats each → ≤ 3 KB) so we
 *  load the entire table into shared memory for fast broadcast.
 */
__global__ void assign_kernel(const float* __restrict__ pixels,
                               const float* __restrict__ centroids,
                               int*   __restrict__ labels,
                               float* __restrict__ dists,
                               int N, int K) {
    // Load centroids into shared memory (one block loads all centroids)
    extern __shared__ float s_centroids[];  // size = K*3 floats

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    for (int j = tid; j < K * 3; j += stride)
        s_centroids[j] = centroids[j];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float pr = pixels[idx * 3 + 0];
    float pg = pixels[idx * 3 + 1];
    float pb = pixels[idx * 3 + 2];

    float best_d = 1e30f;
    int   best_k = 0;

    for (int k = 0; k < K; ++k) {
        float dr = pr - s_centroids[k * 3 + 0];
        float dg = pg - s_centroids[k * 3 + 1];
        float db = pb - s_centroids[k * 3 + 2];
        float d  = dr*dr + dg*dg + db*db;
        if (d < best_d) { best_d = d; best_k = k; }
    }

    labels[idx] = best_k;
    dists[idx]  = best_d;
}

// ─────────────────────────────────────────────
//  RAII device-memory context
// ─────────────────────────────────────────────

struct GKACudaContext {
    float* d_pixels    = nullptr;
    int*   d_labels    = nullptr;
    float* d_dists     = nullptr;
    float* d_centroids = nullptr;
    int    N = 0;
    int    K = 0;

    GKACudaContext(const std::vector<Pixel>& pixels, int K_) : N((int)pixels.size()), K(K_) {
        // Flatten pixel data to SOA-style array [r0,g0,b0, r1,g1,b1, ...]
        std::vector<float> flat(N * 3);
        for (int i = 0; i < N; ++i) {
            flat[i*3+0] = pixels[i].r;
            flat[i*3+1] = pixels[i].g;
            flat[i*3+2] = pixels[i].b;
        }
        CUDA_CHECK(cudaMalloc(&d_pixels,    N * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels,    N     * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dists,     N     * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroids, K * 3 * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_pixels, flat.data(),
                              N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~GKACudaContext() {
        cudaFree(d_pixels);
        cudaFree(d_labels);
        cudaFree(d_dists);
        cudaFree(d_centroids);
    }

    /**
     * Evaluate one chromosome (a set of K centroids).
     * Uploads centroids, launches kernel, downloads labels and ICV.
     */
    double evaluate(const std::vector<Pixel>& centroids_host,
                    std::vector<int>&          labels_host) {
        // Pack centroids
        std::vector<float> c_flat(K * 3);
        for (int k = 0; k < K; ++k) {
            c_flat[k*3+0] = centroids_host[k].r;
            c_flat[k*3+1] = centroids_host[k].g;
            c_flat[k*3+2] = centroids_host[k].b;
        }
        CUDA_CHECK(cudaMemcpy(d_centroids, c_flat.data(),
                              K * 3 * sizeof(float), cudaMemcpyHostToDevice));

        // Launch: 256 threads/block, shared mem = K*3 floats
        const int BLOCK = 256;
        int grid = (N + BLOCK - 1) / BLOCK;
        int smem = K * 3 * sizeof(float);
        assign_kernel<<<grid, BLOCK, smem>>>(d_pixels, d_centroids,
                                              d_labels, d_dists,
                                              N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download labels
        labels_host.resize(N);
        CUDA_CHECK(cudaMemcpy(labels_host.data(), d_labels,
                              N * sizeof(int), cudaMemcpyDeviceToHost));

        // Download distances and compute ICV on host
        std::vector<float> h_dists(N);
        CUDA_CHECK(cudaMemcpy(h_dists.data(), d_dists,
                              N * sizeof(float), cudaMemcpyDeviceToHost));

        double total = 0.0;
        for (float d : h_dists) total += d;
        return total / N;
    }
};

// ─────────────────────────────────────────────
//  Genetic helpers (mirrors genetic_kmeans.cpp)
// ─────────────────────────────────────────────

using Chromosome  = std::vector<Pixel>;
using Population  = std::vector<Chromosome>;
using LabelMatrix = std::vector<std::vector<int>>;

static inline float dist_sq_h(const Pixel& a, const Pixel& b) {
    float dr=a.r-b.r, dg=a.g-b.g, db=a.b-b.b;
    return dr*dr+dg*dg+db*db;
}

static void update_centroids_h(const std::vector<Pixel>& pixels,
                                const std::vector<int>& labels,
                                Chromosome& c,
                                std::mt19937& rng) {
    int K=(int)c.size(), N=(int)pixels.size();
    std::vector<Pixel> acc(K,{0,0,0});
    std::vector<int>   cnt(K,0);
    for (int i=0;i<N;++i){ acc[labels[i]].r+=pixels[i].r; acc[labels[i]].g+=pixels[i].g; acc[labels[i]].b+=pixels[i].b; cnt[labels[i]]++; }
    std::uniform_int_distribution<int> rp(0,N-1);
    for (int k=0;k<K;++k)
        if (cnt[k]>0) c[k]={acc[k].r/cnt[k],acc[k].g/cnt[k],acc[k].b/cnt[k]};
        else c[k]=pixels[rp(rng)];
}

static Chromosome crossover_h(const Chromosome& a, const Chromosome& b, std::mt19937& rng) {
    int K=(int)a.size(); Chromosome child(K);
    std::uniform_int_distribution<int> coin(0,1);
    for (int k=0;k<K;++k) child[k]=coin(rng)?a[k]:b[k];
    return child;
}

static void mutate_h(Chromosome& c, double rate, float sigma, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0,1);
    std::normal_distribution<float> noise(0,sigma);
    for (auto& p:c) if(prob(rng)<(float)rate) {
        p.r=std::clamp(p.r+noise(rng),0.f,255.f);
        p.g=std::clamp(p.g+noise(rng),0.f,255.f);
        p.b=std::clamp(p.b+noise(rng),0.f,255.f);
    }
}

static int tournament_h(const std::vector<double>& f, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(0,(int)f.size()-1);
    int a=d(rng),b=d(rng); return f[a]<f[b]?a:b;
}

static Chromosome init_chromosome_h(const std::vector<Pixel>& pixels, int K, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(0,(int)pixels.size()-1);
    Chromosome c(K); for (auto& p:c) p=pixels[d(rng)]; return c;
}

// ─────────────────────────────────────────────
//  CUDA GKA solver
// ─────────────────────────────────────────────

GKAResult run_gka_cuda(const std::vector<Pixel>& pixels,
                        const GKAParams& params) {
    // Check for a CUDA-capable device; fall back to sequential if none found
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        return run_gka_sequential(pixels, params);
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    const int N = (int)pixels.size();
    const int K = params.K;
    const int P = params.population;
    const int G = params.generations;

    std::mt19937 rng(params.seed);

    // Upload pixel data once
    GKACudaContext ctx(pixels, K);

    Population          pop(P);
    std::vector<double> fitness(P);
    LabelMatrix         labels(P, std::vector<int>(N));

    // ── Initialise population ────────────────
    for (int i = 0; i < P; ++i) {
        pop[i] = init_chromosome_h(pixels, K, rng);
        // Local refine on CPU, then GPU-evaluate
        for (int s = 0; s < params.local_refine_steps; ++s) {
            fitness[i] = ctx.evaluate(pop[i], labels[i]);
            update_centroids_h(pixels, labels[i], pop[i], rng);
        }
        fitness[i] = ctx.evaluate(pop[i], labels[i]);
    }

    int best_idx = (int)(std::min_element(fitness.begin(),fitness.end())-fitness.begin());

    GKAResult result;
    result.fitness_history.reserve(G);

    // ── Evolution loop ───────────────────────
    for (int gen = 0; gen < G; ++gen) {
        Population         children(P);
        std::vector<double> new_fit(P);
        LabelMatrix         new_lbl(P, std::vector<int>(N));

        children[0] = pop[best_idx];
        new_fit[0]  = fitness[best_idx];
        new_lbl[0]  = labels[best_idx];

        for (int i = 1; i < P; ++i) {
            int p1 = tournament_h(fitness, rng);
            int p2 = tournament_h(fitness, rng);
            children[i] = crossover_h(pop[p1], pop[p2], rng);
            mutate_h(children[i], params.mutation_rate, params.mutation_sigma, rng);

            // CPU local refine + GPU evaluate
            for (int s = 0; s < params.local_refine_steps; ++s) {
                new_fit[i] = ctx.evaluate(children[i], new_lbl[i]);
                update_centroids_h(pixels, new_lbl[i], children[i], rng);
            }
            new_fit[i] = ctx.evaluate(children[i], new_lbl[i]);
        }

        pop     = std::move(children);
        fitness = std::move(new_fit);
        labels  = std::move(new_lbl);

        best_idx = (int)(std::min_element(fitness.begin(),fitness.end())-fitness.begin());
        result.fitness_history.push_back(fitness[best_idx]);
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    result.best_centroids = pop[best_idx];
    result.best_labels    = labels[best_idx];
    result.best_fitness   = fitness[best_idx];
    result.wall_time_ms   = std::chrono::duration<double,std::milli>(t_end-t_start).count();
    return result;
}
