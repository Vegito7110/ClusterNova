/*  distance_kernel.cu  –  Batch pairwise distance computation
 *
 *  Used when you need the full N × K distance matrix, e.g. for a
 *  soft-assignment variant or for debugging / profiling the raw
 *  memory-bandwidth achievable on your GPU.
 *
 *  The main GKA pipeline uses assign_kernel.cu (winner-take-all),
 *  which is faster.  This kernel is provided for completeness and
 *  can be switched in via the --kernel flag.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────
//  Error helper
// ─────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess)                                             \
            throw std::runtime_error(std::string("CUDA error: ")           \
                + cudaGetErrorString(err)                                   \
                + " at " __FILE__ ":" + std::to_string(__LINE__));         \
    } while (0)

// ─────────────────────────────────────────────
//  Tiled distance kernel
// ─────────────────────────────────────────────
//  Grid  : (N/TILE_N, K/TILE_K)
//  Block : (TILE_N,   TILE_K)
//
//  Each thread computes dist(pixel[row], centroid[col]).
//  Pixels and centroids are tiled into shared memory.

constexpr int TILE = 16;

/**
 * distance_kernel
 *  pixels    : [N × 3] floats
 *  centroids : [K × 3] floats
 *  out_dists : [N × K] floats  (row = pixel, col = centroid)
 *  N, K
 */
__global__ void distance_kernel(const float* __restrict__ pixels,
                                 const float* __restrict__ centroids,
                                 float*       __restrict__ out_dists,
                                 int N, int K) {
    __shared__ float s_pix[TILE][3];
    __shared__ float s_cen[TILE][3];

    int row = blockIdx.x * TILE + threadIdx.x; // pixel index
    int col = blockIdx.y * TILE + threadIdx.y; // centroid index

    // Cooperatively load tiles
    if (threadIdx.y == 0 && row < N) {
        s_pix[threadIdx.x][0] = pixels[row * 3 + 0];
        s_pix[threadIdx.x][1] = pixels[row * 3 + 1];
        s_pix[threadIdx.x][2] = pixels[row * 3 + 2];
    }
    if (threadIdx.x == 0 && col < K) {
        s_cen[threadIdx.y][0] = centroids[col * 3 + 0];
        s_cen[threadIdx.y][1] = centroids[col * 3 + 1];
        s_cen[threadIdx.y][2] = centroids[col * 3 + 2];
    }
    __syncthreads();

    if (row >= N || col >= K) return;

    float dr = s_pix[threadIdx.x][0] - s_cen[threadIdx.y][0];
    float dg = s_pix[threadIdx.x][1] - s_cen[threadIdx.y][1];
    float db = s_pix[threadIdx.x][2] - s_cen[threadIdx.y][2];

    out_dists[row * K + col] = dr*dr + dg*dg + db*db;
}

// ─────────────────────────────────────────────
//  Host wrapper
// ─────────────────────────────────────────────

/**
 * Compute the full N×K distance matrix on the GPU.
 *
 * pixels_flat    : row-major [N × 3] floats on host
 * centroids_flat : row-major [K × 3] floats on host
 * out            : row-major [N × K] floats on host (output)
 */
void cuda_distance_matrix(const std::vector<float>& pixels_flat,
                           const std::vector<float>& centroids_flat,
                           std::vector<float>&        out,
                           int N, int K) {
    float *d_pix = nullptr, *d_cen = nullptr, *d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pix, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cen, K * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pix, pixels_flat.data(),    N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cen, centroids_flat.data(), K*3*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (K + TILE - 1) / TILE);
    distance_kernel<<<grid, block>>>(d_pix, d_cen, d_out, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    out.resize((size_t)N * K);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, (size_t)N*K*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_pix);
    cudaFree(d_cen);
    cudaFree(d_out);
}
