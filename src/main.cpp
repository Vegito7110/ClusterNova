/*  main.cpp  –  CLI entry point for the genetic K-means image compressor
 *
 *  Usage
 *  ─────
 *  ./genetic_kmeans --mode   [kmeans | gka_seq | gka_omp | gka_cuda]
 *                   --image  <path/to/image.png>
 *                   --K      <clusters>           (default 16)
 *                   --P      <population>         (default 20)
 *                   --G      <generations>        (default 50)
 *                   --mut    <mutation_rate>      (default 0.1)
 *                   --sigma  <mutation_sigma>     (default 15.0)
 *                   --refine <local_refine_steps> (default 2)
 *                   --threads <num_threads>       (OpenMP, default = all)
 *                   --out    <output_dir>         (default results/)
 *                   --seed   <rng_seed>           (default 42)
 *                   --csv    <benchmark_csv>      (default results/bench.csv)
 *
 *  The program
 *    1. Loads the image with OpenCV.
 *    2. Runs the requested algorithm.
 *    3. Reconstructs the quantised image.
 *    4. Saves it as <out_dir>/<mode>_K<K>_<basename>.png.
 *    5. Appends one row to the CSV:
 *       mode, image, width, height, K, P, G, threads,
 *       wall_ms, ICV, PSNR
 */

#include "kmeans.h"
#include "genetic_kmeans.h"
#include <omp.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────
//  Argument parser
// ─────────────────────────────────────────────

struct Args {
    std::string mode      = "gka_omp";
    std::string image_path;
    int         K         = 16;
    int         P         = 20;
    int         G         = 50;
    double      mut       = 0.10;
    float       sigma     = 15.0f;
    int         refine    = 2;
    int         threads   = 0;   // 0 = use OMP_NUM_THREADS
    std::string out_dir   = "results";
    unsigned    seed      = 42;
    std::string csv_path  = "results/bench.csv";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    std::unordered_map<std::string, std::string> kv;
    for (int i = 1; i < argc - 1; i += 2)
        kv[argv[i]] = argv[i+1];

    if (kv.count("--mode"))    a.mode      = kv["--mode"];
    if (kv.count("--image"))   a.image_path = kv["--image"];
    if (kv.count("--K"))       a.K         = std::stoi(kv["--K"]);
    if (kv.count("--P"))       a.P         = std::stoi(kv["--P"]);
    if (kv.count("--G"))       a.G         = std::stoi(kv["--G"]);
    if (kv.count("--mut"))     a.mut       = std::stod(kv["--mut"]);
    if (kv.count("--sigma"))   a.sigma     = std::stof(kv["--sigma"]);
    if (kv.count("--refine"))  a.refine    = std::stoi(kv["--refine"]);
    if (kv.count("--threads")) a.threads   = std::stoi(kv["--threads"]);
    if (kv.count("--out"))     a.out_dir   = kv["--out"];
    if (kv.count("--seed"))    a.seed      = (unsigned)std::stoul(kv["--seed"]);
    if (kv.count("--csv"))     a.csv_path  = kv["--csv"];

    if (a.image_path.empty())
        throw std::invalid_argument("--image <path> is required.");

    return a;
}

// ─────────────────────────────────────────────
//  Image I/O helpers
// ─────────────────────────────────────────────

static std::vector<Pixel> load_image(const std::string& path,
                                      int& width, int& height) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error("Cannot open image: " + path);

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    width  = img.cols;
    height = img.rows;

    std::vector<Pixel> pixels;
    pixels.reserve((size_t)width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            cv::Vec3b v = img.at<cv::Vec3b>(y, x);
            pixels.push_back({ (float)v[0], (float)v[1], (float)v[2] });
        }
    return pixels;
}

static void save_quantised(const std::string&        path,
                            const std::vector<Pixel>& centroids,
                            const std::vector<int>&   labels,
                            int width, int height) {
    cv::Mat out(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            const Pixel& c = centroids[labels[idx]];
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (uchar)std::clamp((int)c.r, 0, 255),
                (uchar)std::clamp((int)c.g, 0, 255),
                (uchar)std::clamp((int)c.b, 0, 255));
        }
    cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
    cv::imwrite(path, out);
}

// ─────────────────────────────────────────────
//  CSV helpers
// ─────────────────────────────────────────────

static void ensure_csv_header(const std::string& csv) {
    if (!fs::exists(csv)) {
        std::ofstream f(csv);
        f << "mode,image,width,height,K,P,G,threads,"
             "wall_ms,ICV,PSNR\n";
    }
}

static void append_csv(const std::string& csv,
                        const std::string& mode,
                        const std::string& image,
                        int width, int height,
                        int K, int P, int G, int threads,
                        double wall_ms, double icv, double psnr) {
    std::ofstream f(csv, std::ios::app);
    f << mode   << ','
      << fs::path(image).filename().string() << ','
      << width  << ',' << height << ','
      << K << ',' << P << ',' << G << ',' << threads << ','
      << wall_ms << ',' << icv << ',' << psnr << '\n';
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        Args a = parse_args(argc, argv);

        // Load image
        int width, height;
        auto pixels = load_image(a.image_path, width, height);
        int N = (int)pixels.size();
        std::cout << "[INFO] Loaded " << a.image_path
                  << "  (" << width << " × " << height
                  << ", " << N << " pixels)\n";

        // ── Run algorithm ─────────────────────
        std::vector<Pixel> centroids;
        std::vector<int>   labels;
        double             wall_ms = 0, icv = 0;
        int                threads_used = a.threads;

        if (a.mode == "kmeans") {
            auto r = run_kmeans(pixels, a.K, 200, a.seed);
            centroids    = r.centroids;
            labels       = r.labels;
            wall_ms      = r.wall_time_ms;
            icv          = r.intra_cluster_variance;
            threads_used = 1;
            std::cout << "[kmeans]  iters=" << r.iterations << "\n";

        } else if (a.mode == "gka_seq" || a.mode == "gka_cuda") {
            GKAParams p{a.K, a.P, a.G, a.mut, a.sigma, a.refine, a.seed};
            GKAResult r = (a.mode == "gka_seq")
                          ? run_gka_sequential(pixels, p)
                          : run_gka_cuda(pixels, p);
            centroids    = r.best_centroids;
            labels       = r.best_labels;
            wall_ms      = r.wall_time_ms;
            icv          = r.best_fitness;
            threads_used = 1;

        } else { // gka_omp (default)
            GKAParams p{a.K, a.P, a.G, a.mut, a.sigma, a.refine, a.seed};
            GKAResult r = run_gka_openmp(pixels, p, a.threads);
            centroids    = r.best_centroids;
            labels       = r.best_labels;
            wall_ms      = r.wall_time_ms;
            icv          = r.best_fitness;
            // Report actual thread count
#ifdef _OPENMP
            #pragma omp parallel
            { if (omp_get_thread_num() == 0) threads_used = omp_get_num_threads(); }
#endif
        }

        // ── Reconstruct + measure PSNR ────────
        std::vector<Pixel> reconstructed(N);
        for (int i = 0; i < N; ++i)
            reconstructed[i] = centroids[labels[i]];
        double psnr = compute_psnr(pixels, reconstructed);

        std::cout << "[RESULT] mode=" << a.mode
                  << "  K="     << a.K
                  << "  wall="  << wall_ms  << " ms"
                  << "  ICV="   << icv
                  << "  PSNR="  << psnr     << " dB\n";

        // ── Save output image ─────────────────
        fs::create_directories(a.out_dir);
        std::string stem = fs::path(a.image_path).stem().string();
        std::string out_img = a.out_dir + "/" + a.mode
                            + "_K" + std::to_string(a.K)
                            + "_" + stem + ".png";
        save_quantised(out_img, centroids, labels, width, height);
        std::cout << "[INFO] Saved: " << out_img << "\n";

        // ── Append to CSV ─────────────────────
        fs::create_directories(fs::path(a.csv_path).parent_path());
        ensure_csv_header(a.csv_path);
        append_csv(a.csv_path,
                   a.mode, a.image_path, width, height,
                   a.K, a.P, a.G, threads_used,
                   wall_ms, icv, psnr);
        std::cout << "[INFO] Benchmark row appended to " << a.csv_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        std::cerr << "Usage: ./genetic_kmeans --image <img> [--mode kmeans|gka_seq|gka_omp|gka_cuda]"
                     " [--K 16] [--P 20] [--G 50] [--threads 8] [--out results/]\n";
        return 1;
    }
    return 0;
}
