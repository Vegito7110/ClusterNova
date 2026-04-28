# Genetic K-means Image Compression
### Parallel implementation using OpenMP + CUDA

A complete pipeline for colour-quantisation-based image compression using a
Genetic K-means Algorithm (GKA).  Three solver variants are provided and
benchmarked against a standard K-means baseline.

---

## Project Layout

```
genetic_kmeans/
├── src/
│   ├── kmeans.cpp          Standard K-means baseline (K-means++ init)
│   ├── genetic_kmeans.cpp  Sequential GKA core
│   ├── parallel_gka.cpp    OpenMP population-loop parallelisation
│   └── main.cpp            CLI entry point
├── include/
│   ├── kmeans.h
│   └── genetic_kmeans.h
├── cuda/
│   ├── assign_kernel.cu    Point-assignment GPU kernel (one thread / pixel)
│   └── distance_kernel.cu  Tiled N×K distance batch kernel
├── data/                   Benchmark images (place 512.png … 4096.png here)
├── results/                Compressed images + bench.csv (auto-created)
├── scripts/
│   ├── benchmark.sh        Full sweep script
│   └── plot_results.py     Figure generator for the report
├── CMakeLists.txt
└── README.md
```

---

## Prerequisites

| Dependency | Minimum version | Notes |
|---|---|---|
| CMake | 3.18 | |
| GCC / Clang | GCC 9 / Clang 10 | C++17 required |
| CUDA Toolkit | 11.0 | Only needed for `gka_cuda` mode |
| OpenMP | 4.5 | Usually bundled with GCC |
| OpenCV | 4.x | For PNG I/O |

### Install OpenCV on Ubuntu
```bash
sudo apt update
sudo apt install libopencv-dev
```

---

## Build

```bash
# 1. Clone / enter the project root
cd genetic_kmeans

# 2. Create build directory
mkdir build && cd build

# 3. Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

#    Optional: target only your GPU (saves compile time)
#    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86

# 4. Compile  (use -j$(nproc) for parallel build)
make -j$(nproc)
```

The binary is `build/genetic_kmeans`.

---

## Usage

```
./genetic_kmeans  --image  <path>
                  --mode   kmeans | gka_seq | gka_omp | gka_cuda
                  --K      <clusters>            (default 16)
                  --P      <population size>     (default 20)
                  --G      <generations>         (default 50)
                  --mut    <mutation rate>        (default 0.10)
                  --sigma  <mutation std-dev px>  (default 15.0)
                  --refine <local K-means steps> (default 2)
                  --threads <OMP threads>        (default = OMP_NUM_THREADS)
                  --out    <output dir>           (default results/)
                  --seed   <rng seed>             (default 42)
                  --csv    <benchmark CSV>        (default results/bench.csv)
```

### Quick examples

```bash
# Standard K-means baseline
./build/genetic_kmeans --mode kmeans --image data/512.png --K 16

# Sequential GKA
./build/genetic_kmeans --mode gka_seq --image data/512.png --K 16 --P 20 --G 50

# OpenMP GKA with 8 threads
./build/genetic_kmeans --mode gka_omp --image data/512.png --K 16 --threads 8

# CUDA GKA
./build/genetic_kmeans --mode gka_cuda --image data/512.png --K 16
```

---

## Benchmark sweep

```bash
# Place images in data/:  512.png  1024.png  2048.png  4096.png
# Then run:
chmod +x scripts/benchmark.sh
./scripts/benchmark.sh
```

Results are appended to `results/bench.csv` with columns:

```
mode, image, width, height, K, P, G, threads, wall_ms, ICV, PSNR
```

### Generate all report figures

```bash
pip install pandas matplotlib seaborn
python scripts/plot_results.py --csv results/bench.csv --out results/plots/
```

---

## Recommended benchmark images

| Image | Where to get |
|---|---|
| Lena / Baboon (classic) | USC-SIPI image database (sipi.usc.edu) |
| Kodak set (24 photos) | r0k.us/graphics/kodak/ |
| DIV2K high-res | data.vision.ee.ethz.ch/cvl/DIV2K/ |

**Resize to exact sizes** with:

```bash
convert source.jpg -resize 512x512!  data/512.png
convert source.jpg -resize 1024x1024! data/1024.png
convert source.jpg -resize 2048x2048! data/2048.png
convert source.jpg -resize 4096x4096! data/4096.png
```

Choose **colour-rich, natural photos** (not cartoons or diagrams).
High-colour-diversity images stress the algorithm most and produce the
most discriminating PSNR differences between modes.

---

## Algorithm overview

```
Population of P chromosomes, each = K centroids (an RGB colour palette)

Per generation:
  ① Elitism:   carry forward the best chromosome unchanged
  ② Selection: binary tournament (size 2)
  ③ Crossover: uniform (each centroid from parent A or B, p = 0.5)
  ④ Mutation:  Gaussian perturbation of each centroid (rate, σ)
  ⑤ Refine:    1-2 K-means update steps per child
  ⑥ Evaluate: ICV = mean squared pixel-to-centroid distance

CUDA variant: step ⑥ runs on GPU (one thread per pixel).
OpenMP variant: the P fitness evaluations in ⑥ run on T CPU threads.
```

---

## Key parameters and their effect

| Parameter | Typical range | Effect |
|---|---|---|
| K | 8 – 64 | Colour palette size; higher K = better PSNR, more time |
| P (population) | 10 – 50 | Larger P explores more; linear cost |
| G (generations) | 30 – 100 | More generations = lower ICV; diminishing returns after ~50 |
| mutation_rate | 0.05 – 0.20 | Higher = more diversity, slower convergence |
| mutation_sigma | 10 – 25 px | Perturbation step size; 15 works well for 0-255 range |
| local_refine | 1 – 3 | More steps exploit local optima; >3 rarely helps |

---

## What to measure and report

### Primary metrics
- **Wall time (ms)** — total algorithm runtime
- **ICV** — intra-cluster variance (fitness); lower = better clustering
- **PSNR (dB)** — image quality vs original; higher = better

### Experiments to run

1. **Quality vs K** — fix mode=gka_omp, vary K ∈ {8,16,32,64}.  
   Plot: PSNR vs K (expect ~logarithmic growth).

2. **OMP scalability** — fix image=1024.png, K=16, vary threads ∈ {1,2,4,8,16}.  
   Plot: speedup = t_seq / t_omp(T) vs T.  
   Expected: near-linear up to physical core count, then plateau.

3. **CUDA vs OMP** — compare best-OMP vs CUDA wall time per (image,K).  
   CUDA wins on large images (>1024²); OMP wins for small images.

4. **GKA vs K-means quality** — compare ICV/PSNR for kmeans vs gka modes.  
   GKA should escape local optima → lower ICV / higher PSNR.

5. **Scalability heatmap** — speedup across all (K, image_size) pairs.

---

## Drawing conclusions for your report

### Parallelism efficiency
- Compute **parallel efficiency** = speedup / T.  
  E.g. speedup=6.2 on 8 threads → efficiency = 77.5 %.
- If efficiency drops sharply at T > physical cores, attribute to hyperthreading overhead.

### CUDA crossover point
- CUDA overhead dominates for small N (≤ 512²).  CUDA wins when N ≥ 1024².  
  Report the crossover image size explicitly.

### Quality improvement
- Express PSNR gain: "GKA achieves X dB higher PSNR than K-means at K=16."
- Show side-by-side compressed images (already saved in results/).

### Amdahl's law check
- The serial fraction s (selection + crossover) limits max speedup.  
  If s ≈ 5 %, max speedup ≈ 1/0.05 = 20×.  Measure s by profiling or timing the serial vs parallel sections separately.

### Convergence
- If you log ICV per generation (fitness_history), plot the convergence curve.  
  A flat tail means you can reduce G; a still-dropping curve means more generations help.

---

## License
MIT
