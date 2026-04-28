cmake_minimum_required(VERSION 3.18)

# ── Project ──────────────────────────────────
project(genetic_kmeans LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD   17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD  17)

# ── Build type ────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ── OpenMP ────────────────────────────────────
find_package(OpenMP REQUIRED)

# ── OpenCV ────────────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ── CUDA architecture ─────────────────────────
# Targets:  sm_60 (Pascal)  sm_70 (Volta)  sm_75 (Turing)
#           sm_80 (Ampere)  sm_86 (Ampere-laptop)  sm_89 (Ada)
# Set CUDA_ARCH to match your GPU, or leave as-is to auto-detect with CMake ≥ 3.24.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ── Include directories ───────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Source files ──────────────────────────────
set(CPP_SOURCES
    src/kmeans.cpp
    src/genetic_kmeans.cpp
    src/parallel_gka.cpp
    src/main.cpp
)

set(CUDA_SOURCES
    cuda/assign_kernel.cu
    cuda/distance_kernel.cu
)

# ── Executable ───────────────────────────────
add_executable(genetic_kmeans
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# ── CUDA flags ────────────────────────────────
set_target_properties(genetic_kmeans PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(genetic_kmeans PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler -Wall
        -Xcompiler -Wextra
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra
    >
)

# ── Link libraries ────────────────────────────
target_link_libraries(genetic_kmeans PRIVATE
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
    cuda
    cudart
)

# ── Install ───────────────────────────────────
install(TARGETS genetic_kmeans DESTINATION bin)

# ── Print configuration summary ───────────────
message(STATUS "=== genetic_kmeans build configuration ===")
message(STATUS "Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard    : ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA arches      : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenMP found     : ${OpenMP_CXX_FOUND}")
message(STATUS "OpenCV version   : ${OpenCV_VERSION}")
message(STATUS "OpenCV libs      : ${OpenCV_LIBS}")
message(STATUS "==========================================")
