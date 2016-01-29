// MAPS - Memory Access Pattern Specification Framework
// http://maps-gpu.github.io/
// Copyright (c) 2015, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>

#include <vector>
#include <map>
#include <memory>

#include <gflags/gflags.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "maps/maps.cuh"
#include "maps/multi/multi.cuh"

DEFINE_int32(width,  4096, "Image width");
DEFINE_int32(height, 2560, "Image height");

DEFINE_bool(multithreading, true, "Run a thread per device");
DEFINE_bool(regression, true, "Perform regression tests");
DEFINE_bool(print_values, false, "Print unequal values");

#define BINS 256

#define ITEMS_PER_THREAD_X 4

DEFINE_int32(block_width, 256, "Block width to use");
DEFINE_int32(block_height, 1, "Block height to use");
DEFINE_int32(items_per_thread, ITEMS_PER_THREAD_X, "Number of items per thread to use");
DEFINE_bool(exhaustive, false, "Test all block size / items per thread configurations");

#ifdef _DEBUG
DEFINE_int32(repetitions, 5, "Number of iterations for test");
#else
DEFINE_int32(repetitions, 50, "Number of iterations for test");
#endif

DEFINE_bool(cpu_regression, true, "Perform regression tests against CPU");
DEFINE_int32(random_seed, -1, "Override random seed (default is current time)");
unsigned int curtime = (unsigned int)time(NULL);

void Histogram_CPU(const unsigned char *in, size_t inStride, unsigned int *out, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            unsigned char val = in[inStride * i + j];
            
            out[val]++;
        }
    }
}

template<int BLOCK_WIDTH, int ITEMS_PER_THREAD>
__global__ void HistogramMMAPS MAPS_MULTIDEF(maps::Window2D<uint8_t, BLOCK_WIDTH, 1, 0, maps::ZeroBoundaries, ITEMS_PER_THREAD> in, 
                                             maps::ReductiveStaticOutput<unsigned int, BINS, BLOCK_WIDTH, ITEMS_PER_THREAD> out)
{
    MAPS_MULTI_INITVARS(in, out);

    #pragma unroll
    MAPS_FOREACH(oiter, out)
    {
        uint8_t value = *in.align(oiter);
        oiter[value]++;
    }

    out.commit();
}


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

bool HistogramCPURegression(const unsigned int *otherResult)
{
    if (!FLAGS_regression)
        return true;

    printf("Comparing with CPU...\n");

    size_t width = FLAGS_width, height = FLAGS_height;

    unsigned int host_hist[BINS] = { 0 };
    maps::pinned_vector<unsigned char> host_image(width * height, 0);

    srand((FLAGS_random_seed < 0) ? curtime : FLAGS_random_seed);

    for (size_t i = 0; i < width * height; ++i)
        host_image[i] = (rand() % BINS);
    
    Histogram_CPU(&host_image[0], sizeof(unsigned char) * width, host_hist, (int)width, (int)height);

    int numErrors = 0;
    for (size_t i = 0; i < BINS; ++i)
    {
        if (host_hist[i] != otherResult[i])
        {
            if (FLAGS_print_values)
                printf("ERROR AT INDEX %d: real: %d, other: %d\n", (int)i, 
                       (int)host_hist[i], (int)otherResult[i]);
            numErrors++;
        }
    }
    
    printf("Comparison %s: Errors: %d\n\n", (numErrors == 0) ? "OK" : "FAILED", numErrors);

    return (numErrors == 0);
}

template<int BLOCK_WIDTH, int ITEMS_PER_THREAD>
bool TestHistogramMultiGPU(int ngpus)
{
    size_t width = FLAGS_width, height = FLAGS_height;

    if (width % ITEMS_PER_THREAD != 0)
    {
        printf("Width (%d) not multiple of items per thread (%d), skipping...\n", (int)width, ITEMS_PER_THREAD);
        return true;
    }
    if (ITEMS_PER_THREAD * BLOCK_WIDTH > width)
    {
        printf("ITEMS PER THREAD * BLOCK WIDTH is too wide, skipping...\n");
        return true;
    }

    unsigned int host_hist[BINS] = { 0 };
    maps::pinned_vector<unsigned char> host_image(width * height, 0);

    srand((FLAGS_random_seed < 0) ? curtime : FLAGS_random_seed);

    for (size_t i = 0; i < width * height; ++i)
        host_image[i] = (rand() % BINS);

    // Create GPU list
    int num_gpus;
    MAPS_CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    std::vector<unsigned int> gpuids;
    for (int i = 0; i < ngpus; ++i)
        gpuids.push_back(i % num_gpus);

    // Create scheduler
    maps::multi::Scheduler sched(gpuids);

    if (!FLAGS_multithreading)
        sched.DisableMultiThreading();

    // Create data
    maps::multi::Matrix<unsigned char> image(width, height);
    maps::multi::Vector<unsigned int> hist((size_t)BINS);

    dim3 block_dims(BLOCK_WIDTH, FLAGS_block_height, 1);
    dim3 grid_dims(maps::RoundUp((int)width / ITEMS_PER_THREAD, block_dims.x), maps::RoundUp((int)height, block_dims.y), 1);

    maps::multi::AnalyzeCall(sched, grid_dims, block_dims,
                             maps::multi::Window2D<uint8_t, BLOCK_WIDTH, 1, 0, maps::ZeroBoundaries, ITEMS_PER_THREAD>(image),
                             maps::multi::ReductiveStaticOutput<unsigned int, BINS, BLOCK_WIDTH, ITEMS_PER_THREAD>(hist));

    image.Bind(&host_image[0], sizeof(unsigned char) * width);
    hist.Bind(host_hist, BINS * sizeof(unsigned int));

    maps::multi::Fill(sched, hist);

    // Clear histogram and invalidate buffer
    maps::multi::Fill(sched, hist);
    sched.Invalidate(image);

    for (int i = 0; i < num_gpus; i++)
    {
        MAPS_CUDA_CHECK(cudaSetDevice(i));
        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }
    MAPS_CUDA_CHECK(cudaSetDevice(0));    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_repetitions; i++)
    {
        maps::multi::Invoke(sched, HistogramMMAPS<BLOCK_WIDTH, ITEMS_PER_THREAD>, grid_dims, block_dims,
                            maps::multi::Window2D<unsigned char, BLOCK_WIDTH, 1, 0, maps::ZeroBoundaries, ITEMS_PER_THREAD>(image),
                            maps::multi::ReductiveStaticOutput<unsigned int, BINS, BLOCK_WIDTH, ITEMS_PER_THREAD>(hist));
    }

    sched.WaitAll();
    for (int i = 0; i < num_gpus; i++)
    {
        MAPS_CUDA_CHECK(cudaSetDevice(i));
        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    printf("Histogram (MAPS-Multi): %f ms\n",
           std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_repetitions);

    // Copy output vectors from GPUs to host memory, aggregating information in the process
    maps::multi::Gather(sched, hist);

    // Account for histogram being incremental (the values have been incremented REPETITIONS times)
    for (int i = 0; i < BINS; ++i)
        host_hist[i] /= FLAGS_repetitions;

    return HistogramCPURegression(host_hist);
}

#define TEST(blockwidth, items) do { \
    if(FLAGS_block_width == blockwidth && FLAGS_items_per_thread == items) \
        return TestHistogramMultiGPU<blockwidth, items>(ngpus);\
    if(FLAGS_exhaustive)\
        overall &= TestHistogramMultiGPU<blockwidth, items>(ngpus);\
} while(0)

#define TESTALL()   do {        \
    TEST(16, 1);              \
    TEST(16, 2);              \
    TEST(16, 4);              \
    TEST(16, 8);              \
    TEST(16, 16);             \
    TEST(32, 1);              \
    TEST(32, 2);              \
    TEST(32, 4);              \
    TEST(32, 8);              \
    TEST(32, 16);             \
    TEST(64, 1);              \
    TEST(64, 2);              \
    TEST(64, 4);              \
    TEST(64, 8);              \
    TEST(64, 16);             \
    TEST(128, 1);             \
    TEST(128, 2);             \
    TEST(128, 4);             \
    TEST(128, 8);             \
    TEST(128, 16);            \
    TEST(192, 1);             \
    TEST(192, 2);             \
    TEST(192, 4);             \
    TEST(192, 8);             \
    TEST(192, 16);            \
    TEST(256, 1);             \
    TEST(256, 2);             \
    TEST(256, 4);             \
    TEST(256, 8);             \
    TEST(256, 16);            \
    TEST(512, 1);             \
    TEST(512, 2);             \
    TEST(512, 4);             \
    TEST(512, 8);             \
    TEST(512, 16);            \
} while(0)

bool TestHistogramMAPSMulti(int ngpus)
{
    bool overall = true;

    TESTALL();

    if (!FLAGS_exhaustive)
    {
        printf("Test not compiled!\n");
        overall = false;
    }
    return overall;
}
