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

/*
Expected Results
----------------
GPU Model   |  Naive  |  MAPS  |  MAPST  | MAPST-ILP
------------+---------+--------+---------|-----------
TITAN BLACK | 348 us  |  88 us |  79 us  |  TBD us
GTX 680     | 500 us  | 141 us | 131 us  |  TBD us
GTX 730M    | 2855 us | 967 us | 748 us  |  343 us
*/

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "cuda_gtest_utils.h"
#include <device_launch_parameters.h>

#include <maps/maps.cuh>
#include <maps/input_containers/internal/io_globalread.cuh>
#include <maps/input_containers/internal/io_globaltoshared.cuh>
#include <maps/input_containers/internal/io_globaltoarray.cuh>
#include <maps/multi/multi.cuh>

#define IMAGE_WIDTH 608
#define IMAGE_HEIGHT 608

#define BW 32
#define BH 32

// Using smaller block sizes to accomodate for memory loads
#define BWILP 16
#define BHILP 16
#define IPX 4
#define IPY 2

#define REPETITIONS 1000

// Unique ID for conv2 input image texture (required for working with maps::Window2DTexture)
#define IMAGE_TEXTURE_UID 1111

#define KERNEL_RADIUS 4

__constant__ float dev_convKernel[(2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1)];

// Simple convolution kernel
float g_convKernel[(2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1)] =
{
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    2.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    3.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    4.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    5.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    6.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    7.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
    8.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f
};

template<int RADIUS>
__global__ void conv2Naive(const float *in, size_t inStride,
                           float *out, size_t outStride,
                           int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float result = 0.0f;

    for (int ky = 0; ky <= 2 * RADIUS; ky++)
        for (int kx = 0; kx <= 2 * RADIUS; kx++)
            result += in[(y - RADIUS + ky) * inStride + (x - RADIUS + kx)] * dev_convKernel[ky * (2 * RADIUS + 1) + kx];

    out[y * outStride + x] = result;
}

template<int RADIUS, int BLOCK_WIDTH, int BLOCK_HEIGHT, int TEXTURE_UID = -1, int ILP_X = 1, int ILP_Y = 1>
__global__ void conv2MAPS(maps::WindowSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, RADIUS, ILP_X, ILP_Y, 1, maps::WB_NOCHECKS, TEXTURE_UID> in,
                          maps::StructuredInjectiveSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> out)
{
    MAPS_INIT(in, out);

    if (out.Items() == 0)
        return;

    #pragma unroll
    MAPS_FOREACH(oiter, out)
    {
        *oiter = 0.0f;
        int i = 0;

        #pragma unroll
        MAPS_FOREACH_ALIGNED(iter, in, oiter)
        {
            *oiter += *iter * dev_convKernel[i++];
        }
    }

    out.commit();
}

TEST(Performance, Window2D_Convolution)
{
#ifndef NDEBUG
    printf("Debug mode detected, skipping test\n");
    return;
#endif

    float *dev_image = NULL, *dev_naiveResult = NULL,
        *dev_MAPSResult = NULL, *dev_MAPSTexResult = NULL, *dev_MAPSILPResult = NULL;
    size_t width = IMAGE_WIDTH, height = IMAGE_HEIGHT, imageStride = 0;

    // Create input data
    std::vector<float> host_image(width * height, 0);
    for (size_t i = 0; i < width * height; ++i)
        host_image[i] = static_cast<float>(i % width);

    // Allocate GPU buffers
    CUASSERT_NOERR(cudaMallocPitch(&dev_image, &imageStride, sizeof(float) * width, height));
    CUASSERT_NOERR(cudaMalloc(&dev_naiveResult, sizeof(float) * width * height));
    CUASSERT_NOERR(cudaMalloc(&dev_MAPSResult, sizeof(float) * width * height));
    CUASSERT_NOERR(cudaMalloc(&dev_MAPSTexResult, sizeof(float) * width * height));
    CUASSERT_NOERR(cudaMalloc(&dev_MAPSILPResult, sizeof(float) * width * height));

    // Create GPU texture

    // Set texture parameters
    typedef typename maps::UniqueTexRef2D<float>::template TexId<IMAGE_TEXTURE_UID> TexId;

    TexId::tex.addressMode[0] = cudaAddressModeClamp;
    TexId::tex.addressMode[1] = cudaAddressModeClamp;
    TexId::tex.filterMode = cudaFilterModeLinear;

    // Copy and compare the results
    std::vector<float> host_resultNaive(width * height, 0), host_resultMAPS(width * height, 0),
        host_resultMAPSTex(width * height, 0), host_resultMAPSILP(width * height, 0);

    // Bind texture to data
    CUASSERT_NOERR(TexId::BindTexture(dev_image, width, height, imageStride));

    dim3 block_dims(BW, BH, 1);
    dim3 grid_dims(maps::RoundUp(width, block_dims.x), maps::RoundUp(height, block_dims.y), 1);
    
    dim3 block_dims_ilp(BWILP, BHILP, 1);
    dim3 grid_dims_ilp(maps::RoundUp(width, block_dims_ilp.x * IPX), maps::RoundUp(height, block_dims_ilp.y * IPY), 1);

    // Copy input data to GPU
    CUASSERT_NOERR(cudaMemcpyToSymbolAsync(dev_convKernel, g_convKernel,
        sizeof(float)* (2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1),
        0, cudaMemcpyHostToDevice));
    CUASSERT_NOERR(cudaMemcpy2DAsync(dev_image, imageStride, &host_image[0], sizeof(float)* width,
        sizeof(float)* width, height, cudaMemcpyHostToDevice));


    cudaDeviceSynchronize();
    auto nt1 = std::chrono::high_resolution_clock::now();

    // Run all three versions
    for (int i = 0; i < REPETITIONS; i++)
    {
        conv2Naive<KERNEL_RADIUS> <<<grid_dims, block_dims>>>(dev_image, imageStride / sizeof(float),
                                                              dev_naiveResult, width,
                                                              width, height);
    }

    cudaDeviceSynchronize();
    auto nt2 = std::chrono::high_resolution_clock::now();

    CUASSERT_NOERR(cudaMemcpy(&host_resultNaive[0], dev_naiveResult, sizeof(float)* width * height, cudaMemcpyDeviceToHost));

    maps::WindowSingleGPU<float, 2, BW, BH, 1, KERNEL_RADIUS> win;
    win.m_ptr = dev_image;
    win.m_stride = imageStride / sizeof(float);
    win.m_dimensions[0] = width; win.m_dimensions[1] = height;

    maps::StructuredInjectiveSingleGPU<float, 2, BW, BH, 1> soout;
    soout.m_ptr = dev_MAPSResult;
    soout.m_stride = width;
    soout.m_dimensions[0] = width; soout.m_dimensions[1] = height;

    cudaDeviceSynchronize();
    auto mt1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPETITIONS; i++)
    {
        conv2MAPS<KERNEL_RADIUS, BW, BH> <<<grid_dims, block_dims>>>(win, soout);
    }

    cudaDeviceSynchronize();
    auto mt2 = std::chrono::high_resolution_clock::now();


    CUASSERT_NOERR(cudaMemcpy(&host_resultMAPS[0], dev_MAPSResult, sizeof(float)* width * height, cudaMemcpyDeviceToHost));

    soout.m_ptr = dev_MAPSTexResult;

    maps::WindowSingleGPU<float, 2, BW, BH, 1, KERNEL_RADIUS, 1, 1, 1, maps::WB_NOCHECKS, IMAGE_TEXTURE_UID> wintex;
    wintex.m_ptr = dev_image;
    wintex.m_stride = imageStride / sizeof(float);
    wintex.m_dimensions[0] = width; wintex.m_dimensions[1] = height;

    cudaDeviceSynchronize();
    auto mtt1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPETITIONS; i++)
    {
        conv2MAPS<KERNEL_RADIUS, BW, BH, IMAGE_TEXTURE_UID> <<<grid_dims, block_dims>>>(wintex, soout);
    }
    
    CUASSERT_NOERR(cudaDeviceSynchronize());
    auto mtt2 = std::chrono::high_resolution_clock::now();

    CUASSERT_NOERR(cudaMemcpy(&host_resultMAPSTex[0], dev_MAPSTexResult, sizeof(float)* width * height, cudaMemcpyDeviceToHost));

    maps::WindowSingleGPU<float, 2, BWILP, BHILP, 1, KERNEL_RADIUS, IPX, IPY, 1, maps::WB_NOCHECKS, IMAGE_TEXTURE_UID> wintexilp;
    wintexilp.m_ptr = dev_image;
    wintexilp.m_stride = imageStride / sizeof(float);
    wintexilp.m_dimensions[0] = width; wintexilp.m_dimensions[1] = height;

    maps::StructuredInjectiveSingleGPU<float, 2, BWILP, BHILP, 1, IPX, IPY> sooutilp;
    sooutilp.m_ptr = dev_MAPSILPResult;
    sooutilp.m_stride = width;
    sooutilp.m_dimensions[0] = width; sooutilp.m_dimensions[1] = height;

    cudaDeviceSynchronize();
    auto mit1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPETITIONS; i++)
    {
        conv2MAPS<KERNEL_RADIUS, BWILP, BHILP, IMAGE_TEXTURE_UID, IPX, IPY> <<<grid_dims_ilp, block_dims_ilp>>>(wintexilp, sooutilp);
    }


    CUASSERT_NOERR(cudaDeviceSynchronize());
    auto mit2 = std::chrono::high_resolution_clock::now();
    
    CUASSERT_NOERR(TexId::UnbindTexture());

    CUASSERT_NOERR(cudaMemcpy(&host_resultMAPSILP[0], dev_MAPSILPResult, sizeof(float)* width * height, cudaMemcpyDeviceToHost));

    int numErrorsMAPS = 0, numErrorsMAPSTex = 0, numErrorsMAPSILP = 0;
    float meanErrorMAPS = 0.0f, meanErrorMAPSTex = 0.0f, meanErrorMAPSILP = 0.0f;

    // Do not compare the results in the outer borders
    for (size_t y = KERNEL_RADIUS; y < height - KERNEL_RADIUS; ++y)
    {
        for (size_t x = KERNEL_RADIUS; x < width - KERNEL_RADIUS; ++x)
        {
            // Test Naive vs. MAPS
            if (fabs(host_resultNaive[y * width + x] - host_resultMAPS[y * width + x]) > 1e-6)
            {
                if (numErrorsMAPS == 0)
                    printf("MAPS: First error in (%d, %d): %f != %f\n", x, y,
                           host_resultNaive[y * width + x], host_resultMAPS[y * width + x]);

                numErrorsMAPS++;
            }
            meanErrorMAPS += fabs(host_resultNaive[y * width + x] - host_resultMAPS[y * width + x]);

            // Test Naive vs. MAPS (Texture)
            if (fabs(host_resultNaive[y * width + x] - host_resultMAPSTex[y * width + x]) > 1e-6)
            {
                if (numErrorsMAPSTex == 0)
                    printf("MAPS(Texture): First error in (%d, %d): %f != %f\n", x, y,
                           host_resultNaive[y * width + x], host_resultMAPSTex[y * width + x]);

                numErrorsMAPSTex++;
            }
            meanErrorMAPSTex += fabs(host_resultNaive[y * width + x] - host_resultMAPSTex[y * width + x]);

            // Test Naive vs. MAPS (Texture+ILP)
            if (fabs(host_resultNaive[y * width + x] - host_resultMAPSILP[y * width + x]) > 1e-6)
            {
                if (numErrorsMAPSILP == 0)
                    printf("MAPS(Texture+ILP): First error in (%d, %d): %f != %f\n", x, y,
                    host_resultNaive[y * width + x], host_resultMAPSILP[y * width + x]);

                numErrorsMAPSILP++;
            }
            meanErrorMAPSILP += fabs(host_resultNaive[y * width + x] - host_resultMAPSILP[y * width + x]);
        }
    }

    printf("Conv2 of a %dx%d image with a %dx%d kernel (%d times)\n", width, height, KERNEL_RADIUS * 2 + 1, KERNEL_RADIUS * 2 + 1, REPETITIONS);

    printf("Naive kernel time: %f ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(nt2 - nt1).count() / 1000.0 / REPETITIONS);
    printf("MAPS  kernel time: %f ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(mt2 - mt1).count() / 1000.0 / REPETITIONS);
    printf("MAPST kernel time: %f ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(mtt2 - mtt1).count() / 1000.0 / REPETITIONS);
    printf("MAPSTILP kernel time: %f ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(mit2 - mit1).count() / 1000.0 / REPETITIONS);

    printf("Number of errors: Naive vs. MAPS = %d, MAPS(Texture) = %d, MAPS(Tex+ILP) = %d\n", numErrorsMAPS, numErrorsMAPSTex, numErrorsMAPSILP);
    printf("Mean error:       Naive vs. MAPS = %f, MAPS(Texture) = %f, MAPS(Tex+ILP) = %f\n",
           meanErrorMAPS / (float)((width - 2 * KERNEL_RADIUS) * (height - 2 * KERNEL_RADIUS)),
           meanErrorMAPSTex / (float)((width - 2 * KERNEL_RADIUS) * (height - 2 * KERNEL_RADIUS)),
           meanErrorMAPSILP / (float)((width - 2 * KERNEL_RADIUS) * (height - 2 * KERNEL_RADIUS)));

    ASSERT_EQ(numErrorsMAPS, 0);
    ASSERT_EQ(numErrorsMAPSTex, 0);
    ASSERT_EQ(numErrorsMAPSILP, 0);

    // Free allocated data
    CUASSERT_NOERR(cudaFree(dev_image));
    CUASSERT_NOERR(cudaFree(dev_naiveResult));
    CUASSERT_NOERR(cudaFree(dev_MAPSResult));
    CUASSERT_NOERR(cudaFree(dev_MAPSTexResult));
    CUASSERT_NOERR(cudaFree(dev_MAPSILPResult));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();
}
