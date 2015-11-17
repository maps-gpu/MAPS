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
GPU Model   |  Naive  |  MAPS  | CUBLAS |
------------+---------+--------+--------|
TITAN BLACK | 1.49 ms | 504 us | 160 us |
GTX 680     | 2.38 ms | 843 us | 190 us |

*/

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <gtest/gtest.h>
#include "cuda_gtest_utils.h"
#include <device_launch_parameters.h>

#include <maps/multi/multi.cuh>
#include <maps/maps.cuh>

#define MATRIX_SIZE 512
#define BLOCK_SIZE 32
#define REPETITIONS 1000

#define MATRIX_WIDTH  MATRIX_SIZE
#define MATRIX_HEIGHT MATRIX_SIZE

#define BW BLOCK_SIZE
#define BH BLOCK_SIZE

// Unique ID for sgemm input matrix textures
#define MATRIX_A_TEXTURE_UID 2222
#define MATRIX_B_TEXTURE_UID 2223

// Criterion for comparison
static const float kEpsilon = 1e-4;

__global__ void sgemmNaive(const float *A, size_t aStride,
                           const float *B, size_t bStride,
                           float *C, size_t cStride,
                           int width, int height, int k)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float result = 0.0f;

    for (int i = 0; i < k; ++i)
        result += A[y * aStride + i] * B[i * bStride + x];

    C[y * cStride + x] = result;
}

template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void GEMMKernel(maps::BlockSingleGPU<T, 2, 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1,1,1,maps::WB_NOCHECKS, MATRIX_A_TEXTURE_UID, maps::GR_TEXTURE> A,
                           maps::BlockSingleGPU<T, 2, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1,1,1,maps::WB_NOCHECKS, MATRIX_B_TEXTURE_UID, maps::GR_TEXTURE> B,
                           maps::StructuredInjectiveSingleGPU<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> C)
{
    MAPS_INIT(A, B, C);

    *C.begin() = Initialize<T>(0);

    // Perform the multiplication
    do
    {
        #pragma unroll
        MAPS_FOREACH(oiter, C)
        {
            // Initialize B's iterator as well
            auto B_iter = B.align(oiter);

            #pragma unroll
            MAPS_FOREACH_ALIGNED(A_iter, A, oiter)
            {
                *oiter += (*A_iter) * (*B_iter);
                ++B_iter;
            }
        }

        // Advance chunks efficiently
        maps::NextChunkAll(A, B);
    } while (!A.isDone());

    // Write out results
    C.commit();
}

TEST(Performance, Block2D_MatrixMultiplication)
{
#ifndef NDEBUG
    printf("Debug mode detected, skipping test\n");
    return;
#endif

    float *dev_A = NULL, *dev_B = NULL, *dev_naiveResult = NULL,
        *dev_MAPSResult = NULL, *dev_CUBLASResult = NULL;
    int width = MATRIX_WIDTH, height = MATRIX_HEIGHT;
    size_t aStride = 0, bStride = 0, cStride = 0;

    // Create input data
    std::vector<float> host_A(width * height, 0), host_B(width * height, 0);
    for (size_t i = 0; i < width * height; ++i)
    {
        host_A[i] = static_cast<float>(i % width);
        host_B[i] = static_cast<float>(i % height) * 0.5f;
    }

    // Allocate GPU buffers
    CUASSERT_NOERR(cudaMallocPitch(&dev_A, &aStride, sizeof(float) * width, height));
    CUASSERT_NOERR(cudaMallocPitch(&dev_B, &bStride, sizeof(float) * width, height));
    CUASSERT_NOERR(cudaMallocPitch(&dev_naiveResult, &cStride, sizeof(float) * width, height));
    CUASSERT_NOERR(cudaMallocPitch(&dev_MAPSResult, &cStride, sizeof(float) * width, height));
    CUASSERT_NOERR(cudaMallocPitch(&dev_CUBLASResult, &cStride, sizeof(float) * width, height));
    
    // Create GPU textures

    // Set texture parameters
    typedef typename maps::UniqueTexRef2D<float>::template TexId<MATRIX_A_TEXTURE_UID> TexIdA;
    typedef typename maps::UniqueTexRef2D<float>::template TexId<MATRIX_B_TEXTURE_UID> TexIdB;

    TexIdA::tex.addressMode[0] = cudaAddressModeClamp;
    TexIdA::tex.addressMode[1] = cudaAddressModeClamp;
    TexIdA::tex.filterMode = cudaFilterModeLinear;

    TexIdB::tex.addressMode[0] = cudaAddressModeClamp;
    TexIdB::tex.addressMode[1] = cudaAddressModeClamp;
    TexIdB::tex.filterMode = cudaFilterModeLinear;

    // Copy and compare the results
    std::vector<float> host_resultNaive(width * height, 0), host_resultMAPS(width * height, 0),
        host_resultCUBLAS(width * height, 0);

    // Bind textures to data
    CUASSERT_NOERR(TexIdA::BindTexture(dev_A, width, height, aStride));
    CUASSERT_NOERR(TexIdB::BindTexture(dev_B, width, height, bStride));

    dim3 block_dims(BW, BH, 1);
    dim3 grid_dims(maps::RoundUp(width, block_dims.x), maps::RoundUp(height, block_dims.y), 1);

    // Copy input data to GPU
    CUASSERT_NOERR(cudaMemcpy2DAsync(dev_A, aStride, &host_A[0], sizeof(float)* width,
                                     sizeof(float)* width, height, cudaMemcpyHostToDevice));
    CUASSERT_NOERR(cudaMemcpy2DAsync(dev_B, bStride, &host_B[0], sizeof(float)* width,
                                     sizeof(float)* width, height, cudaMemcpyHostToDevice));


    cudaDeviceSynchronize();
    auto nt1 = std::chrono::high_resolution_clock::now();

    // Run all three versions
    for (int i = 0; i < REPETITIONS; i++)
    {
        sgemmNaive<<<grid_dims, block_dims>>>(dev_A, (int)aStride / sizeof(float),
                                              dev_B, (int)bStride / sizeof(float),
                                              dev_naiveResult, (int)cStride / sizeof(float),
                                              width, height, width);
    }

    cudaDeviceSynchronize();
    auto nt2 = std::chrono::high_resolution_clock::now();

    CUASSERT_NOERR(cudaMemcpy2D(&host_resultNaive[0], sizeof(float)*width, dev_naiveResult, cStride, sizeof(float)* width, height, cudaMemcpyDeviceToHost));
    
    // MAPS

    // Create structures
    maps::BlockSingleGPU<float, 2, 0, BW, BH, 1, 1, 1, 1, maps::WB_NOCHECKS, MATRIX_A_TEXTURE_UID, maps::GR_TEXTURE> A;
    maps::BlockSingleGPU<float, 2, 1, BW, BH, 1, 1, 1, 1, maps::WB_NOCHECKS, MATRIX_B_TEXTURE_UID, maps::GR_TEXTURE> B;
    maps::StructuredInjectiveSingleGPU<float, 2, BW, BH, 1> C;

    A.m_ptr = dev_A;
    A.m_dimensions[0] = width;
    A.m_dimensions[1] = height;
    A.m_stride = (int)aStride / sizeof(float);

    B.m_ptr = dev_B;
    B.m_dimensions[0] = height;
    B.m_dimensions[1] = width;
    B.m_stride = (int)bStride / sizeof(float);

    C.m_ptr = dev_MAPSResult;
    C.m_dimensions[0] = width;
    C.m_dimensions[1] = height;
    C.m_stride = (int)cStride / sizeof(float);

    cudaDeviceSynchronize();
    auto mt1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPETITIONS; i++)
    {
        GEMMKernel<float, BW, BH><<<grid_dims, block_dims>>>(A, B, C);
    }

    cudaDeviceSynchronize();
    auto mt2 = std::chrono::high_resolution_clock::now();


    CUASSERT_NOERR(cudaMemcpy2D(&host_resultMAPS[0], sizeof(float)*width, dev_MAPSResult, cStride, sizeof(float)* width, height, cudaMemcpyDeviceToHost));

    CUASSERT_NOERR(TexIdA::UnbindTexture());
    CUASSERT_NOERR(TexIdB::UnbindTexture());

    // CUBLAS
    cublasHandle_t handle;
    ASSERT_EQ(cublasCreate(&handle), CUBLAS_STATUS_SUCCESS);
    float alpha = 1.0f, beta = 0.0f;

    cudaDeviceSynchronize();
    auto ct1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPETITIONS; i++)
    {
        // CUBLAS matrix representation is transposed by default
        ASSERT_EQ(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, width, width, height, &alpha, 
                              dev_A, (int)aStride / sizeof(float), dev_B, (int)bStride / sizeof(float),
                              &beta, dev_CUBLASResult, (int)cStride / sizeof(float)), CUBLAS_STATUS_SUCCESS);
    }


    CUASSERT_NOERR(cudaDeviceSynchronize());
    auto ct2 = std::chrono::high_resolution_clock::now();
      
    CUASSERT_NOERR(cudaMemcpy2D(&host_resultCUBLAS[0], sizeof(float)*width, dev_CUBLASResult, cStride, sizeof(float)* width, height, cudaMemcpyDeviceToHost));

    int numErrorsMAPS = 0, numErrorsCUBLAS = 0;
    float meanErrorMAPS = 0.0f, meanErrorCUBLAS = 0.0f;

    // Compare results
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            // Test Naive vs. MAPS
            if (fabs(1.0f - (host_resultNaive[y * width + x] / host_resultMAPS[y * width + x])) > kEpsilon)
            {
                if (numErrorsMAPS == 0)
                    printf("MAPS: First error in (%d, %d): %f != %f\n", x, y,
                           host_resultNaive[y * width + x], host_resultMAPS[y * width + x]);

                numErrorsMAPS++;
            }
            meanErrorMAPS += fabs(host_resultNaive[y * width + x] - host_resultMAPS[y * width + x]);

            // Test Naive vs. CUBLAS (result is transposed)
            if (fabs(1.0f - (host_resultNaive[y * width + x] / host_resultCUBLAS[x * width + y])) > kEpsilon)
            {
                if (numErrorsCUBLAS == 0)
                    printf("CUBLAS: First error in (%d, %d): %f != %f\n", x, y,
                           host_resultNaive[y * width + x], host_resultCUBLAS[x * width + y]);

                numErrorsCUBLAS++;
            }
            meanErrorCUBLAS += fabs(host_resultNaive[y * width + x] - host_resultCUBLAS[x * width + y]);
        }
    }

    printf("Matrix multiplication of two %dx%d matrices (%d times)\n", width, height, REPETITIONS);

    printf("Naive  kernel time: %f ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(nt2 - nt1).count() / 1000.0 / REPETITIONS);
    printf("MAPS   kernel time: %f ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(mt2 - mt1).count() / 1000.0 / REPETITIONS);
    printf("CUBLAS kernel time: %f ms\n\n", std::chrono::duration_cast<std::chrono::microseconds>(ct2 - ct1).count() / 1000.0 / REPETITIONS);
    
    printf("Number of errors: Naive vs. MAPS = %d, Naive vs. CUBLAS = %d\n", numErrorsMAPS, numErrorsCUBLAS);
    printf("Mean error:       Naive vs. MAPS = %f, Naive vs. CUBLAS = %f\n",
           meanErrorMAPS / (float)(width * height),
           meanErrorCUBLAS / (float)(width * height));

    ASSERT_EQ(numErrorsMAPS, 0);
    ASSERT_EQ(meanErrorCUBLAS, 0);

    // Free allocated data
    CUASSERT_NOERR(cudaFree(dev_A));
    CUASSERT_NOERR(cudaFree(dev_B));
    CUASSERT_NOERR(cudaFree(dev_naiveResult));
    CUASSERT_NOERR(cudaFree(dev_MAPSResult));
    CUASSERT_NOERR(cudaFree(dev_CUBLASResult));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();
}
