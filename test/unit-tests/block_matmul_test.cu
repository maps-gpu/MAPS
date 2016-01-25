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

#include <cmath>
#include <random>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "cuda_gtest_utils.h"

#include <cublas_v2.h>

#include <maps/input_containers/internal/io_common.cuh>
#include <maps/input_containers/internal/io_global.cuh>
#include <maps/input_containers/internal/io_boundaries.cuh>
#include <maps/input_containers/internal/io_globaltoshared.cuh>
#include <maps/input_containers/internal/io_globaltoarray.cuh>
#include <maps/input_containers/block.cuh>
#include <maps/multi/multi.cuh>

//////////////////////////////////////////////////////////////////////////////

// 8x8 ILP

// Sizes for GEMM
static const unsigned int kSizes[] = {
    32,
    128,
    192,
};

static const unsigned int kRandomSeed = 9699;
static const float kEpsilon = 1e-4;

// Chunk dimension (width and height) to use
static const unsigned int kChunkWidth =  8;
static const unsigned int kChunkHeight = 8;

// 8x8 blocks
static const unsigned int kBS = 8;

template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, bool A_TRANSPOSED,
         bool B_TRANSPOSED>
__global__ void GEMMKernel(maps::BlockSingleGPU<T, 2, A_TRANSPOSED ? 1 : 0,
                                                BLOCK_WIDTH,
                                                BLOCK_HEIGHT, 1, 1, 1, 1,
                                                maps::ZeroBoundaries, BLOCK_WIDTH,
                                                BLOCK_HEIGHT, 1, 
                                                maps::CustomOrdering<(A_TRANSPOSED ? 1 : 0), (A_TRANSPOSED ? 0 : 1)> > A,
                           maps::BlockSingleGPU<T, 2, B_TRANSPOSED ? 0 : 1,
                                                BLOCK_WIDTH,
                                                BLOCK_HEIGHT, 1, 1, 1, 1,
                                                maps::ZeroBoundaries, BLOCK_WIDTH,
                                                BLOCK_HEIGHT, 1, 
                                                maps::CustomOrdering<(B_TRANSPOSED ? 1 : 0), (B_TRANSPOSED ? 0 : 1)> > B,
                           maps::StructuredInjectiveSingleGPU<T, 2,
                                                              BLOCK_WIDTH,
                                                              BLOCK_HEIGHT,
                                                              1> C)
{
    MAPS_INIT(A, B, C);

    #pragma unroll
    MAPS_FOREACH(oiter, C)
       *oiter = Initialize<T>(0);

    // Perform the multiplication
    for (int i = 0; i < A.chunks(); ++i)
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
        maps::NextChunkAll(A, B);
    }

    // Write out results (the condition is for matrices that do not evenly 
    // divide by the block size)
    if (C.Items() > 0)
    {
        C.commit();
    }
}

template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, bool A_TRANSPOSED,
         bool B_TRANSPOSED>
void TestGEMM(int m, int k, int n)
{   
    // Allocate GPU memory
    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_Cregression = nullptr;
    size_t A_stride = 0, B_stride = 0, C_stride = 0, Creg_stride = 0;
    CUASSERT_NOERR(cudaMallocPitch(&d_A, &A_stride, sizeof(T) * k, m));
    CUASSERT_NOERR(cudaMallocPitch(&d_B, &B_stride, sizeof(T) * n, k));
    CUASSERT_NOERR(cudaMallocPitch(&d_C, &C_stride, sizeof(T) * n, m));
    CUASSERT_NOERR(cudaMallocPitch(&d_Cregression, &Creg_stride, sizeof(T) * n, m));

    // Initialize input
    std::mt19937 gen(kRandomSeed);
    std::uniform_real_distribution<T> ud(T(-50.0), T(100.0));
    std::vector<T> host_A(k*m), host_B(n*k), host_C(n*m), host_Creg(n*m);

    // Initialize A    
    for (size_t y = 0; y < m; ++y)
        for (size_t x = 0; x < k; ++x)
            host_A[y * n + x] = y * n + x;//ud(gen);

    // Initialize B
    for (size_t y = 0; y < k; ++y)
        for (size_t x = 0; x < n; ++x)
            host_B[y * k + x] = y * k + x;//ud(gen);
   
    // Copy input
    CUASSERT_NOERR(cudaMemcpy2D(d_A, A_stride, &host_A[0],
                                sizeof(T) * k, sizeof(T) * k, m,
                                cudaMemcpyHostToDevice));
    CUASSERT_NOERR(cudaMemcpy2D(d_B, B_stride, &host_B[0],
                                sizeof(T) * n, sizeof(T) * n, k,
                                cudaMemcpyHostToDevice));

    // Create structures
    maps::BlockSingleGPU<T, 2, A_TRANSPOSED ? 1 : 0, BLOCK_WIDTH, BLOCK_HEIGHT,
                         1, 1, 1, 1, maps::ZeroBoundaries, BLOCK_WIDTH,
                         BLOCK_HEIGHT, 1,
                         maps::CustomOrdering<(A_TRANSPOSED ? 1 : 0), (A_TRANSPOSED ? 0 : 1)> > A;
    maps::BlockSingleGPU<T, 2, B_TRANSPOSED ? 0 : 1, BLOCK_WIDTH, BLOCK_HEIGHT,
                         1, 1, 1, 1, maps::ZeroBoundaries, BLOCK_WIDTH,
                         BLOCK_HEIGHT, 1,
                         maps::CustomOrdering<(B_TRANSPOSED ? 1 : 0), (B_TRANSPOSED ? 0 : 1)> > B;
    maps::StructuredInjectiveSingleGPU<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> C;
    
    A.m_ptr = d_A;
    A.m_dimensions[0] = k;
    A.m_dimensions[1] = m;
    A.m_stride = (int)A_stride / sizeof(T);
    
    B.m_ptr = d_B;
    B.m_dimensions[0] = n;
    B.m_dimensions[1] = k;
    B.m_stride = (int)B_stride / sizeof(T);

    C.m_ptr = d_C;
    C.m_dimensions[0] = n;
    C.m_dimensions[1] = m;
    C.m_stride = (int)C_stride / sizeof(T);

    dim3 block_dims(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    dim3 grid_dims(maps::RoundUp(C.m_dimensions[0], block_dims.x), 
                   maps::RoundUp(C.m_dimensions[1], block_dims.y), 1);

    // Run test
    GEMMKernel<T, BLOCK_WIDTH, BLOCK_HEIGHT, A_TRANSPOSED, B_TRANSPOSED>
        <<<grid_dims, block_dims>>>(A, B, C);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Run regression (CUBLAS)
    cublasHandle_t handle;
    ASSERT_EQ(cublasCreate(&handle), CUBLAS_STATUS_SUCCESS);
    float alpha = 1.0f, beta = 0.0f;

    // CUBLAS matrix representation is transposed by default
    ASSERT_EQ(cublasSgemm(handle,
                          A_TRANSPOSED ? CUBLAS_OP_N : CUBLAS_OP_T,
                          B_TRANSPOSED ? CUBLAS_OP_N : CUBLAS_OP_T,
                          m, n, k, &alpha, 
                          d_A, (int)A_stride / sizeof(T), d_B,
                          (int)B_stride / sizeof(T), 
                          &beta, d_Cregression, (int)C_stride / sizeof(T)),
              CUBLAS_STATUS_SUCCESS);

    // Copy output
    CUASSERT_NOERR(cudaMemcpy2D(&host_C[0], sizeof(T) * n, d_C, C_stride,
                                sizeof(T) * n, m, cudaMemcpyDeviceToHost));
    CUASSERT_NOERR(cudaMemcpy2D(&host_Creg[0], sizeof(T) * n, d_Cregression,
                                Creg_stride, sizeof(T) * n, m,
                                cudaMemcpyDeviceToHost));

    // Check results
    for (size_t y = 0; y < m; ++y)
        for (size_t x = 0; x < n; ++x)
            EXPECT_LE(fabs(1.0f - (host_C[y * n + x] / host_Creg[x * m + y])), kEpsilon) << "at index (" << y << ", " << x
                << ") (" << host_C[y * n + x] << " != " << host_Creg[x * m + y] 
                << ") with size: " << m << " (block " << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << ")";

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_A));
    CUASSERT_NOERR(cudaFree(d_B));
    CUASSERT_NOERR(cudaFree(d_C));
    CUASSERT_NOERR(cudaFree(d_Cregression));
    ASSERT_EQ(cublasDestroy(handle), CUBLAS_STATUS_SUCCESS);
}

TEST(Block, SGEMM_NN)
{
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    for (int i = 0; i < num_sizes; ++i)
        TestGEMM<float, kBS, kBS, false, false>(kSizes[i], kSizes[i],
                                                kSizes[i]);
}

TEST(Block, SGEMM_NT)
{
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    for (int i = 0; i < num_sizes; ++i)
        TestGEMM<float, kBS, kBS, false, true>(kSizes[i], kSizes[i],
                                               kSizes[i]);
}

TEST(Block, SGEMM_TN)
{
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    for (int i = 0; i < num_sizes; ++i)
        TestGEMM<float, kBS, kBS, true, false>(kSizes[i], kSizes[i],
                                               kSizes[i]);
}

TEST(Block, SGEMM_TT)
{
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    for (int i = 0; i < num_sizes; ++i)
        TestGEMM<float, kBS, kBS, true, true>(kSizes[i], kSizes[i],
                                              kSizes[i]);
}
