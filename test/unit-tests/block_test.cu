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
#include <maps/input_containers/internal/io_globalread.cuh>
#include <maps/input_containers/internal/io_globaltoshared.cuh>
#include <maps/input_containers/internal/io_globaltoarray.cuh>
#include <maps/input_containers/block.cuh>
#include <maps/multi/multi.cuh>


// Test Block (ND) by fundamental linear algebra operations (matrix-vector product for 1D, matrix multiplication for 2D)

//////////////////////////////////////////////////////////////////////////////
// Block (1D)

// This kernel sums all the elements times the current index and stores the results
template <typename T, int BLOCK_WIDTH>
__global__ void SimpleBlock1D(maps::BlockSingleGPU<T, 1, 0, BLOCK_WIDTH, 1, 1> in,
                              maps::StructuredInjectiveSingleGPU<T, 1, BLOCK_WIDTH, 1, 1> out)
{
    __shared__ typename decltype(in)::SharedData sdata;
    in.init(sdata);
    out.init();
    
    int idx = threadIdx.x + blockIdx.x * BLOCK_WIDTH;

    if (out.Items() == 0)
        return;

    *out.begin() = Initialize<T>(0);

    do
    {
        #pragma unroll
        MAPS_FOREACH(oiter, out)
        {
            #pragma unroll
            MAPS_FOREACH_ALIGNED(iter, in, oiter)
            {
                *oiter += idx * *iter;
            }
        }

        in.nextChunk();
    } while (!in.isDone());

    out.commit();
}

template <typename T, int BLOCK_WIDTH>
void TestBlock1D(int size)
{
    // Allocate GPU memory
    T *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in,  sizeof(T) * size));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(T) * size));

    // Initialize input and compute output
    std::vector<T> in_val(size), out_val(size, Initialize<T>(0)),
                   expected_out_val(size, Initialize<T>(0));
    T expected_output_base = Initialize<T>(0);
    for (int i = 0; i < size; ++i)
    {
        in_val[i] = Initialize<T>(i);
        expected_output_base += in_val[i];
    }
    for (int i = 0; i < size; ++i)
        expected_out_val[i] = i * expected_output_base;

    // Copy input
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val[0], sizeof(T) * size, cudaMemcpyHostToDevice));

    // Create structures
    maps::BlockSingleGPU<T, 1, 0, BLOCK_WIDTH, 1, 1> in;
    maps::StructuredInjectiveSingleGPU<T, 1, BLOCK_WIDTH, 1, 1> out;
    
    in.m_ptr = d_in;
    in.m_dimensions[0] = size;
    in.m_stride = size;

    out.m_ptr = d_out;
    out.m_dimensions[0] = size;
    out.m_stride = size;
        
    dim3 block_dims(BLOCK_WIDTH, 1, 1);
    dim3 grid_dims(maps::RoundUp(size, block_dims.x), 1, 1);

    // Run test
    SimpleBlock1D<T, BLOCK_WIDTH> <<<grid_dims, block_dims>>>(in, out);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(T) * size, cudaMemcpyDeviceToHost));

    // Check results
    for (int i = 0; i < size; ++i)
        ASSERT_EQ(out_val[i], expected_out_val[i]) << "at index " << i;

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

TEST(Block, Block1DSimple)
{
    TestBlock1D<unsigned short, 32>(128);
}

//////////////////////////////////////////////////////////////////////////////
// Block (2D)

// This kernel sums all rows/columns (depends on TRANSPOSED) and stores the results
template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, bool TRANSPOSED>
__global__ void SimpleBlock2D(maps::BlockSingleGPU<T, 2, TRANSPOSED ? 1 : 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1> in,
                              T *out)
{
    __shared__ typename decltype(in)::SharedData sdata;
    in.init(sdata);

    int idx = (TRANSPOSED ? (threadIdx.x + blockIdx.x * BLOCK_WIDTH) : (threadIdx.y + blockIdx.y * BLOCK_HEIGHT));
    if (idx >= in.m_dimensions[TRANSPOSED ? 0 : 1])
        return;
    
    T result = Initialize<T>(0);

    do
    {
        #pragma unroll
        MAPS_FOREACH(iter, in)
        {
            result += *iter;
        }
        
        in.nextChunk();
    } while (!in.isDone());

    // Only the first one in the block writes output
    bool isFirstThread = ((TRANSPOSED ? threadIdx.y : threadIdx.x) == 0);
    if (isFirstThread)
        out[idx] = result;
}

template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, bool TRANSPOSED>
void TestBlock2D(int matrix_width, int matrix_height)
{
    // Allocate GPU memory
    T *d_in = nullptr, *d_out = nullptr;
    const int out_size = (TRANSPOSED ? matrix_width : matrix_height);
    size_t in_stride = 0;
    CUASSERT_NOERR(cudaMallocPitch(&d_in, &in_stride, sizeof(T) * matrix_width, matrix_height));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(T) * out_size));

    // Initialize input
    size_t totalsize = matrix_width * matrix_height;
    std::vector<T> in_val(totalsize), 
                   out_val(out_size, Initialize<T>(0)),
                   expected_out_val(out_size, Initialize<T>(0));
    for (int y = 0; y < matrix_height; ++y)
    {
        for (int x = 0; x < matrix_width; ++x)
        {
            int i = y * matrix_width + x;
            in_val[i] = Initialize<T>(i);            

            // Compute expected output values
            expected_out_val[TRANSPOSED ? x : y] += in_val[i];
        }
    }
    
    // Copy input
    CUASSERT_NOERR(cudaMemcpy2D(d_in, in_stride, &in_val[0], sizeof(T) * matrix_width,
                                sizeof(T) * matrix_width, matrix_height, cudaMemcpyHostToDevice));

    // Create structures
    maps::BlockSingleGPU<T, 2, TRANSPOSED ? 1 : 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1> in;
    
    in.m_ptr = d_in;
    in.m_dimensions[0] = matrix_width;
    in.m_dimensions[1] = matrix_height;
    in.m_stride = (int)in_stride / sizeof(T);
    
    dim3 block_dims(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    dim3 grid_dims(TRANSPOSED ? maps::RoundUp(matrix_width, block_dims.x) : 1, 
                   TRANSPOSED ? 1 : maps::RoundUp(matrix_height, block_dims.y), 1);

    // Run test
    SimpleBlock2D<T, BLOCK_WIDTH, BLOCK_HEIGHT, TRANSPOSED> <<<grid_dims, block_dims>>>(in, d_out);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(T) * out_size, cudaMemcpyDeviceToHost));

    // Check results
    for (int i = 0; i < out_size; ++i)
        ASSERT_EQ(out_val[i], expected_out_val[i]) << "at index " << i;

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

TEST(Block, Block2DSimple)
{
    TestBlock2D<int, 3, 2, false>(12, 6);
}

TEST(Block, Block2DTransposedSimple)
{
    TestBlock2D<int, 3, 2, true>(12, 6);
}

// This kernel sums all rows/columns (depends on TRANSPOSED) and stores the results
template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, bool TRANSPOSED, int ILP_X = 1, int ILP_Y = 1>
__global__ void SimpleBlock2DILP(maps::BlockSingleGPU<T, 2, TRANSPOSED ? 1 : 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> in,
                                 maps::StructuredInjective2DSingleGPU<T, BLOCK_WIDTH, BLOCK_HEIGHT, ILP_X, ILP_Y> out)
{
    MAPS_INIT(in, out);
    
    #pragma unroll
    MAPS_FOREACH(oiter, out)
    {
        *oiter = Initialize<T>(0);
    }

    do
    {
        #pragma unroll
        MAPS_FOREACH(oiter, out)
        {
            #pragma unroll
            MAPS_FOREACH_ALIGNED(iter, in, oiter)
            {
                *oiter += *iter;
            }
        }
        
        in.nextChunk();
    } while (!in.isDone());

    if(out.Items() > 0)
        out.commit();
}

template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, bool TRANSPOSED, int ILP_X, int ILP_Y>
void TestBlock2DILP(int matrix_width, int matrix_height)
{
    // Allocate GPU memory
    T *d_in = nullptr, *d_out = nullptr;
    const int out_size = (TRANSPOSED ? matrix_width : matrix_height);
    size_t in_stride = 0, out_stride = 0;
    CUASSERT_NOERR(cudaMallocPitch(&d_in, &in_stride, sizeof(T) * matrix_width, matrix_height));
    CUASSERT_NOERR(cudaMallocPitch(&d_out,&out_stride,sizeof(T) * matrix_width, matrix_height));

    // Initialize input
    size_t totalsize = matrix_width * matrix_height;
    std::vector<T> in_val(totalsize), 
                   out_val(totalsize, Initialize<T>(0)),
                   expected_out_val(out_size, Initialize<T>(0));
    for (int y = 0; y < matrix_height; ++y)
    {
        for (int x = 0; x < matrix_width; ++x)
        {
            int i = y * matrix_width + x;
            in_val[i] = Initialize<T>(i);            

            // Compute expected output values
            expected_out_val[TRANSPOSED ? x : y] += in_val[i];
        }
    }
    
    // Copy input
    CUASSERT_NOERR(cudaMemcpy2D(d_in, in_stride, &in_val[0], sizeof(T) * matrix_width,
                                sizeof(T) * matrix_width, matrix_height, cudaMemcpyHostToDevice));

    // Create structures
    maps::BlockSingleGPU<T, 2, TRANSPOSED ? 1 : 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> in;
    maps::StructuredInjective2DSingleGPU<float, BLOCK_WIDTH, BLOCK_HEIGHT, ILP_X, ILP_Y> out;
    
    in.m_ptr = d_in;
    in.m_dimensions[0] = matrix_width;
    in.m_dimensions[1] = matrix_height;
    in.m_stride = (int)in_stride / sizeof(T);

    out.m_ptr = d_out;
    out.m_dimensions[0] = matrix_width;
    out.m_dimensions[1] = matrix_height;
    out.m_stride = (int)out_stride / sizeof(T);
    
    dim3 block_dims(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    dim3 grid_dims(maps::RoundUp(matrix_width,  block_dims.x * ILP_X), 
                   maps::RoundUp(matrix_height, block_dims.y * ILP_Y), 1);
    
    // Run test
    SimpleBlock2DILP<T, BLOCK_WIDTH, BLOCK_HEIGHT, TRANSPOSED, ILP_X, ILP_Y> <<<grid_dims, block_dims>>>(in, out);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy2D(&out_val[0], sizeof(T) * matrix_width, d_out, out_stride, sizeof(T) * matrix_width, matrix_height,
                                cudaMemcpyDeviceToHost));

    // Check results
    for (int y = 0; y < matrix_height; ++y)
    {
        for (int x = 0; x < matrix_width; ++x)
        {
            ASSERT_EQ(out_val[y * matrix_width + x], expected_out_val[TRANSPOSED ? x : y]) << "at index (" << x << ", " << y << ")"
                << " Using ILP = " << ILP_X << ", " << ILP_Y;
        }
    }

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}


TEST(Block, Block2DILP)
{
    size_t width = 1200, height = 2400;

    #define TEST_BLOCK_ILP(IPX, IPY) TestBlock2DILP<float, 32, 8, false, IPX, IPY>(width, height)

    // Test various ILP configurations
    TEST_BLOCK_ILP(1, 1);
    TEST_BLOCK_ILP(2, 1);
    TEST_BLOCK_ILP(3, 1);
    TEST_BLOCK_ILP(4, 1);
    TEST_BLOCK_ILP(1, 2);
    TEST_BLOCK_ILP(1, 3);
    TEST_BLOCK_ILP(4, 2);
    TEST_BLOCK_ILP(5, 3);
    TEST_BLOCK_ILP(3, 5);
    TEST_BLOCK_ILP(8, 1);
    TEST_BLOCK_ILP(10, 1);

    #undef TEST_BLOCK_ILP
}

TEST(Block, Block2DTILP)
{
    size_t width = 1200, height = 2400;

    #define TEST_BLOCK_ILP(IPX, IPY) TestBlock2DILP<float, 32, 8, true, IPX, IPY>(width, height)

    // Test various ILP configurations
    TEST_BLOCK_ILP(1, 1);
    TEST_BLOCK_ILP(2, 1);
    TEST_BLOCK_ILP(3, 1);
    TEST_BLOCK_ILP(4, 1);
    TEST_BLOCK_ILP(1, 2);
    TEST_BLOCK_ILP(1, 3);
    TEST_BLOCK_ILP(4, 2);
    TEST_BLOCK_ILP(5, 3);
    TEST_BLOCK_ILP(3, 5);
    TEST_BLOCK_ILP(8, 1);
    TEST_BLOCK_ILP(10, 1);

    #undef TEST_BLOCK_ILP
}


//////////////////////////////////////////////////////////////////////////////
// Complex tests

// Sizes for GEMM and GEMV (results in square matrices)
static const unsigned int kSizes[] = {
    1,
    3,
    32,
    128,
    192,
    600,
    608,
    1024,
};

static const unsigned int kRandomSeed = 9690;
static const float kEpsilon = 1e-4;

// Block dimension (width and height) to use
static const unsigned int kBDIM = 32;

template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void GEMMKernel(maps::BlockSingleGPU<T, 2, 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1, 1, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, maps::WB_ZERO> A,
                           maps::BlockSingleGPU<T, 2, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1, 1, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, maps::WB_ZERO> B,
                           maps::StructuredInjectiveSingleGPU<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> C)
{
    __shared__ typename decltype(A)::SharedData a_sdata;
    __shared__ typename decltype(B)::SharedData b_sdata;

    // Initialize A and B asynchronously
    A.init_async(a_sdata);
    B.init_async(b_sdata);
    C.init();

    __syncthreads();

    // Since we use "init_async", we have to call "postsync" after the syncthreads
    A.init_async_postsync();
    B.init_async_postsync();    

    T result = Initialize<T>(0);

    // Perform the multiplication
    do
    {
        // Initialize B's iterator as well
        auto B_iter = B.begin();        

        #pragma unroll
        MAPS_FOREACH(A_iter, A)
        {
            result += (*A_iter) * (*B_iter);
            ++B_iter;
        }

        // Advance chunks efficiently
        __syncthreads();
        A.nextChunkAsync();
        B.nextChunkAsync();
        if (decltype(A)::SYNC_AFTER_NEXTCHUNK)
            __syncthreads();

    } while (!A.isDone());

    // Write out results (the condition is for matrices that do not evenly 
    // divide by the block size)
    if (C.Items() > 0)
    {
        *C.begin() = result;
        C.commit();
    }
}

template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT>
void TestGEMM(int m, int n, int k)
{
    // Allocate GPU memory
    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_Cregression = nullptr;
    size_t A_stride = 0, B_stride = 0, C_stride = 0, Creg_stride = 0;
    CUASSERT_NOERR(cudaMallocPitch(&d_A, &A_stride, sizeof(T) * n, m));
    CUASSERT_NOERR(cudaMallocPitch(&d_B, &B_stride, sizeof(T) * k, n));
    CUASSERT_NOERR(cudaMallocPitch(&d_C, &C_stride, sizeof(T) * k, m));
    CUASSERT_NOERR(cudaMallocPitch(&d_Cregression, &Creg_stride, sizeof(T) * k, m));

    // Initialize input
    std::mt19937 gen(kRandomSeed);
    std::uniform_real_distribution<T> ud(T(-50.0), T(100.0));
    std::vector<T> host_A(m*n), host_B(n*k), host_C(m*k), host_Creg(m*k);

    // Initialize A    
    for (size_t y = 0; y < m; ++y)
        for (size_t x = 0; x < n; ++x)
            host_A[y * n + x] = ud(gen);

    // Initialize B
    for (size_t y = 0; y < n; ++y)
        for (size_t x = 0; x < k; ++x)
            host_B[y * k + x] = ud(gen);
   
    // Copy input
    CUASSERT_NOERR(cudaMemcpy2D(d_A, A_stride, &host_A[0], sizeof(T) * n, sizeof(T) * n, m, cudaMemcpyHostToDevice));
    CUASSERT_NOERR(cudaMemcpy2D(d_B, B_stride, &host_B[0], sizeof(T) * k, sizeof(T) * k, n, cudaMemcpyHostToDevice));

    // Create structures
    maps::BlockSingleGPU<T, 2, 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1, 1, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, maps::WB_ZERO> A;
    maps::BlockSingleGPU<T, 2, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1, 1, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, maps::WB_ZERO> B;
    maps::StructuredInjectiveSingleGPU<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> C;
    
    A.m_ptr = d_A;
    A.m_dimensions[0] = n;
    A.m_dimensions[1] = m;
    A.m_stride = (int)A_stride / sizeof(T);
    
    B.m_ptr = d_B;
    B.m_dimensions[0] = k;
    B.m_dimensions[1] = n;
    B.m_stride = (int)B_stride / sizeof(T);

    C.m_ptr = d_C;
    C.m_dimensions[0] = k;
    C.m_dimensions[1] = m;
    C.m_stride = (int)C_stride / sizeof(T);

    dim3 block_dims(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    dim3 grid_dims(maps::RoundUp(C.m_dimensions[0], block_dims.x), 
                   maps::RoundUp(C.m_dimensions[1], block_dims.y), 1);

    // Run test
    GEMMKernel<T, BLOCK_WIDTH, BLOCK_HEIGHT><<<grid_dims, block_dims>>>(A, B, C);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Run regression (CUBLAS)
    cublasHandle_t handle;
    ASSERT_EQ(cublasCreate(&handle), CUBLAS_STATUS_SUCCESS);
    float alpha = 1.0f, beta = 0.0f;

    // CUBLAS matrix representation is transposed by default
    ASSERT_EQ(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, k, n, &alpha, 
                          d_A, (int)A_stride / sizeof(T), d_B, (int)B_stride / sizeof(T), 
                          &beta, d_Cregression, (int)C_stride / sizeof(T)), CUBLAS_STATUS_SUCCESS);

    // Copy output
    CUASSERT_NOERR(cudaMemcpy2D(&host_C[0], sizeof(T) * k, d_C, C_stride, sizeof(T) * k, m, cudaMemcpyDeviceToHost));
    CUASSERT_NOERR(cudaMemcpy2D(&host_Creg[0], sizeof(T) * k, d_Cregression, Creg_stride, sizeof(T) * k, m, cudaMemcpyDeviceToHost));

    // Check results
    for (size_t y = 0; y < m; ++y)
        for (size_t x = 0; x < k; ++x)
            ASSERT_LE(fabs(1.0f - (host_C[y * k + x] / host_Creg[x * m + y])), kEpsilon) << "at index (" << y << ", " << x
                << ") (" << host_C[y * k + x] << " != " << host_Creg[x * m + y] 
                << ") with size: " << m << " (block " << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << ")";

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_A));
    CUASSERT_NOERR(cudaFree(d_B));
    CUASSERT_NOERR(cudaFree(d_C));
    CUASSERT_NOERR(cudaFree(d_Cregression));
    ASSERT_EQ(cublasDestroy(handle), CUBLAS_STATUS_SUCCESS);
}

TEST(Block, SGEMMTestRandom)
{
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    for (int i = 0; i < num_sizes; ++i)
        TestGEMM<float, kBDIM, kBDIM>(kSizes[i], kSizes[i], kSizes[i]);
}

template<typename T, int BLOCK_HEIGHT>
__global__ void GEMVKernel(maps::BlockSingleGPU<T, 2, 0, 1, BLOCK_HEIGHT, 1, 1, 1, 1, maps::WB_ZERO> A,
                           maps::BlockSingleGPU<T, 1, 0, 1, BLOCK_HEIGHT, 1, 1, 1, 1, maps::WB_ZERO> x,
                           maps::StructuredInjectiveSingleGPU<T, 2, 1, BLOCK_HEIGHT, 1> b)
{
    __shared__ typename decltype(A)::SharedData a_sdata;
    __shared__ typename decltype(x)::SharedData x_sdata;

    // Initialize A and x asynchronously
    A.init_async(a_sdata);
    x.init_async(x_sdata);
    b.init();

    __syncthreads();

    // Since we use "init_async", we have to call "postsync" after the syncthreads
    A.init_async_postsync();
    x.init_async_postsync();    

    T result = Initialize<T>(0);

    // Perform the multiplication
    do
    {
        // Initialize x's iterator as well
        auto x_iter = x.begin();        

        #pragma unroll
        MAPS_FOREACH(A_iter, A)
        {
            result += (*A_iter) * (*x_iter);
            ++x_iter;
        }

        // Advance chunks efficiently
        __syncthreads();
        A.nextChunkAsync();
        x.nextChunkAsync();
        if (decltype(A)::SYNC_AFTER_NEXTCHUNK)
            __syncthreads();

    } while (!A.isDone());

    // Write out results (the condition is for matrices that do not evenly 
    // divide by the block size)
    if (b.Items() > 0)
    {
        *b.begin() = result;
        b.commit();
    }
}

template<typename T, int BLOCK_HEIGHT>
void TestGEMV(int m, int n)
{
    // Allocate GPU memory
    T *d_A = nullptr, *d_x = nullptr, *d_b = nullptr, *d_bregression = nullptr;
    size_t A_stride = 0;
    CUASSERT_NOERR(cudaMallocPitch(&d_A, &A_stride, sizeof(T) * n, m));
    CUASSERT_NOERR(cudaMalloc(&d_x, sizeof(T) * n));
    CUASSERT_NOERR(cudaMalloc(&d_b, sizeof(T) * m));
    CUASSERT_NOERR(cudaMalloc(&d_bregression, sizeof(T) * m));

    // Initialize input
    std::mt19937 gen(kRandomSeed);
    std::uniform_real_distribution<T> ud(T(-50.0), T(100.0));
    std::vector<T> host_A(m*n), host_x(n), host_b(m), host_breg(m);

    // Initialize A    
    for (size_t y = 0; y < m; ++y)
        for (size_t x = 0; x < n; ++x)
            host_A[y * n + x] = ud(gen);

    // Initialize x
    for (size_t i = 0; i < n; ++i)
        host_x[i] = ud(gen);
   
    // Copy input
    CUASSERT_NOERR(cudaMemcpy2D(d_A, A_stride, &host_A[0], sizeof(T) * n, sizeof(T) * n, m, cudaMemcpyHostToDevice));
    CUASSERT_NOERR(cudaMemcpy(d_x, &host_x[0], sizeof(T) * n, cudaMemcpyHostToDevice));

    // Create structures
    maps::BlockSingleGPU<T, 2, 0, 1, BLOCK_HEIGHT, 1, 1, 1, 1, maps::WB_ZERO> A;
    maps::BlockSingleGPU<T, 1, 0, 1, BLOCK_HEIGHT, 1, 1, 1, 1, maps::WB_ZERO> x;
    maps::StructuredInjectiveSingleGPU<T, 2, 1, BLOCK_HEIGHT, 1> b;
    
    A.m_ptr = d_A;
    A.m_dimensions[0] = n;
    A.m_dimensions[1] = m;
    A.m_stride = (int)A_stride / sizeof(T);
    
    x.m_ptr = d_x;
    x.m_dimensions[0] = n;
    x.m_stride = n;

    b.m_ptr = d_b;
    b.m_dimensions[0] = 1;
    b.m_dimensions[1] = m;
    b.m_stride = 1;

    dim3 block_dims(1, BLOCK_HEIGHT, 1);
    dim3 grid_dims(1, maps::RoundUp(b.m_dimensions[1], block_dims.y), 1);

    // Run test
    GEMVKernel<T, BLOCK_HEIGHT><<<grid_dims, block_dims>>>(A, x, b);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Run regression (CUBLAS)
    cublasHandle_t handle;
    ASSERT_EQ(cublasCreate(&handle), CUBLAS_STATUS_SUCCESS);
    float alpha = 1.0f, beta = 0.0f;

    // CUBLAS matrix representation is transposed by default
    ASSERT_EQ(cublasSgemv(handle, CUBLAS_OP_T, m, n, &alpha, 
                          d_A, (int)A_stride / sizeof(T), d_x, 1, &beta, d_bregression, 1), CUBLAS_STATUS_SUCCESS);

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&host_b[0],    d_b, sizeof(T) * m, cudaMemcpyDeviceToHost));
    CUASSERT_NOERR(cudaMemcpy(&host_breg[0], d_bregression, sizeof(T) * m, cudaMemcpyDeviceToHost));

    // Check results
    for (size_t i = 0; i < m; ++i)
        ASSERT_LE(fabs(1.0f - (host_b[i] / host_breg[i])), kEpsilon) << "at index " << i
            << " (" << host_b[i] << " != " << host_breg[i] << ") with size: " << m 
            << " (block size: " << BLOCK_HEIGHT << ")";

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_A));
    CUASSERT_NOERR(cudaFree(d_x));
    CUASSERT_NOERR(cudaFree(d_b));
    CUASSERT_NOERR(cudaFree(d_bregression));
    ASSERT_EQ(cublasDestroy(handle), CUBLAS_STATUS_SUCCESS);
}

TEST(Block, SGEMVTestRandom)
{
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    for (int i = 0; i < num_sizes; ++i)
        TestGEMV<float, kBDIM>(kSizes[i], kSizes[i]);
}
