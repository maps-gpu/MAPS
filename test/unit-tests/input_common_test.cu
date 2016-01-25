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
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <maps/internal/common.h>
#include <maps/internal/type_traits.hpp>
#include <maps/input_containers/internal/io_common.cuh>
#include <maps/input_containers/internal/io_global.cuh>
#include <maps/input_containers/internal/io_boundaries.cuh>
#include <maps/input_containers/internal/io_globaltoshared.cuh>

#include "cuda_gtest_utils.h"

TEST(CommonUtilities, Power)
{
    static_assert((maps::Power<9, 3>::value) == (9 * 9 * 9), "Test 1 failed");
    static_assert((maps::Power<900, 0>::value) == 1, "Test 2 failed");
    static_assert((maps::Power<2, 3>::value) == 8, "Test 3 failed");
    static_assert((maps::Power<2, 4>::value) == 16, "Test 4 failed");
    static_assert((maps::Power<4, 2>::value) == 16, "Test 5 failed");
    static_assert((maps::Power<4, 1>::value) == 4, "Test 6 failed");
    static_assert((maps::Power<-1, 2>::value) == 1, "Test 7 failed");
    static_assert((maps::Power<-1, 3>::value) == -1, "Test 8 failed");

    int number = maps::Power<-1, 5>::value;
    double othernumber = pow(-1.0, 5.0);

    ASSERT_EQ((double)number, othernumber) << "Runtime test failed";
}

template<typename T, typename Kernel>
static void SimpleGPUTest(Kernel kernel)
{
    // Allocate GPU memory
    T *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in, sizeof(T)));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(T)));

    // Copy input
    T in_val = Initialize<T>(123), out_val = Initialize<T>(0);
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val, sizeof(T), cudaMemcpyHostToDevice));

    // Run test
    void *args[] = { &d_in, &d_out };
    void *actual_kernel = (void *)kernel;
    CUASSERT_NOERR(cudaLaunchKernel(actual_kernel, dim3(1, 1, 1), dim3(1, 1, 1), args));
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val, d_out, sizeof(T), cudaMemcpyDeviceToHost));

    ASSERT_EQ(in_val, out_val);

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

template <typename T, typename GlobalIOScheme>
__global__ void ReadGlobalKernel(const T *in, T *out)
{
    GlobalIOScheme::template Read1D<T>(in, 0, *out);
}

TEST(CommonUtilities, ReadGlobal8)
{
    SimpleGPUTest<unsigned char>(ReadGlobalKernel<unsigned char, maps::DirectIO>);
}

TEST(CommonUtilities, ReadGlobal16)
{
    SimpleGPUTest<short>(ReadGlobalKernel<short, maps::DirectIO>);
}

TEST(CommonUtilities, ReadGlobal32)
{
    SimpleGPUTest<float>(ReadGlobalKernel<float, maps::DirectIO>);
}

TEST(CommonUtilities, ReadGlobal64)
{
    SimpleGPUTest<double>(ReadGlobalKernel<double, maps::DirectIO>);
}

TEST(CommonUtilities, ReadGlobal128)
{
    SimpleGPUTest<float4>(ReadGlobalKernel<float4, maps::DirectIO>);
}

TEST(CommonUtilities, ReadGlobalDistinct8)
{
    SimpleGPUTest<unsigned char>(ReadGlobalKernel<unsigned char, maps::DistinctIO>);
}

TEST(CommonUtilities, ReadGlobalDistinct16)
{
    SimpleGPUTest<short>(ReadGlobalKernel<short, maps::DistinctIO>);
}

TEST(CommonUtilities, ReadGlobalDistinct32)
{
    SimpleGPUTest<float>(ReadGlobalKernel<float, maps::DistinctIO>);
}

TEST(CommonUtilities, ReadGlobalDistinct64)
{
    SimpleGPUTest<double>(ReadGlobalKernel<double, maps::DistinctIO>);
}

TEST(CommonUtilities, ReadGlobalDistinct128)
{
    SimpleGPUTest<float4>(ReadGlobalKernel<float4, maps::DistinctIO>);
}

template <typename T, typename GlobalIOScheme>
__global__ void WriteGlobalKernel(const T *in, T *out)
{
    GlobalIOScheme::template Write<T>(out, 0, *in);
}

TEST(CommonUtilities, WriteGlobal8)
{
    SimpleGPUTest<unsigned char>(WriteGlobalKernel<unsigned char, maps::DirectIO>);
}

TEST(CommonUtilities, WriteGlobal16)
{
    SimpleGPUTest<short>(WriteGlobalKernel<short, maps::DirectIO>);
}

TEST(CommonUtilities, WriteGlobal32)
{
    SimpleGPUTest<float>(WriteGlobalKernel<float, maps::DirectIO>);
}

TEST(CommonUtilities, WriteGlobal64)
{
    SimpleGPUTest<double>(WriteGlobalKernel<double, maps::DirectIO>);
}

TEST(CommonUtilities, WriteGlobal128)
{
    SimpleGPUTest<float4>(WriteGlobalKernel<float4, maps::DirectIO>);
}

template <typename Kernel>
void *GetKernelFunc(Kernel kernel)
{
    return (void *)kernel;
}

TEST(CommonUtilities, ReadGlobalTexture1D_32)
{
    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in, sizeof(float)));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float)));

    // Copy input
    float in_val = 12321.0f, out_val = 0.0f;
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val, sizeof(float), cudaMemcpyHostToDevice));

    // Set texture parameters
    typedef typename maps::UniqueTexRef1D<float>::template TexId<1111> TexId;

    TexId::tex.addressMode[0] = cudaAddressModeClamp;
    TexId::tex.addressMode[1] = cudaAddressModeClamp;
    TexId::tex.filterMode = cudaFilterModeLinear;

    CUASSERT_NOERR(TexId::BindTexture(d_in, sizeof(float)));

    // Run test
    void *args[] = { &d_in, &d_out };
    CUASSERT_NOERR(cudaLaunchKernel(GetKernelFunc(ReadGlobalKernel<float, maps::TextureIO<1111> >), dim3(1, 1, 1), dim3(1, 1, 1), args));
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_EQ(in_val, out_val);

    TexId::UnbindTexture();

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

TEST(CommonUtilities, ReadGlobalTexture1D_128)
{
    // Allocate GPU memory
    float4 *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in, sizeof(float4)));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float4)));

    // Copy input
    float4 in_val = make_float4(12321.0f, 2.56f, 3.0f, 4.0f), out_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val, sizeof(float4), cudaMemcpyHostToDevice));

    // Set texture parameters
    typedef typename maps::UniqueTexRef1D<float4>::template TexId<1111> TexId;

    TexId::tex.addressMode[0] = cudaAddressModeClamp;
    TexId::tex.addressMode[1] = cudaAddressModeClamp;
    TexId::tex.filterMode = cudaFilterModeLinear;

    CUASSERT_NOERR(TexId::BindTexture(d_in, sizeof(float4)));

    // Run test
    void *args[] = { &d_in, &d_out };
    CUASSERT_NOERR(cudaLaunchKernel(GetKernelFunc(ReadGlobalKernel<float4, maps::TextureIO<1111> >), dim3(1, 1, 1), dim3(1, 1, 1), args));
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val, d_out, sizeof(float4), cudaMemcpyDeviceToHost));

    ASSERT_EQ(in_val, out_val);

    TexId::UnbindTexture();

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

template <typename T, typename GlobalIOScheme>
__global__ void ReadGlobalTex2DKernel(T *out)
{
    GlobalIOScheme::template Read2D<T>(nullptr, 0, 0, 0, *out);
}

TEST(CommonUtilities, ReadGlobalTexture2D_32)
{
    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in, sizeof(float)));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float)));

    // Copy input
    float in_val = 12321.0f, out_val = 0.0f;
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val, sizeof(float), cudaMemcpyHostToDevice));

    // Set texture parameters
    typedef typename maps::UniqueTexRef2D<float>::template TexId<1112> TexId;

    TexId::tex.addressMode[0] = cudaAddressModeClamp;
    TexId::tex.addressMode[1] = cudaAddressModeClamp;
    TexId::tex.filterMode = cudaFilterModeLinear;

    CUASSERT_NOERR(TexId::BindTexture(d_in, 1, 1, sizeof(float)));

    // Run test
    void *args[] = { &d_out };
    CUASSERT_NOERR(cudaLaunchKernel(GetKernelFunc(ReadGlobalTex2DKernel<float, maps::TextureIO<1112> >), dim3(1, 1, 1), dim3(1, 1, 1), args));
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_EQ(in_val, out_val);

    TexId::UnbindTexture();

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

TEST(CommonUtilities, ReadGlobalTexture2D_128)
{
    // Allocate GPU memory
    float4 *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in, sizeof(float4)));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float4)));

    // Copy input
    float4 in_val = make_float4(12321.0f, 2.56f, 3.0f, 4.0f), out_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val, sizeof(float4), cudaMemcpyHostToDevice));

    // Set texture parameters
    typedef typename maps::UniqueTexRef2D<float4>::template TexId<1112> TexId;

    TexId::tex.addressMode[0] = cudaAddressModeClamp;
    TexId::tex.addressMode[1] = cudaAddressModeClamp;
    TexId::tex.filterMode = cudaFilterModeLinear;

    CUASSERT_NOERR(TexId::BindTexture(d_in, 1, 1, sizeof(float4)));

    // Run test
    void *args[] = { &d_out };
    CUASSERT_NOERR(cudaLaunchKernel(GetKernelFunc(ReadGlobalTex2DKernel<float4, maps::TextureIO<1112> >), dim3(1, 1, 1), dim3(1, 1, 1), args));
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val, d_out, sizeof(float4), cudaMemcpyDeviceToHost));

    ASSERT_EQ(in_val, out_val);

    TexId::UnbindTexture();

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

// With a single block
template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, typename Kernel>
static void SimpleSharedGPUTest(Kernel kernel, bool bApron)
{
    enum
    {
        TOTAL_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * BLOCK_DEPTH,
    };

    // Allocate GPU memory
    T *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in,  sizeof(T) * TOTAL_SIZE));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(T) * TOTAL_SIZE));

    // Initialize input
    std::vector<T> in_val(TOTAL_SIZE), out_val(TOTAL_SIZE);
    T expected_out_val = Initialize<T>(0);
    for (int z = 0; z < BLOCK_DEPTH; ++z)
    {
        T expected_out_surface = Initialize<T>(0);

        for (int y = 0; y < BLOCK_HEIGHT; ++y)
        {
            T expected_out_row = Initialize<T>(0);

            for (int x = 0; x < BLOCK_WIDTH; ++x)
            {
                int i = z * BLOCK_HEIGHT * BLOCK_WIDTH + y * BLOCK_WIDTH + x;
                in_val[i] = Initialize<T>(i);
                out_val[i] = Initialize<T>(0);

                expected_out_row += in_val[i];

                // Sum the two outstanding values too ("wrapped apron")
                if (bApron && BLOCK_WIDTH > 1 && (x == 0 || x == (BLOCK_WIDTH - 1)))                    
                    expected_out_row += in_val[i];
            }

            expected_out_surface += expected_out_row;

            // Sum the two outstanding values too ("wrapped apron")
            if (bApron && BLOCK_HEIGHT > 1 && (y == 0 || y == (BLOCK_HEIGHT - 1)))
                expected_out_surface += expected_out_row;
        }

        expected_out_val += expected_out_surface;

        // Sum the two outstanding values too ("wrapped apron")
        if (bApron && BLOCK_DEPTH > 1 && (z == 0 || z == (BLOCK_DEPTH - 1)))
            expected_out_val += expected_out_surface;
    }
    
    // Copy input
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val[0], sizeof(T) * TOTAL_SIZE, cudaMemcpyHostToDevice));

    // Run test
    void *args[] = { &d_in, &d_out };
    void *actual_kernel = (void *)kernel;
    CUASSERT_NOERR(cudaLaunchKernel(actual_kernel, dim3(1, 1, 1), dim3(BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH), args));
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(T) * TOTAL_SIZE, cudaMemcpyDeviceToHost));

    for (int i = 0; i < TOTAL_SIZE; ++i)
        ASSERT_EQ(out_val[i], expected_out_val) << "at index " << i;

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

template <typename T, int BLOCK_WIDTH>
__global__ void GlobalToShared1DKernel(const T *in, T *out)
{
    enum {
        SHARED_SIZE = BLOCK_WIDTH + 2,
    };

    __shared__ T smem[SHARED_SIZE];

    maps::GlobalToShared1D<T, BLOCK_WIDTH, 1, 1, SHARED_SIZE, false, maps::WrapBoundaries, maps::DirectIO>(in, BLOCK_WIDTH, -1, smem, 0, 1);

    T result = Initialize<T>(0);
    for (int i = 0; i < SHARED_SIZE; ++i)
        result += smem[i];

    out[threadIdx.x] = result;
}

TEST(CommonUtilities, GlobalToShared1D_32)
{
    SimpleSharedGPUTest<float, 8, 1, 1>(GlobalToShared1DKernel<float, 8>, true);
}

TEST(CommonUtilities, GlobalToShared1D_128)
{
    SimpleSharedGPUTest<float4, 8, 1, 1>(GlobalToShared1DKernel<float4, 8>, true);
}

// Not a multiple of 128-bit
TEST(CommonUtilities, GlobalToShared1D_96)
{
    SimpleSharedGPUTest<int3, 8, 1, 1>(GlobalToShared1DKernel<int3, 8>, true);
}

template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void GlobalToShared2DSimpleKernel(const T *in, T *out)
{
    enum {
        SHARED_SIZE = (BLOCK_WIDTH) * (BLOCK_HEIGHT),
    };

    __shared__ T smem[SHARED_SIZE];

    maps::GlobalToShared2D<T, BLOCK_WIDTH, BLOCK_HEIGHT, 1, BLOCK_WIDTH, BLOCK_WIDTH,
        BLOCK_HEIGHT, false, maps::NoBoundaries, maps::DirectIO>(
        in, BLOCK_WIDTH, BLOCK_WIDTH, 0, BLOCK_HEIGHT, 0, smem, 0, 1);

    T result = Initialize<T>(0);
    for (int i = 0; i < SHARED_SIZE; ++i)
        result += smem[i];

    out[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = result;
}

// Mostly corresponds to Block2D

TEST(CommonUtilities, GlobalToShared2DSimple_32)
{
    SimpleSharedGPUTest<float, 16, 8, 1>(GlobalToShared2DSimpleKernel<float, 16, 8>, false);
}

TEST(CommonUtilities, GlobalToShared2DSimple_128)
{
    SimpleSharedGPUTest<float4, 16, 8, 1>(GlobalToShared2DSimpleKernel<float4, 16, 8>, false);
}

// Not a multiple of 128-bit
TEST(CommonUtilities, GlobalToShared2DSimple_96)
{
    SimpleSharedGPUTest<int3, 16, 8, 1>(GlobalToShared2DSimpleKernel<int3, 16, 8>, false);
}


template <typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void GlobalToShared2DKernel(const T *in, T *out)
{
    enum {
        SHARED_SIZE = (BLOCK_WIDTH + 2) * (BLOCK_HEIGHT + 2),
    };

    __shared__ T smem[SHARED_SIZE];

    maps::GlobalToShared2D<T, BLOCK_WIDTH, BLOCK_HEIGHT, 1, BLOCK_WIDTH + 2, BLOCK_WIDTH + 2,
        BLOCK_HEIGHT + 2, false, maps::WrapBoundaries, maps::DirectIO>(
        in, BLOCK_WIDTH, BLOCK_WIDTH, -1, BLOCK_HEIGHT, -1, smem, 0, 1);

    T result = Initialize<T>(0);
    for (int i = 0; i < SHARED_SIZE; ++i)
        result += smem[i];

    out[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = result;
}

// General 2D case

TEST(CommonUtilities, GlobalToShared2D_32)
{
    SimpleSharedGPUTest<float, 16, 8, 1>(GlobalToShared2DKernel<float, 16, 8>, true);
}

TEST(CommonUtilities, GlobalToShared2D_128)
{
    SimpleSharedGPUTest<float4, 16, 8, 1>(GlobalToShared2DKernel<float4, 16, 8>, true);
}

// Not a multiple of 128-bit
TEST(CommonUtilities, GlobalToShared2D_96)
{
    SimpleSharedGPUTest<int3, 16, 8, 1>(GlobalToShared2DKernel<int3, 16, 8>, true);
}
