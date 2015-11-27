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
#include <string>
#include <sstream>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "cuda_gtest_utils.h"

#include <maps/input_containers/internal/io_common.cuh>
#include <maps/input_containers/internal/io_globalread.cuh>
#include <maps/input_containers/internal/io_globaltoshared.cuh>
#include <maps/input_containers/internal/io_globaltoarray.cuh>
#include <maps/multi/multi.cuh>
#include <maps/multi/pinned_allocation.h>

// Test Window (ND) by convolution (1-3 dimensions, varying sizes)

// CUDA block dimensions
#define BW 8
#define BH 8
#define BD 8

// Convolution radius range
#define MIN_WINDOW_RADIUS 0
#define MAX_WINDOW_RADIUS 6

// For repeatable randomization (kernels)
static const int kRandomSeed1 = 1234;

// For repeatable randomization (images)
static const int kRandomSeed2 = 4321;

// For comparing float values
static const float kEpsilon = 1e-6f;

// Sizes (exhaustive, uses each of the permutations, M^N)
static const unsigned int kSizes[] = {
    1,
    3,
    32,
    128,
    192,
    600,
    608,
};

// 1 - 3 dimensions
__constant__ float kConvKernel[(MAX_WINDOW_RADIUS * 2 + 1) * 
                               (MAX_WINDOW_RADIUS * 2 + 1) *
                               (MAX_WINDOW_RADIUS * 2 + 1)];

template <int DIMS, int RADIUS, maps::BorderBehavior BORDERS>
struct ConvRegression
{
    static void RunConvRegression(float *device_in, float *device_out, 
                                  float *host_output, unsigned int buffer_size[DIMS]);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
          int RADIUS, maps::BorderBehavior BORDERS, int IPX = 1, 
          int IPY = 1, int IPZ = 1>
__global__ void MAPSConvolution(maps::WindowSingleGPU<float, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, RADIUS, IPX, IPY, IPZ, BORDERS> in,
                                maps::StructuredInjectiveSingleGPU<float, DIMS, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, IPX, IPY> out)
{    
    MAPS_INIT(in, out);

    if (out.Items() == 0)
        return;

    #pragma unroll
    MAPS_FOREACH(oiter, out)
    {
        float result = 0.0f;
        int i = 0;
        
        #pragma unroll
        MAPS_FOREACH_ALIGNED(iter, in, oiter)
        {
            result += *iter * kConvKernel[i++];
        }
        *oiter = result;
    }

    out.commit();
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

template<maps::BorderBehavior BORDERS>
__device__ float GetElement1D(const float *buffer, int width, int x)
{
    switch (BORDERS)
    {
    default:
    case maps::WB_NOCHECKS:
        return buffer[x];

    case maps::WB_COPY:
        return buffer[maps::Clamp(x, 0, width - 1)];

    case maps::WB_WRAP:
        return buffer[maps::Wrap(x, width)];

    case maps::WB_ZERO:
        if (x < 0 || x >= width)
            return 0.0f;

        return buffer[x];
    }
}

template<int WINDOW_RADIUS, maps::BorderBehavior BORDERS>
__global__ void Conv1Regression(const float *buffer, int width,
                                float *result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
        return;

    float local_result = 0.0f;

    #pragma unroll
    for (int i = 0; i < WINDOW_RADIUS * 2 + 1; ++i)
        local_result += kConvKernel[i] * GetElement1D<BORDERS>(buffer, width, x + i - WINDOW_RADIUS);

    result[x] = local_result;
}

template <int RADIUS, maps::BorderBehavior BORDERS>
struct ConvRegression<1, RADIUS, BORDERS>
{
    static void RunConvRegression(float *device_in, float *device_out, 
                                  float *host_output, unsigned int buffer_size[1])
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        dim3 block_dims(BW);
        dim3 grid_dims(maps::RoundUp(buffer_size[0], BW));

        Conv1Regression<RADIUS, BORDERS> <<<grid_dims, block_dims>>>(
            device_in, buffer_size[0], device_out);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        ASSERT_EQ(cudaMemcpy(host_output, device_out,
            sizeof(float) * buffer_size[0],
            cudaMemcpyDeviceToHost), cudaSuccess);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<maps::BorderBehavior BORDERS>
__device__ float GetElement2D(const float *buffer, int width,
                              int height, unsigned int stride,
                              int x, int y)
{
    switch (BORDERS)
    {
    default:
    case maps::WB_NOCHECKS:
        return buffer[y * stride + x];

    case maps::WB_COPY:
        return buffer[maps::Clamp(y, 0, height - 1) * stride + maps::Clamp(x, 0, width - 1)];

    case maps::WB_WRAP:
        return buffer[maps::Wrap(y, height) * stride + maps::Wrap(x, width)];

    case maps::WB_ZERO:
        if (y < 0 || y >= height || x < 0 || x >= width)
            return 0.0f;

        return buffer[y * stride + x];
    }
}

template<int WINDOW_RADIUS, maps::BorderBehavior BORDERS>
__global__ void Conv2Regression(const float *buffer, int width, 
                                int height, unsigned int stride,
                                float *result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float local_result = 0.0f;

    #pragma unroll
    for (int i = 0; i < WINDOW_RADIUS * 2 + 1; ++i)
    {
        #pragma unroll
        for (int j = 0; j < WINDOW_RADIUS * 2 + 1; ++j)
        {
            local_result += kConvKernel[i * (WINDOW_RADIUS * 2 + 1) + j] *
                            GetElement2D<BORDERS>(buffer, width, height, stride,
                                                  x + j - WINDOW_RADIUS, y + i - WINDOW_RADIUS);
        }
    }
    
    result[y * stride + x] = local_result;
}


template <int RADIUS, maps::BorderBehavior BORDERS>
struct ConvRegression<2, RADIUS, BORDERS>
{
    static void RunConvRegression(float *device_in, float *device_out, 
                                  float *host_output, unsigned int buffer_size[2])
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        dim3 block_dims(BW, BH);
        dim3 grid_dims(maps::RoundUp(buffer_size[0], BW),
                       maps::RoundUp(buffer_size[1], BH));

        Conv2Regression<RADIUS, BORDERS> <<<grid_dims, block_dims>>>(
            device_in, buffer_size[0], buffer_size[1],
            buffer_size[0], device_out);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        ASSERT_EQ(cudaMemcpy(host_output, device_out,
            sizeof(float) * buffer_size[0] * buffer_size[1],
            cudaMemcpyDeviceToHost), cudaSuccess);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<maps::BorderBehavior BORDERS>
__device__ float GetElement3D(const float *buffer, int lx, int ly,
                              int lz, unsigned int stride, 
                              int x, int y, int z)
{
    switch (BORDERS)
    {
    default:
    case maps::WB_NOCHECKS:
        return buffer[z * ly * stride + y * stride + x];

    case maps::WB_COPY:
        return buffer[maps::Clamp(z, 0, lz - 1) * ly * stride + maps::Clamp(y, 0, ly - 1) * stride + maps::Clamp(x, 0, lx - 1)];

    case maps::WB_WRAP:
        return buffer[maps::Wrap(z, lz) * ly * stride + maps::Wrap(y, ly) * stride + maps::Wrap(x, lx)];

    case maps::WB_ZERO:
        if (z < 0 || z >= lz || y < 0 || y >= ly || x < 0 || x >= lx)
            return 0.0f;

        return buffer[z * ly * stride + y * stride + x];
    }
}

template<int WINDOW_RADIUS, maps::BorderBehavior BORDERS>
__global__ void Conv3Regression(const float *buffer, int lx, 
                                int ly, int lz, 
                                unsigned int stride, float *result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz)
        return;

    float local_result = 0.0f;

    #pragma unroll
    for (int i = 0; i < WINDOW_RADIUS * 2 + 1; ++i)
    {
        #pragma unroll
        for (int j = 0; j < WINDOW_RADIUS * 2 + 1; ++j)
        {
            #pragma unroll
            for (int k = 0; k < WINDOW_RADIUS * 2 + 1; ++k)
            {
                int ind = i * (WINDOW_RADIUS * 2 + 1) * (WINDOW_RADIUS * 2 + 1) + 
                          j * (WINDOW_RADIUS * 2 + 1) + k;

                local_result += kConvKernel[ind] *
                                GetElement3D<BORDERS>(buffer, lx, ly, lz, stride,
                                                      x + k - WINDOW_RADIUS, 
                                                      y + j - WINDOW_RADIUS,
                                                      z + i - WINDOW_RADIUS);
            }
        }
    }
    
    result[z * ly * stride + y * stride + x] = local_result;
}

template <int RADIUS, maps::BorderBehavior BORDERS>
struct ConvRegression<3, RADIUS, BORDERS>
{
    static void RunConvRegression(float *device_in, float *device_out, 
                                  float *host_output, unsigned int buffer_size[3])
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        dim3 block_dims(BW, BH, BD);
        dim3 grid_dims(maps::RoundUp(buffer_size[0], BW),
                       maps::RoundUp(buffer_size[1], BH), 
                       maps::RoundUp(buffer_size[2], BD));

        Conv3Regression<RADIUS, BORDERS><<<grid_dims, block_dims>>>(
            device_in, buffer_size[0], buffer_size[1], 
            buffer_size[2], buffer_size[0], device_out);

        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        ASSERT_EQ(cudaMemcpy(host_output, device_out, 
            sizeof(float) * buffer_size[0] * buffer_size[1] * buffer_size[2],
            cudaMemcpyDeviceToHost), cudaSuccess);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// Note that the "Unit" convolution tests also check the "Regression" functions for errors.

inline void CheckDevices(int& num_gpus)
{
    num_gpus = 0;
    ASSERT_EQ(cudaGetDeviceCount(&num_gpus), cudaSuccess);

    ASSERT_GE(num_gpus, 1);
}

static inline const char *BoundariesToString(maps::BorderBehavior borders)
{
    switch (borders)
    {
    default:
        return "N/A";
    case maps::WB_NOCHECKS:
        return "unchecked";
    case maps::WB_WRAP:
        return "wrapped";
    case maps::WB_COPY:
        return "clamped";
    case maps::WB_ZERO:
        return "zero";
    }
}

template <int DIMS>
std::string BufferSize(unsigned int buffer_size[DIMS])
{
    std::stringstream ss;

    ss << buffer_size[0];
    for (int i = 1; i < DIMS; ++i)
        ss << "x" << buffer_size[i];

    return ss.str();
}

template <int DIMS, int RADIUS, maps::BorderBehavior BORDERS, int IPX = 1, int IPY = 1, int IPZ = 1>
void TestWindow(bool random_kernel, unsigned int buffer_size[DIMS],
                float *host_kernel, float *buffer_in, float *buffer_out,
                float *buffer_regression, float *device_in, float *device_out)
{
    int num_gpus = 0;
    CheckDevices(num_gpus);

    // Verify that the buffer is large enough to handle NOCHECKS
    if (BORDERS == maps::WB_NOCHECKS)
    {
        for (int i = 0; i < DIMS; ++i)
        {
            if (buffer_size[i] < (RADIUS * 2 + 1))
            {
                printf("Buffer size mismatch (dim[%d] = %d (< %d)), skipping test\n", i, buffer_size[i], RADIUS * 2 + 1);
                return;
            }
        }
    }

    // Verify that wrap will work (it will not wrap the buffer more than once)
    if (BORDERS == maps::WB_WRAP)
    {
        for (int i = 0; i < DIMS; ++i)
        {
            // Skip 
            if (RADIUS > (int)buffer_size[i])
                return;
        }
    }

    // Verify that ILP optimizations will work
    if (buffer_size[0] < IPX)
    {
        printf("Buffer X size is too short for ILP optimization test (%d < %d)\n",
               buffer_size[0], IPX);
        return;
    }
    if (buffer_size[0] % IPX != 0)
    {
        printf("Buffer X size (%d) is not divisible by ILP amount (%d)\n", buffer_size[0], IPX);
        return;
    }
    if (DIMS < 2 && IPY > 1)
        return;
    if (DIMS >= 2 && buffer_size[1] < IPY)
    {
        printf("Buffer Y size is too short for ILP optimization test (%d < %d)\n",
               buffer_size[1], IPY);
        return;
    }
    if (DIMS >= 2 && buffer_size[1] % IPY != 0)
    {
        printf("Buffer Y size (%d) is not divisible by ILP amount (%d)\n", buffer_size[1], IPY);
        return;
    }
    if (DIMS < 3 && IPZ > 1)
        return;
    if (DIMS >= 3 && buffer_size[2] < IPZ)
    {
        printf("Buffer Z size is too short for ILP optimization test (%d < %d)\n",
               buffer_size[2], IPZ);
        return;
    }
    if (DIMS >= 3 && buffer_size[2] % IPZ != 0)
    {
        printf("Buffer Z size (%d) is not divisible by ILP amount (%d)\n", buffer_size[2], IPZ);
        return;
    }

    unsigned int kernelsize = maps::Power<2 * RADIUS + 1, DIMS>::value;

    // Prepare random number generator.
    std::mt19937 gen(kRandomSeed1);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    // Randomize if necessary, otherwise create a "unit" kernel.
    if (random_kernel)
    {        
        for (unsigned int i = 0; i < kernelsize; ++i)
        {
            host_kernel[i] = ud(gen);
        }
    }
    else
    {
        memset(host_kernel, 0, sizeof(float) * kernelsize);
        host_kernel[kernelsize / 2] = 1.0f;
    }

    // Copy convolution kernel to all GPUs.
    for (int i = 0; i < num_gpus; ++i)
    {
        ASSERT_EQ(cudaSetDevice(i), cudaSuccess);
        ASSERT_EQ(cudaMemcpyToSymbol(kConvKernel, &host_kernel[0], 
                                     sizeof(float) * kernelsize), 
                  cudaSuccess);
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    size_t total_size = 1;
    for (int i = 0; i < DIMS; ++i)
        total_size *= buffer_size[i];

    maps::WindowSingleGPU<float, DIMS, BW, ((DIMS >= 2) ? BH : 1),
                    ((DIMS >= 3) ? BD : 1), RADIUS, IPX, IPY, IPZ, BORDERS> win;
    win.m_ptr = device_in;
    win.m_stride = buffer_size[0];
    
    maps::StructuredInjectiveSingleGPU<float, DIMS, BW, ((DIMS >= 2) ? BH : 1),
                                             ((DIMS >= 3) ? BD : 1), IPX, IPY> soout;
    soout.m_ptr = device_out;
    soout.m_stride = buffer_size[0];
    
    for (int i = 0; i < DIMS; ++i)
    {
        win.m_dimensions[i] = buffer_size[i];
        soout.m_dimensions[i] = buffer_size[i];
    }

    dim3 grid_dims(maps::RoundUp(buffer_size[0], BW * IPX),
                   (DIMS >= 2) ? maps::RoundUp(buffer_size[1], BH * IPY) : 1,
                   (DIMS >= 3) ? maps::RoundUp(buffer_size[2], BD * IPZ) : 1);
    dim3 block_dims(BW, 
                    (DIMS >= 2) ? BH : 1,
                    (DIMS >= 3) ? BD : 1);
    
    MAPSConvolution<DIMS, BW, ((DIMS >= 2) ? BH : 1), 
                    ((DIMS >= 3) ? BD : 1), RADIUS, BORDERS, IPX, IPY, IPZ> <<<grid_dims, block_dims>>>(win, soout);
    CUASSERT_NOERR(cudaGetLastError());
    CUASSERT_NOERR(cudaDeviceSynchronize());

    CUASSERT_NOERR(cudaMemcpy(buffer_out, device_out, sizeof(float) * total_size, cudaMemcpyDeviceToHost));

    // If unit kernel, verify by testing equality to source buffer.
    if (!random_kernel)
    {
        for (size_t i = 0; i < total_size; ++i)
            ASSERT_LE(fabs(buffer_in[i] - buffer_out[i]), kEpsilon)
                << "Unequal values in unit convolution, index " << i
                << " (" << buffer_out[i] << " != " << buffer_in[i]
                << ") when convolving a " << BufferSize<DIMS>(buffer_size)
                << " buffer with a kernel with radius " << RADIUS
                << ", using " << BoundariesToString(BORDERS) << " boundaries";
    }
    else // Otherwise, use regression
    {
        ConvRegression<DIMS, RADIUS, BORDERS>::RunConvRegression(
            device_in, device_out, buffer_regression, buffer_size);

        for (size_t i = 0; i < total_size; ++i)
            ASSERT_LE(fabs(buffer_regression[i] - buffer_out[i]), kEpsilon)
                << "Unequal values in randomized convolution, index " << i
                << " (" << buffer_out[i] << " != " << buffer_regression[i]
                << ") when convolving a " << BufferSize<DIMS>(buffer_size)
                << " buffer with a kernel with radius " << RADIUS
                << ", using " << BoundariesToString(BORDERS) << " boundaries";
    }
}

template<int DIMS, bool RANDOMIZED, int RAD_COUNTER, int RAD_END>
struct WindowRadiusLoop
{
    static void Loop(unsigned int size[DIMS], float *hKernel, float *hIn, float *hOut,
                     float *hRegression, float *dRegressionIn, float *dRegressionOut)
    {
        TestWindow<DIMS, RAD_COUNTER, maps::WB_WRAP>(RANDOMIZED, size, hKernel, hIn, hOut, 
                                                     hRegression, dRegressionIn, dRegressionOut);
        TestWindow<DIMS, RAD_COUNTER, maps::WB_COPY>(RANDOMIZED, size, hKernel, hIn, hOut, 
                                                     hRegression, dRegressionIn, dRegressionOut);
        TestWindow<DIMS, RAD_COUNTER, maps::WB_ZERO>(RANDOMIZED, size, hKernel, hIn, hOut, 
                                                     hRegression, dRegressionIn, dRegressionOut);

        WindowRadiusLoop<DIMS, RANDOMIZED, RAD_COUNTER + 1, RAD_END>::Loop(size, hKernel, hIn, hOut,
                                                                           hRegression, dRegressionIn, 
                                                                           dRegressionOut);
    }
};

template<int DIMS, bool RANDOMIZED, int RAD_END>
struct WindowRadiusLoop<DIMS, RANDOMIZED, RAD_END, RAD_END>
{
    static void Loop(unsigned int size[DIMS], float *hKernel, float *hIn, float *hOut,
                     float *hRegression, float *dRegressionIn, float *dRegressionOut)
    {
        TestWindow<DIMS, RAD_END, maps::WB_WRAP>(RANDOMIZED, size, hKernel, hIn, hOut,
                                                 hRegression, dRegressionIn, dRegressionOut);
        TestWindow<DIMS, RAD_END, maps::WB_COPY>(RANDOMIZED, size, hKernel, hIn, hOut,
                                                 hRegression, dRegressionIn, dRegressionOut);
        TestWindow<DIMS, RAD_END, maps::WB_ZERO>(RANDOMIZED, size, hKernel, hIn, hOut,
                                                 hRegression, dRegressionIn, dRegressionOut);
    }
};

template<int DIMS, bool RANDOMIZED, int RAD_COUNTER, int RAD_END>
struct WindowRadiusLoopNoChecks
{
    static void Loop(unsigned int size[DIMS], float *hKernel, float *hIn, float *hOut,
                     float *hRegression, float *dRegressionIn, float *dRegressionOut)
    {
        TestWindow<DIMS, RAD_COUNTER, maps::WB_NOCHECKS>(RANDOMIZED, size, hKernel, hIn, hOut,
                                                         hRegression, dRegressionIn, dRegressionOut);

        WindowRadiusLoopNoChecks<DIMS, RANDOMIZED, RAD_COUNTER + 1, RAD_END>::Loop(
            size, hKernel, hIn, hOut, hRegression, dRegressionIn, dRegressionOut);
    }
};

template<int DIMS, bool RANDOMIZED, int RAD_END>
struct WindowRadiusLoopNoChecks<DIMS, RANDOMIZED, RAD_END, RAD_END>
{
    static void Loop(unsigned int size[DIMS], float *hKernel, float *hIn, float *hOut,
                     float *hRegression, float *dRegressionIn, float *dRegressionOut)
    {
        TestWindow<DIMS, RAD_END, maps::WB_NOCHECKS>(RANDOMIZED, size, hKernel, hIn, hOut,
                                                     hRegression, dRegressionIn, dRegressionOut);
    }
};

TEST(Window, Window1DUnit)
{
    // Prepare buffers in advance.
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    size_t max_buffer_size = kSizes[num_sizes - 1] * sizeof(float);
    size_t max_kernel_size = (MAX_WINDOW_RADIUS * 2 + 1) * sizeof(float);
    maps::pinned_vector<float> hKernel(max_kernel_size, 0.0f),
        hBuffIn(max_buffer_size),
        hBuffOut(max_buffer_size),
        hBuffRegression(max_buffer_size);
    float *dRegressionIn = nullptr, *dRegressionOut = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionIn, max_buffer_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionOut, max_buffer_size), cudaSuccess);

    // Prepare random number generator.
    std::mt19937 gen(kRandomSeed2);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    for (size_t i = 0; i < max_buffer_size / sizeof(float); ++i)
        hBuffIn[i] = ud(gen);
    
    ASSERT_EQ(cudaMemcpy(dRegressionIn, &hBuffIn[0], max_buffer_size, 
        cudaMemcpyHostToDevice), cudaSuccess);

    // Loop over the various buffer sizes.
    for (int size_ind = 0; size_ind < sizeof(kSizes) / sizeof(unsigned int); ++size_ind)
    {
        unsigned int size[1] = { kSizes[size_ind] };

        WindowRadiusLoop<1, false, MIN_WINDOW_RADIUS, MAX_WINDOW_RADIUS>::Loop(
            size, &hKernel[0], &hBuffIn[0], &hBuffOut[0], &hBuffRegression[0],
            dRegressionIn, dRegressionOut);
    }

    ASSERT_EQ(cudaFree(dRegressionIn), cudaSuccess);
    ASSERT_EQ(cudaFree(dRegressionOut), cudaSuccess);
}


TEST(Window, Window1DRandom)
{ 
    // Prepare buffers in advance.
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    size_t max_buffer_size = kSizes[num_sizes - 1] * sizeof(float);
    size_t max_kernel_size = (MAX_WINDOW_RADIUS * 2 + 1) * sizeof(float);
    maps::pinned_vector<float> hKernel(max_kernel_size, 0.0f),
        hBuffIn(max_buffer_size),
        hBuffOut(max_buffer_size),
        hBuffRegression(max_buffer_size);
    float *dRegressionIn = nullptr, *dRegressionOut = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionIn, max_buffer_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionOut, max_buffer_size), cudaSuccess);

    // Prepare random number generator.
    std::mt19937 gen(kRandomSeed2);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    for (size_t i = 0; i < max_buffer_size / sizeof(float); ++i)
        hBuffIn[i] = ud(gen);

    ASSERT_EQ(cudaMemcpy(dRegressionIn, &hBuffIn[0], max_buffer_size,
        cudaMemcpyHostToDevice), cudaSuccess);

    // Loop over the various buffer sizes.
    for (int size_ind = 0; size_ind < sizeof(kSizes) / sizeof(unsigned int); ++size_ind)
    {
        unsigned int size[1] = { kSizes[size_ind] };

        WindowRadiusLoop<1, true, MIN_WINDOW_RADIUS, MAX_WINDOW_RADIUS>::Loop(
            size, &hKernel[0], &hBuffIn[0], &hBuffOut[0], &hBuffRegression[0],
            dRegressionIn, dRegressionOut);
    }

    ASSERT_EQ(cudaFree(dRegressionIn), cudaSuccess);
    ASSERT_EQ(cudaFree(dRegressionOut), cudaSuccess);
}

TEST(Window, Window2DUnit)
{
    // Prepare buffers in advance.
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    size_t max_buffer_size = kSizes[num_sizes - 1] * kSizes[num_sizes - 1] * sizeof(float);
    size_t max_kernel_size = (MAX_WINDOW_RADIUS * 2 + 1) * 
        (MAX_WINDOW_RADIUS * 2 + 1) * sizeof(float);
    maps::pinned_vector<float> hKernel(max_kernel_size, 0.0f),
        hBuffIn(max_buffer_size),
        hBuffOut(max_buffer_size),
        hBuffRegression(max_buffer_size);
    float *dRegressionIn = nullptr, *dRegressionOut = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionIn, max_buffer_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionOut, max_buffer_size), cudaSuccess);

    // Prepare random number generator.
    std::mt19937 gen(kRandomSeed2);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    for (size_t i = 0; i < max_buffer_size / sizeof(float); ++i)
        hBuffIn[i] = ud(gen);

    ASSERT_EQ(cudaMemcpy(dRegressionIn, &hBuffIn[0], max_buffer_size,
        cudaMemcpyHostToDevice), cudaSuccess);

    // Loop over the various buffer sizes.
    for (int size_ind1 = 0; size_ind1 < sizeof(kSizes) / sizeof(unsigned int); ++size_ind1)
    {
        for (int size_ind2 = 0; size_ind2 < sizeof(kSizes) / sizeof(unsigned int); ++size_ind2)
        {
            unsigned int size[2] = { kSizes[size_ind1], kSizes[size_ind2] };

            WindowRadiusLoop<2, false, MIN_WINDOW_RADIUS, MAX_WINDOW_RADIUS>::Loop(
                size, &hKernel[0], &hBuffIn[0], &hBuffOut[0], &hBuffRegression[0],
                dRegressionIn, dRegressionOut);
        }
    }

    ASSERT_EQ(cudaFree(dRegressionIn), cudaSuccess);
    ASSERT_EQ(cudaFree(dRegressionOut), cudaSuccess);
}

TEST(Window, Window2DRandom)
{
    // Prepare buffers in advance.
    int num_sizes = sizeof(kSizes) / sizeof(unsigned int);
    size_t max_buffer_size = kSizes[num_sizes - 1] * kSizes[num_sizes - 1] * sizeof(float);
    size_t max_kernel_size = (MAX_WINDOW_RADIUS * 2 + 1) *
        (MAX_WINDOW_RADIUS * 2 + 1) * sizeof(float);
    maps::pinned_vector<float> hKernel(max_kernel_size, 0.0f),
        hBuffIn(max_buffer_size),
        hBuffOut(max_buffer_size),
        hBuffRegression(max_buffer_size);
    float *dRegressionIn = nullptr, *dRegressionOut = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionIn, max_buffer_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionOut, max_buffer_size), cudaSuccess);

    // Prepare random number generator.
    std::mt19937 gen(kRandomSeed2);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    for (size_t i = 0; i < max_buffer_size / sizeof(float); ++i)
        hBuffIn[i] = ud(gen);

    ASSERT_EQ(cudaMemcpy(dRegressionIn, &hBuffIn[0], max_buffer_size,
        cudaMemcpyHostToDevice), cudaSuccess);

    // Loop over the various buffer sizes.
    for (int size_ind1 = 0; size_ind1 < sizeof(kSizes) / sizeof(unsigned int); ++size_ind1)
    {
        for (int size_ind2 = 0; size_ind2 < sizeof(kSizes) / sizeof(unsigned int); ++size_ind2)
        {
            unsigned int size[2] = { kSizes[size_ind1], kSizes[size_ind2] };

            WindowRadiusLoop<2, true, MIN_WINDOW_RADIUS, MAX_WINDOW_RADIUS>::Loop(
                size, &hKernel[0], &hBuffIn[0], &hBuffOut[0], &hBuffRegression[0],
                dRegressionIn, dRegressionOut);
        }
    }

    ASSERT_EQ(cudaFree(dRegressionIn), cudaSuccess);
    ASSERT_EQ(cudaFree(dRegressionOut), cudaSuccess);
}

TEST(Window, Window2DILP)
{
    unsigned int size[2] = { 1200, 2400 };

    // Prepare buffers in advance.
    size_t max_buffer_size = size[0] * size[1] * sizeof(float);
    size_t max_kernel_size = (MAX_WINDOW_RADIUS * 2 + 1) *
        (MAX_WINDOW_RADIUS * 2 + 1) * sizeof(float);
    maps::pinned_vector<float> hKernel(max_kernel_size, 0.0f),
        hBuffIn(max_buffer_size),
        hBuffOut(max_buffer_size),
        hBuffRegression(max_buffer_size);
    float *dRegressionIn = nullptr, *dRegressionOut = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionIn, max_buffer_size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRegressionOut, max_buffer_size), cudaSuccess);

    // Prepare random number generator.
    std::mt19937 gen(kRandomSeed2);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    for (size_t i = 0; i < max_buffer_size / sizeof(float); ++i)
        hBuffIn[i] = ud(gen);

    ASSERT_EQ(cudaMemcpy(dRegressionIn, &hBuffIn[0], max_buffer_size,
        cudaMemcpyHostToDevice), cudaSuccess);

    #define TEST_WINDOW_ILP(IPX, IPY) do {                                              \
        TestWindow<2, 0, maps::WB_WRAP, IPX, IPY>(                                      \
            true, size, &hKernel[0], &hBuffIn[0], &hBuffOut[0], &hBuffRegression[0],    \
            dRegressionIn, dRegressionOut);                                             \
        TestWindow<2, 1, maps::WB_WRAP, IPX, IPY>(                                      \
            true, size, &hKernel[0], &hBuffIn[0], &hBuffOut[0], &hBuffRegression[0],    \
            dRegressionIn, dRegressionOut);                                             \
    } while (0)


    // Test various ILP configurations
    TEST_WINDOW_ILP(1, 1);
    TEST_WINDOW_ILP(2, 1);
    TEST_WINDOW_ILP(3, 1);
    TEST_WINDOW_ILP(4, 1);
    TEST_WINDOW_ILP(1, 2);
    TEST_WINDOW_ILP(1, 3);
    TEST_WINDOW_ILP(4, 2);
    TEST_WINDOW_ILP(5, 3);
    TEST_WINDOW_ILP(3, 5);
    TEST_WINDOW_ILP(8, 1);
    TEST_WINDOW_ILP(10, 1);

    #undef TEST_WINDOW_ILP

    
    ASSERT_EQ(cudaFree(dRegressionIn), cudaSuccess);
    ASSERT_EQ(cudaFree(dRegressionOut), cudaSuccess);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_WIDTH, int RADIUS>
__global__ void RelativeIndex1DKernel(maps::WindowSingleGPU<float, 1, BLOCK_WIDTH, 1, 1, RADIUS, 1, 1, 1, maps::WB_WRAP> in,
                                      maps::StructuredInjectiveSingleGPU<float, 1, BLOCK_WIDTH, 1, 1> out)
{
    __shared__ typename decltype(in)::SharedData sdata;

    in.init(sdata);
    out.init();

    if (out.Items() == 0)
        return;

    // Only use the "O" indices
    // O X X

    *out.begin() = in.at(-1);

    out.commit();
}

TEST(Window, RelativeIndex1D)
{
    enum
    {
        BLOCK_WIDTH = 32,
        BLOCKS = 5,
        TOTAL_SIZE = BLOCK_WIDTH * BLOCKS,
    };

    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in,  sizeof(float) * TOTAL_SIZE));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float) * TOTAL_SIZE));

    // Initialize input
    std::vector<float> in_val(TOTAL_SIZE), out_val(TOTAL_SIZE);

    for (int x = 0; x < TOTAL_SIZE; ++x)
    {
        in_val[x] = (float)x;
        out_val[x] = 0.0f;
    }
    
    // Copy input
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val[0], sizeof(float) * TOTAL_SIZE, cudaMemcpyHostToDevice));

    // Create structures
    maps::WindowSingleGPU<float, 1, BLOCK_WIDTH, 1, 1, 1, 1, 1, 1, maps::WB_WRAP> win;
    win.m_ptr = d_in;
    win.m_stride = win.m_dimensions[0] = TOTAL_SIZE;

    maps::StructuredInjectiveSingleGPU<float, 1, BLOCK_WIDTH, 1, 1> soout;
    soout.m_ptr = d_out;
    soout.m_stride = soout.m_dimensions[0] = TOTAL_SIZE;

    // Run test
    RelativeIndex1DKernel<BLOCK_WIDTH, 1><<<BLOCKS, BLOCK_WIDTH>>>(win, soout);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(float) * TOTAL_SIZE, cudaMemcpyDeviceToHost));

    for (int i = 0; i < TOTAL_SIZE; ++i)
        ASSERT_EQ(out_val[i], in_val[maps::Wrap(i - 1, TOTAL_SIZE)]) << "at index " << i;

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}


template <int BLOCK_WIDTH, int BLOCK_HEIGHT, int RADIUS>
__global__ void RelativeIndex2DKernel(maps::WindowSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, RADIUS, 1, 1, 1, maps::WB_WRAP> in,
                                      maps::StructuredInjectiveSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> out)
{
    __shared__ typename decltype(in)::SharedData sdata;

    in.init(sdata);
    out.init();

    if (out.Items() == 0)
        return;

    // Only use the "O" indices
    // X X X
    // X X X
    // X X O

    *out.begin() = in.at(1, 1);

    out.commit();
}


TEST(Window, RelativeIndex2D)
{
    enum
    {
        BLOCK_WIDTH = 32,
        BLOCK_HEIGHT = 16,
        XBLOCKS = 2,
        YBLOCKS = 3,
        TOTAL_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * XBLOCKS * YBLOCKS,

        TOTAL_WIDTH = BLOCK_WIDTH * XBLOCKS,
        TOTAL_HEIGHT = BLOCK_HEIGHT * YBLOCKS,
    };

    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in,  sizeof(float) * TOTAL_SIZE));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float) * TOTAL_SIZE));

    // Initialize input
    std::vector<float> in_val(TOTAL_SIZE), out_val(TOTAL_SIZE);

    for (int x = 0; x < TOTAL_SIZE; ++x)
    {
        in_val[x] = (float)x;
        out_val[x] = 0.0f;
    }
    
    // Copy input
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val[0], sizeof(float) * TOTAL_SIZE, cudaMemcpyHostToDevice));

    // Create structures
    maps::WindowSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1, 1, 1, 1, maps::WB_WRAP> win;
    win.m_ptr = d_in;
    win.m_stride = win.m_dimensions[0] = TOTAL_WIDTH;
    win.m_dimensions[1] = TOTAL_HEIGHT;

    maps::StructuredInjectiveSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> soout;
    soout.m_ptr = d_out;
    soout.m_stride = soout.m_dimensions[0] = TOTAL_WIDTH;
    soout.m_dimensions[1] = TOTAL_HEIGHT;

    // Run test
    RelativeIndex2DKernel<BLOCK_WIDTH, BLOCK_HEIGHT, 1> <<<dim3(XBLOCKS, YBLOCKS), dim3(BLOCK_WIDTH, BLOCK_HEIGHT)>>>(win, soout);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(float) * TOTAL_SIZE, cudaMemcpyDeviceToHost));

    for (int y = 0; y < TOTAL_HEIGHT; ++y)
    {
        for (int x = 0; x < TOTAL_WIDTH; ++x)
        {
            ASSERT_EQ(out_val[y * TOTAL_WIDTH + x], 
                      in_val[maps::Wrap(y + 1, TOTAL_HEIGHT) * TOTAL_WIDTH + maps::Wrap(x + 1, TOTAL_WIDTH)])
                      << "at index (" << x << ", " << y << ")";
        }
    }

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}


template <int BLOCK_WIDTH, int BLOCK_HEIGHT, int RADIUS>
__global__ void RelativeIndexAligned2DKernel(maps::WindowSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, RADIUS, 1, 1, 1, maps::WB_WRAP> in,
                                             maps::StructuredInjectiveSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> out)
{
    __shared__ typename decltype(in)::SharedData sdata;

    in.init(sdata);
    out.init();

    if (out.Items() == 0)
        return;

    // Only use the "O" indices
    // X O X
    // O X O
    // X O X

    #pragma unroll
    MAPS_FOREACH(oiter, out)
    {
        *oiter = in.aligned_at(oiter, 0, -1) + 
                 in.aligned_at(oiter, -1, 0) +
                 in.aligned_at(oiter,  1, 0) +
                 in.aligned_at(oiter,  0, 1);
    }

    out.commit();
}

TEST(Window, RelativeIndexAligned2D)
{
    enum
    {
        BLOCK_WIDTH = 32,
        BLOCK_HEIGHT = 16,
        XBLOCKS = 2,
        YBLOCKS = 3,
        TOTAL_SIZE = BLOCK_WIDTH * BLOCK_HEIGHT * XBLOCKS * YBLOCKS,

        TOTAL_WIDTH = BLOCK_WIDTH * XBLOCKS,
        TOTAL_HEIGHT = BLOCK_HEIGHT * YBLOCKS,
    };

    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in,  sizeof(float) * TOTAL_SIZE));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float) * TOTAL_SIZE));

    // Initialize input
    std::vector<float> in_val(TOTAL_SIZE), out_val(TOTAL_SIZE);

    for (int x = 0; x < TOTAL_SIZE; ++x)
    {
        in_val[x] = (float)x;
        out_val[x] = 0.0f;
    }
    
    // Copy input
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val[0], sizeof(float) * TOTAL_SIZE, cudaMemcpyHostToDevice));

    // Create structures
    maps::WindowSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 1, 1, 1, 1, maps::WB_WRAP> win;
    win.m_ptr = d_in;
    win.m_stride = win.m_dimensions[0] = TOTAL_WIDTH;
    win.m_dimensions[1] = TOTAL_HEIGHT;

    maps::StructuredInjectiveSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> soout;
    soout.m_ptr = d_out;
    soout.m_stride = soout.m_dimensions[0] = TOTAL_WIDTH;
    soout.m_dimensions[1] = TOTAL_HEIGHT;

    // Run test
    RelativeIndexAligned2DKernel<BLOCK_WIDTH, BLOCK_HEIGHT, 1> <<<dim3(XBLOCKS, YBLOCKS), dim3(BLOCK_WIDTH, BLOCK_HEIGHT)>>>(win, soout);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(float) * TOTAL_SIZE, cudaMemcpyDeviceToHost));

    for (int y = 0; y < TOTAL_HEIGHT; ++y)
    {
        for (int x = 0; x < TOTAL_WIDTH; ++x)
        {
            ASSERT_EQ(out_val[y * TOTAL_WIDTH + x], 
                      (in_val[maps::Wrap(y - 1, TOTAL_HEIGHT) * TOTAL_WIDTH + x] +
                       in_val[y * TOTAL_WIDTH + maps::Wrap(x - 1, TOTAL_WIDTH)] +
                       in_val[y * TOTAL_WIDTH + maps::Wrap(x + 1, TOTAL_WIDTH)] +
                       in_val[maps::Wrap(y + 1, TOTAL_HEIGHT) * TOTAL_WIDTH + x]))                
                      << "at index (" << x << ", " << y << ")";
        }
    }

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

template <int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void NoRadiusSingleGPUKernel(maps::WindowSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 0, 1, 1, 1, maps::WB_ZERO> in,
                                        maps::StructuredInjectiveSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> out)
{
    MAPS_INIT(in, out);

    if (out.Items() == 0)
        return;

    #pragma unroll
    MAPS_FOREACH(oiter, out)
    {
        *oiter = *in.align(oiter);
    }

    out.commit();
}

template <int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void NoRadiusMultiGPUKernel(MAPS_MULTIDEF2,
                                       maps::Window<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 0, 1, 1, 1, maps::WB_ZERO> in,
                                       maps::StructuredInjective2D<float, BLOCK_WIDTH, BLOCK_HEIGHT> out)
{
    MAPS_MULTI_INITVARS(in, out);

    if (out.Items() == 0)
        return;

    #pragma unroll
    MAPS_FOREACH(oiter, out)
    {
        *oiter = *in.align(oiter);
    }

    out.commit();
}

TEST(Window, NoRadiusSingleGPU)
{
    enum
    {
        BLOCK_WIDTH = 32,
        BLOCK_HEIGHT = 16,
        TOTAL_WIDTH = 5000,
        TOTAL_HEIGHT = 38,

        XBLOCKS = (TOTAL_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        YBLOCKS = (TOTAL_HEIGHT + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
        TOTAL_SIZE = TOTAL_WIDTH * TOTAL_HEIGHT,
    };

    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in,  sizeof(float) * TOTAL_SIZE));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float) * TOTAL_SIZE));

    // Initialize input
    std::vector<float> in_val(TOTAL_SIZE), out_val(TOTAL_SIZE);

    for (int x = 0; x < TOTAL_SIZE; ++x)
    {
        in_val[x] = (float)x;
        out_val[x] = 0.0f;
    }
    
    // Copy input
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val[0], sizeof(float) * TOTAL_SIZE, cudaMemcpyHostToDevice));

    // Create structures
    maps::WindowSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 0, 1, 1, 1, maps::WB_ZERO> win;
    win.m_ptr = d_in;
    win.m_stride = win.m_dimensions[0] = TOTAL_WIDTH;
    win.m_dimensions[1] = TOTAL_HEIGHT;

    maps::StructuredInjectiveSingleGPU<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1> soout;
    soout.m_ptr = d_out;
    soout.m_stride = soout.m_dimensions[0] = TOTAL_WIDTH;
    soout.m_dimensions[1] = TOTAL_HEIGHT;

    // Run test
    NoRadiusSingleGPUKernel<BLOCK_WIDTH, BLOCK_HEIGHT> <<<dim3(XBLOCKS, YBLOCKS), dim3(BLOCK_WIDTH, BLOCK_HEIGHT)>>>(win, soout);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(float) * TOTAL_SIZE, cudaMemcpyDeviceToHost));

    for (int y = 0; y < TOTAL_HEIGHT; ++y)
    {
        for (int x = 0; x < TOTAL_WIDTH; ++x)
        {
            ASSERT_EQ(out_val[y * TOTAL_WIDTH + x], in_val[y * TOTAL_WIDTH + x])
                      << "at index (" << x << ", " << y << ")";
        }
    }

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

TEST(Window, NoRadiusMultiGPU)
{
    enum
    {
        BLOCK_WIDTH = 32,
        BLOCK_HEIGHT = 16,
        TOTAL_WIDTH = 5000,
        TOTAL_HEIGHT = 38,

        XBLOCKS = (TOTAL_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        YBLOCKS = (TOTAL_HEIGHT + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
        TOTAL_SIZE = TOTAL_WIDTH * TOTAL_HEIGHT,
    };

    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in,  sizeof(float) * TOTAL_SIZE));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float) * TOTAL_SIZE));

    // Initialize input
    std::vector<float> in_val(TOTAL_SIZE), out_val(TOTAL_SIZE);

    for (int x = 0; x < TOTAL_SIZE; ++x)
    {
        in_val[x] = (float)x;
        out_val[x] = 0.0f;
    }
    
    // Copy input
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val[0], sizeof(float) * TOTAL_SIZE, cudaMemcpyHostToDevice));

    // Create structures
    maps::Window<float, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 0, 1, 1, 1, maps::WB_ZERO> win;
    win.m_ptr = d_in;
    win.m_stride = win.m_dimensions[0] = TOTAL_WIDTH;
    win.m_dimensions[1] = TOTAL_HEIGHT;
    win.m_containsApron = true;
    win.block_offset = 0;
    win.m_gridWidth = XBLOCKS;

    maps::StructuredInjective2D<float, BLOCK_WIDTH, BLOCK_HEIGHT> soout;
    soout.m_ptr = d_out;
    soout.m_stride = soout.m_dimensions[0] = TOTAL_WIDTH;
    soout.m_dimensions[1] = TOTAL_HEIGHT;
    soout.grid_dims = dim3(XBLOCKS, YBLOCKS);
    soout.blockId = make_uint3(0,0,0);

    // Run test
    NoRadiusMultiGPUKernel<BLOCK_WIDTH, BLOCK_HEIGHT> <<<dim3(XBLOCKS * YBLOCKS), dim3(BLOCK_WIDTH, BLOCK_HEIGHT)>>>(
        0, dim3(XBLOCKS, YBLOCKS), make_uint3(0,0,0), win, soout);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(float) * TOTAL_SIZE, cudaMemcpyDeviceToHost));

    for (int y = 0; y < TOTAL_HEIGHT; ++y)
    {
        for (int x = 0; x < TOTAL_WIDTH; ++x)
        {
            ASSERT_EQ(out_val[y * TOTAL_WIDTH + x], in_val[y * TOTAL_WIDTH + x])
                      << "at index (" << x << ", " << y << ")";
        }
    }

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}

TEST(Window, NoRadiusMAPSMulti)
{
    enum
    {
        BLOCK_WIDTH = 32,
        BLOCK_HEIGHT = 16,
        TOTAL_WIDTH = 5000,
        TOTAL_HEIGHT = 38,
        
        XBLOCKS = (TOTAL_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        YBLOCKS = (TOTAL_HEIGHT + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
        TOTAL_SIZE = TOTAL_WIDTH * TOTAL_HEIGHT,
    };

    // Allocate GPU memory
    maps::multi::Matrix<float> in(TOTAL_WIDTH, TOTAL_HEIGHT),
                               out(TOTAL_WIDTH, TOTAL_HEIGHT);

    // Initialize input
    std::vector<float> in_val(TOTAL_SIZE), out_val(TOTAL_SIZE);

    for (int x = 0; x < TOTAL_SIZE; ++x)
    {
        in_val[x] = (float)x;
        out_val[x] = 0.0f;
    }

    // Bind matrices
    in.Bind(&in_val[0]);
    out.Bind(&out_val[0]);

    maps::multi::Scheduler sched{0};

    sched.AnalyzeCall(dim3(), dim3(BLOCK_WIDTH, BLOCK_HEIGHT),
                      maps::multi::Window2D<float, BLOCK_WIDTH, BLOCK_HEIGHT, 0>(in),
                      maps::multi::StructuredInjectiveMatrixO<float>(out));

    // Run test
    sched.Invoke(NoRadiusMultiGPUKernel<BLOCK_WIDTH, BLOCK_HEIGHT>, dim3(), dim3(BLOCK_WIDTH, BLOCK_HEIGHT),
                 maps::multi::Window2D<float, BLOCK_WIDTH, BLOCK_HEIGHT, 0>(in),
                 maps::multi::StructuredInjectiveMatrixO<float>(out));
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    sched.Gather<false>(out);

    for (int y = 0; y < TOTAL_HEIGHT; ++y)
    {
        for (int x = 0; x < TOTAL_WIDTH; ++x)
        {
            ASSERT_EQ(out_val[y * TOTAL_WIDTH + x], in_val[y * TOTAL_WIDTH + x])
                << "at index (" << x << ", " << y << ")";
        }
    }
}
