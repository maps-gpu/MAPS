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
#include <chrono>
#include <random>

#include <vector>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <maps/maps.cuh>

// Options
// Comment to disable result verification against CPU
#define VERIFY_CPU

// Running parameters
enum
{
    BLOCK_WIDTH = 32,
    BLOCK_HEIGHT = 16,
    ELEMENTS_PER_THREAD_X = 4,
    ELEMENTS_PER_THREAD_Y = 1,

    IMAGE_WIDTH = 2048,
    IMAGE_HEIGHT = 2048,

    REPETITIONS = 100,

    MAX_RADIUS = 6
};

// Helper definition to obtain convolution kernel size from radius
#define CONV_DIM (2*RADIUS+1)

// Helper function for boundary conditions (clamp value to edge)
__host__ __device__ __forceinline__ int Clamp(int value, int max)
{
    if (value < 0) return 0;
    if (value >= max) return (max - 1);
    return value;
}

// Constant memory allocation for optimized convolution kernels
__constant__ float kConvKernel[(MAX_RADIUS * 2 + 1)*(MAX_RADIUS * 2 + 1)];

///////////////////////////////////////////////////////////////////
// Helper classes

struct ScopedTime
{
    std::chrono::time_point<std::chrono::high_resolution_clock> m_begin;
    ScopedTime() { m_begin = std::chrono::high_resolution_clock::now(); }
    ~ScopedTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        printf("%lf ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - m_begin).count() / 1000.0 / (double)REPETITIONS);
    }

};

///////////////////////////////////////////////////////////////////
// CPU version

template <int RADIUS>
void Conv2_CPU(const float *in, float *out,
               int width, int height, int stride,
               const float conv_kernel[CONV_DIM*CONV_DIM])
{
    // For each input pixel
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float result = 0.0f;

            // Convolve
            for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
                for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                    result += in[Clamp(y + ky, height) * stride + Clamp(x + kx, width)] *
                              conv_kernel[(ky + RADIUS) * CONV_DIM + (kx + RADIUS)];
                }
            }

            // Fill output image
            out[y * stride + x] = result;
        }
    }
}

///////////////////////////////////////////////////////////////////
// GPU version (Naive)

template <int RADIUS>
__global__ void Conv2Naive(const float *in, int width, int height, int stride,
                           const float *convKernel, float *out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float result = 0.0f;
    for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            result += in[Clamp(y + ky, height) * stride + Clamp(x + kx, width)] *
                convKernel[(ky + RADIUS) * CONV_DIM + (kx + RADIUS)];
        }
    }

    out[y * stride + x] = result;
}

///////////////////////////////////////////////////////////////////
// GPU version (Optimized)

template <int RADIUS>
__global__ void Conv2Optimized(const float *in, int width, int height, 
                               int stride, float *out) {
    int x = blockIdx.x * BLOCK_WIDTH, tidx = threadIdx.x;
    int y = blockIdx.y * BLOCK_HEIGHT, tidy = threadIdx.y;
    enum
    {
        DIAMETER = 2 * RADIUS,
        BLOCK_STRIDE = (BLOCK_WIDTH + DIAMETER),
    };
    
    // Allocate shared memory
    __shared__ float s_temp[BLOCK_STRIDE * (BLOCK_HEIGHT + DIAMETER)];

    // Load top-left portion of shared memory
    s_temp[BLOCK_STRIDE * tidy + tidx] = 
        in[Clamp(y + tidy - RADIUS, height) * stride +
           Clamp(x + tidx - RADIUS, width)];

    // Load shared memory edges
    if (tidx < DIAMETER) { // Right edge
        s_temp[BLOCK_STRIDE * tidy + 
               tidx + BLOCK_WIDTH] = in[Clamp(y + tidy - RADIUS, height) * stride +
                                        Clamp(x + tidx + BLOCK_WIDTH - RADIUS, width)];
    }
    if (tidy < DIAMETER) { // Bottom edge
        s_temp[BLOCK_STRIDE * (tidy + BLOCK_HEIGHT) + tidx] = 
            in[Clamp(y + tidy + BLOCK_HEIGHT - RADIUS, height) * stride +
               Clamp(x + tidx - RADIUS, width)];
    }
    if (tidx < DIAMETER && tidy < DIAMETER) { // Bottom-right edge
        s_temp[BLOCK_STRIDE * (tidy + BLOCK_HEIGHT) + 
               tidx + BLOCK_WIDTH] = in[Clamp(y + tidy + BLOCK_HEIGHT - RADIUS, height) * stride + 
                                        Clamp(x + tidx + BLOCK_WIDTH - RADIUS, width)];
    }
    // Synchronize thread-blocks
    __syncthreads();

    // Convolve
    float result = 0.0f;
    #pragma unroll
    for (int ky = 0; ky < CONV_DIM; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < CONV_DIM; ++kx) {
            result += s_temp[BLOCK_STRIDE * (tidy + ky) + 
                             (tidx + kx)] * kConvKernel[ky * CONV_DIM + kx];
        }
    }
    
    // Store outputs
    out[(y+tidy) * stride + (x+tidx)] = result;
}

///////////////////////////////////////////////////////////////////
// GPU version (MAPS)

template<int RADIUS>
__global__ void Conv2MAPS MAPS_MULTIDEF(
      maps::Window2D<float, BLOCK_WIDTH, BLOCK_HEIGHT, RADIUS, 
                     maps::ClampBoundaries, ELEMENTS_PER_THREAD_X, 
                     ELEMENTS_PER_THREAD_Y> in,
      maps::StructuredInjective2D<float, BLOCK_WIDTH, BLOCK_HEIGHT,
                                  ELEMENTS_PER_THREAD_X, 
                                  ELEMENTS_PER_THREAD_Y> out) {

    MAPS_MULTI_INITVARS(in, out);                  // Initialize multi-GPU abstraction and containers

    MAPS_FOREACH(oiter, out) {                     // Loop over output elements
        int i = 0;
        *oiter = 0.0f;

        MAPS_FOREACH_ALIGNED(iter, in, oiter) {    // For each output, loop over inputs according to pattern
            *oiter += *iter * kConvKernel[i++];
        }
    }

    out.commit();                                  // Write all outputs to global memory
}

///////////////////////////////////////////////////////////////////
// Host code

bool RegressionTest(const float *actual, const float *expected,
                    int width, int height, int stride)
{
#ifdef VERIFY_CPU
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float actual_val = actual[y * stride + x];
            float expected_val = expected[y * stride + x];
            if (fabs(actual_val - expected_val) > 1e-4)
            {
                printf("Error while comparing index (%d, %d): actual = %f, expected = %f\n",
                       x, y, actual_val, expected_val);
                return false;
            }
        }
    }
#endif
    return true;
}

template <int RADIUS>
void RunConv2(float *host_input, float *host_output)
{
    size_t sz = IMAGE_WIDTH*IMAGE_HEIGHT;

    // Prepare convolution kernel
    float conv_kernel[CONV_DIM*CONV_DIM] = {0};
    for (int i = 0; i < CONV_DIM*CONV_DIM; ++i)
        conv_kernel[i] = (float)(i + 1) / (float)(CONV_DIM*CONV_DIM);

    // Load kernel to constant memory
    MAPS_CUDA_CHECK(cudaMemcpyToSymbol(kConvKernel, conv_kernel, 
                                       CONV_DIM*CONV_DIM*sizeof(float)));

#ifdef VERIFY_CPU
    // Prepare CPU regression data
    std::vector<float> cpu_regression(IMAGE_WIDTH*IMAGE_HEIGHT, 0);
    Conv2_CPU<RADIUS>(host_input, &cpu_regression[0], IMAGE_WIDTH,
                      IMAGE_HEIGHT, IMAGE_WIDTH, conv_kernel);
#endif

    // Kernel dimensions
    dim3 grid_dims(maps::RoundUp(IMAGE_WIDTH, BLOCK_WIDTH),
                   maps::RoundUp(IMAGE_HEIGHT, BLOCK_HEIGHT));
    dim3 block_dims(BLOCK_WIDTH, BLOCK_HEIGHT);

    ////////////////////////////////////////////////////////
    // Naive

    // Allocate GPU memory
    float *dev_naiveinput, *dev_naiveoutput, *dev_convkernel;
    MAPS_CUDA_CHECK(cudaMalloc(&dev_naiveinput, sz * sizeof(float)));
    MAPS_CUDA_CHECK(cudaMalloc(&dev_naiveoutput, sz * sizeof(float)));
    MAPS_CUDA_CHECK(cudaMalloc(&dev_convkernel, CONV_DIM * CONV_DIM * sizeof(float)));

    // Copy memory to GPU
    MAPS_CUDA_CHECK(cudaMemcpy(dev_naiveinput, host_input, sz * sizeof(float), 
                               cudaMemcpyHostToDevice));
    MAPS_CUDA_CHECK(cudaMemcpy(dev_convkernel, conv_kernel, CONV_DIM * CONV_DIM * sizeof(float), 
                               cudaMemcpyHostToDevice));

    printf("Naive %dx%d: ", CONV_DIM, CONV_DIM);
    MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    {
        ScopedTime timer;

        // Run kernels
        for (int i = 0; i < REPETITIONS; ++i)
        {
            Conv2Naive<RADIUS> <<<grid_dims, block_dims>>>(
                dev_naiveinput, IMAGE_WIDTH, IMAGE_HEIGHT, 
                IMAGE_WIDTH, dev_convkernel, dev_naiveoutput);
        }

        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy to host
    MAPS_CUDA_CHECK(cudaMemcpy(host_output, dev_naiveoutput, sz * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    RegressionTest(host_output, &cpu_regression[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH);

    // Free GPU memory
    MAPS_CUDA_CHECK(cudaFree(dev_naiveinput));
    MAPS_CUDA_CHECK(cudaFree(dev_naiveoutput));
    MAPS_CUDA_CHECK(cudaFree(dev_convkernel));

    ////////////////////////////////////////////////////////
    // Optimized
    
    // Allocate GPU memory
    float *dev_optinput, *dev_optoutput;
    size_t stride = 0;
    MAPS_CUDA_CHECK(cudaMallocPitch(&dev_optinput, &stride, IMAGE_WIDTH * sizeof(float), IMAGE_HEIGHT));
    MAPS_CUDA_CHECK(cudaMallocPitch(&dev_optoutput, &stride, IMAGE_WIDTH * sizeof(float), IMAGE_HEIGHT));

    // Copy memory to GPU
    MAPS_CUDA_CHECK(cudaMemcpy2D(dev_optinput, stride, 
                                 host_input, IMAGE_WIDTH * sizeof(float),
                                 IMAGE_WIDTH * sizeof(float), IMAGE_HEIGHT,
                                 cudaMemcpyHostToDevice));


    printf("Optimized %dx%d: ", CONV_DIM, CONV_DIM);
    MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    {
        ScopedTime timer;
        
        // Run kernels
        for (int i = 0; i < REPETITIONS; ++i)
        {
            Conv2Optimized<RADIUS> <<<grid_dims, block_dims>>>(
                dev_optinput, IMAGE_WIDTH, IMAGE_HEIGHT, 
                (int)stride / sizeof(float), dev_optoutput);
        }

        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy to host
    MAPS_CUDA_CHECK(cudaMemcpy2D(host_output, IMAGE_WIDTH * sizeof(float), 
                                 dev_optoutput, stride, 
                                 IMAGE_WIDTH * sizeof(float), IMAGE_HEIGHT,
                                 cudaMemcpyDeviceToHost));

    // Verify
    RegressionTest(host_output, &cpu_regression[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH);

    // Free GPU memory
    MAPS_CUDA_CHECK(cudaFree(dev_optinput));
    MAPS_CUDA_CHECK(cudaFree(dev_optoutput));

    ////////////////////////////////////////////////////////
    // MAPS

    // Create scheduler and data structures
    maps::multi::Scheduler sched({ 0 }); // Only use first GPU
    maps::multi::Matrix<float> input(IMAGE_WIDTH, IMAGE_HEIGHT), 
                               output(IMAGE_WIDTH, IMAGE_HEIGHT);
    
    // Bind input and output matrices
    input.Bind(host_input); output.Bind(host_output);

    // Reserve memory for the matrices, given the input/output memory access patterns
    sched.AnalyzeCall(dim3(), block_dims,
                      maps::multi::Window2D<float, BLOCK_WIDTH, BLOCK_HEIGHT, RADIUS, 
                                            maps::ClampBoundaries, ELEMENTS_PER_THREAD_X, 
                                            ELEMENTS_PER_THREAD_Y>(input),
                      maps::multi::StructuredInjectiveMatrixO<float, ELEMENTS_PER_THREAD_X, 
                                                              ELEMENTS_PER_THREAD_Y>(output));

    // First invocation only copies memory to GPU
    sched.Invoke(Conv2MAPS<RADIUS>, dim3(), block_dims,
                 maps::multi::Window2D<float, BLOCK_WIDTH, BLOCK_HEIGHT, RADIUS, 
                                       maps::ClampBoundaries, ELEMENTS_PER_THREAD_X, 
                                       ELEMENTS_PER_THREAD_Y>(input),
                 maps::multi::StructuredInjectiveMatrixO<float, ELEMENTS_PER_THREAD_X, 
                                                         ELEMENTS_PER_THREAD_Y>(output));
    sched.WaitAll();

    printf("MAPS %dx%d: ", CONV_DIM, CONV_DIM);
    MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    {
        ScopedTime timer;

        for (int i = 0; i < REPETITIONS; ++i)
        {
            sched.Invoke(Conv2MAPS<RADIUS>, dim3(), block_dims,
                         maps::multi::Window2D<float, BLOCK_WIDTH, BLOCK_HEIGHT, RADIUS, 
                                               maps::ClampBoundaries, ELEMENTS_PER_THREAD_X, 
                                               ELEMENTS_PER_THREAD_Y>(input),
                         maps::multi::StructuredInjectiveMatrixO<float, ELEMENTS_PER_THREAD_X, 
                                                                 ELEMENTS_PER_THREAD_Y>(output));
        }
        sched.WaitAll();
        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }
    sched.Gather<false>(output);
    RegressionTest(host_output, &cpu_regression[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH);

    printf("\n");
}

int main(int argc, char **argv)
{
    size_t sz = IMAGE_WIDTH * IMAGE_HEIGHT;

    // Allocate input and output images
    std::vector<float> host_image(sz, 0.0f), host_result(sz, 0.0f);

    // Randomize
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Create random values
    for (size_t i = 0; i < sz; ++i)
        host_image[i] = dis(gen);

    // Run all convolution radius settings
    RunConv2<0>(&host_image[0], &host_result[0]);
    RunConv2<1>(&host_image[0], &host_result[0]);
    RunConv2<2>(&host_image[0], &host_result[0]);
    RunConv2<3>(&host_image[0], &host_result[0]);
    RunConv2<4>(&host_image[0], &host_result[0]);
    RunConv2<5>(&host_image[0], &host_result[0]);
    RunConv2<6>(&host_image[0], &host_result[0]);

    return 0;
}
