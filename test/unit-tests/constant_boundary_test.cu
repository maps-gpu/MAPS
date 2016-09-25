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

#include <maps/maps.cuh>
#include <maps/optional/constant_boundaries.cuh>

#include "cuda_gtest_utils.h"

static const int kWidth = 31;
static const int kHeight = 32;
#define MY_CONSTANT 3
static const float kConstant = 3.14f;

template <typename T, typename Boundaries>
__global__ void CBKernel(maps::Block2DXSingleGPU<T, 32, 32, 1,1,1, Boundaries> in, T *out)
{
    MAPS_INIT(in);

    int xind = threadIdx.x + blockIdx.x * blockDim.x;
    int yind = threadIdx.y + blockIdx.y * blockDim.y;
    if(xind >= in.m_dimensions[0] || yind >= in.m_dimensions[1])
        return;

    T result = T(0);

    // Perform the multiplication
    for (int j = 0; j < in.chunks(); ++j)
    {
        MAPS_FOREACH(iter, in)
        {
            result += *iter;
        }

        // Advance chunks efficiently
        maps::NextChunkAll(in);
    }


    // Output result
    out[yind * in.m_dimensions[0] + xind] = result;
}

TEST(ConstantBoundary, SingleGPU)
{
    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in, sizeof(float) * kWidth * kHeight));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float) * kWidth * kHeight));

    // Zero input
    CUASSERT_NOERR(cudaMemset(d_in, 0, sizeof(float) * kWidth * kHeight));
    CUASSERT_NOERR(cudaMemset(d_out, 0, sizeof(float) * kWidth * kHeight));

    // Set boundary parameters
    maps::ConstantBoundaries<MY_CONSTANT>::template SetConstant<float>(kConstant);

    maps::Block2DXSingleGPU<float, 32, 32, 1, 1, 1, maps::ConstantBoundaries<MY_CONSTANT>> in;

    in.m_ptr = d_in;
    in.m_dimensions[0] = kWidth;
    in.m_dimensions[1] = kHeight;
    in.m_stride = kWidth;

    // Run test (launch a 1-block kernel)
    CBKernel<float, maps::ConstantBoundaries<MY_CONSTANT>> <<<1, dim3(32, 32)>>>(in, d_out);
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    std::vector<float> out_val (kWidth * kHeight, 0.0f);
    CUASSERT_NOERR(cudaMemcpy(&out_val[0], d_out, sizeof(float) * kWidth * kHeight, cudaMemcpyDeviceToHost));

    for (int i = 0; i < kWidth * kHeight; ++i)
    {
        ASSERT_LE(fabs(out_val[i] - kConstant), 1e-6) << "Invalid value at index (" << (i % kWidth) << ", " << (i / kWidth)
                                                  << "): " << out_val[i];
    }

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}
