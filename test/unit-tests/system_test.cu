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

#include <maps/input_containers/internal/io_common.cuh>
#include <maps/input_containers/internal/io_globalread.cuh>
#include <maps/input_containers/internal/io_globaltoshared.cuh>
#include <maps/input_containers/internal/io_globaltoarray.cuh>
#include <maps/input_containers/window.cuh>
#include <maps/multi/multi.cuh>

#include "cuda_gtest_utils.h"

__global__ void InliningKernel(maps::WindowSingleGPU<float, 2, 1, 1, 1, 0> in, 
                               maps::StructuredInjectiveSingleGPU<float, 2, 1, 1, 1> out)
{
    MAPS_INIT(in, out);

    #pragma unroll
    MAPS_FOREACH(oiter, out)
    {
        #pragma unroll
        MAPS_FOREACH_ALIGNED(iter, in, oiter)
        {
            *oiter = *iter;
        }
    }

    out.commit();
}

// If this test succeeds, the SASS should show a simple copy kernel.
TEST(SystemTests, Inlining)
{
    // Allocate GPU memory
    float *d_in = nullptr, *d_out = nullptr;
    CUASSERT_NOERR(cudaMalloc(&d_in, sizeof(float)));
    CUASSERT_NOERR(cudaMalloc(&d_out, sizeof(float)));

    // Copy input
    float in_val = 123.0f, out_val = 0.0f;
    CUASSERT_NOERR(cudaMemcpy(d_in, &in_val, sizeof(float), cudaMemcpyHostToDevice));

    // Create structures
    maps::WindowSingleGPU<float, 2, 1, 1, 1, 0> win;
    win.m_ptr = d_in;
    win.m_stride = win.m_dimensions[0] = win.m_dimensions[1] = 1;

    maps::StructuredInjectiveSingleGPU<float, 2, 1, 1, 1> soout;
    soout.m_ptr = d_out;
    soout.m_stride = soout.m_dimensions[0] = soout.m_dimensions[1] = 1;

    // Run test
    void *args[] = { &win, &soout };
    void *actual_kernel = (void *)InliningKernel;
    CUASSERT_NOERR(cudaLaunchKernel(actual_kernel, dim3(1, 1, 1), dim3(1, 1, 1), args));
    CUASSERT_NOERR(cudaDeviceSynchronize());

    // Copy output
    CUASSERT_NOERR(cudaMemcpy(&out_val, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_EQ(in_val, out_val);

    // Free GPU memory
    CUASSERT_NOERR(cudaFree(d_in));
    CUASSERT_NOERR(cudaFree(d_out));
}
