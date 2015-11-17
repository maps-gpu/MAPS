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
#include <gtest/gtest.h>

#include <maps/input_containers/internal/io_common.cuh>
#include <maps/input_containers/internal/io_globaltoarray.cuh>
#include <maps/multi/multi.cuh>

// Test segmentation by FillGPUID kernel, make sure segmentation works for 1-4 GPUs with 1-3 Dimensions

template <int DIMS>
__global__ void FillGPUID MAPS_MULTIDEF(maps::StructuredInjectiveOutput<int, DIMS, 10, 1, 1> buffer)
{
    MAPS_MULTI_INIT();

    buffer.init();
    if (buffer.Items() == 0)
        return;

    *buffer.begin() = deviceIdx;

    buffer.commit();
}

template <typename T, int DIMS>
maps::multi::Datum<T, DIMS> CreateNDimDatum(int n);

template<>
maps::multi::Datum<int, 1> CreateNDimDatum(int n) { return maps::multi::Datum<int, 1>(n * 10); }

template<>
maps::multi::Datum<int, 2> CreateNDimDatum(int n) { return maps::multi::Datum<int, 2>(11, n * 10); }

template<>
maps::multi::Datum<int, 3> CreateNDimDatum(int n) { return maps::multi::Datum<int, 3>(11, 12, n * 10); }

template <int DIMS>
void MGPU_ND_Test(int test_gpus)
{
    int num_gpus = 0;
    ASSERT_EQ(cudaGetDeviceCount(&num_gpus), cudaSuccess);

    ASSERT_GE(num_gpus, 1);

    std::vector<unsigned int> gpuids;
    for (int i = 0; i < test_gpus; ++i)
        gpuids.push_back(i % num_gpus);

    // Initialize
    maps::multi::Scheduler sched (gpuids);
    maps::multi::Datum<int, DIMS> dat = CreateNDimDatum<int, DIMS>(test_gpus);
    
    size_t datum_size = 1;
    for (unsigned int i = 0; i < dat.GetDimensions(); ++i)
        datum_size *= dat.GetDimension(i);

    maps::pinned_vector<int> hdat(datum_size, 0);

    dat.Bind(&hdat[0]);
    
    // Analyze
    sched.AnalyzeCall(dim3(), dim3(10), 
                      maps::multi::StructuredInjectiveOutput<int, DIMS>(dat));

    // Invoke
    maps::multi::taskHandle_t task = 0;
    task = sched.Invoke(FillGPUID<DIMS>, dim3(), dim3(10),
                        maps::multi::StructuredInjectiveOutput<int, DIMS>(dat));
    ASSERT_NE(task, 0) << "Kernel invocation failed";

    // Gather
    ASSERT_EQ(sched.Gather<false>(dat), true) << "Gather failed";

    size_t device_segment_size = datum_size / test_gpus;

    // Verify
    for (int i = 0; i < test_gpus; ++i)
        for (int j = 0; j < device_segment_size; ++j)
            ASSERT_EQ(hdat[i * device_segment_size + j], i)
                << "Expected correct GPU ID "
                << i << " at index " << (i*device_segment_size + j)
                << " when running with " << test_gpus << " GPUs";
}

TEST(Segmentation, MGPU_1D)
{
    for (int i = 1; i <= 4; ++i)
        MGPU_ND_Test<1>(i);
}

TEST(Segmentation, MGPU_2D)
{
    for (int i = 1; i <= 4; ++i)
        MGPU_ND_Test<2>(i);
}

TEST(Segmentation, MGPU_3D)
{
    for (int i = 1; i <= 4; ++i)
        MGPU_ND_Test<3>(i);
}

TEST(DatumSegment, Exclude)
{
    maps::multi::DatumSegment a(2), b(2), c(2);

    a.m_offset = { 5, 5 };
    a.m_dimensions = { 20, 10 };

    // Nothing to exclude
    c = a;
    b.m_offset = { 50, 50 };
    b.m_dimensions = { 10, 10 };

    ASSERT_EQ(c.Exclude(b), true);
    EXPECT_EQ(c.m_offset[0], a.m_offset[0]);
    EXPECT_EQ(c.m_offset[1], a.m_offset[1]);
    EXPECT_EQ(c.m_dimensions[0], a.m_dimensions[0]);
    EXPECT_EQ(c.m_dimensions[1], a.m_dimensions[1]);

    // Right exclusion
    c = a;
    b.m_offset = { 8, 5 };
    b.m_dimensions = { 20, 10 };

    ASSERT_EQ(c.Exclude(b), true);
    EXPECT_EQ(c.m_offset[0], 5);
    EXPECT_EQ(c.m_offset[1], 5);
    EXPECT_EQ(c.m_dimensions[0], 3);
    EXPECT_EQ(c.m_dimensions[1], 10);

    // Top exclusion
    c = a;
    b.m_offset = { 0, 0 };
    b.m_dimensions = { 30, 6 };

    ASSERT_EQ(c.Exclude(b), true);
    EXPECT_EQ(c.m_offset[0], 5);
    EXPECT_EQ(c.m_offset[1], 6);
    EXPECT_EQ(c.m_dimensions[0], 20);
    EXPECT_EQ(c.m_dimensions[1], 9);

    // Bottom exclusion
    c = a;
    b.m_offset = { 0, 14 };
    b.m_dimensions = { 30, 6 };

    ASSERT_EQ(c.Exclude(b), true);
    EXPECT_EQ(c.m_offset[0], 5);
    EXPECT_EQ(c.m_offset[1], 5);
    EXPECT_EQ(c.m_dimensions[0], 20);
    EXPECT_EQ(c.m_dimensions[1], 9);

    // Invalid exclusion
    c = a;
    b.m_offset = { 6, 14 };
    b.m_dimensions = { 30, 6 };

    ASSERT_EQ(c.Exclude(b), false);
}

