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

#include <gtest/gtest.h>

#include <maps/multi/multi.cuh>

#define FILL_VALUE 5

void FillTest(int ngpus)
{
    unsigned int arrsize = ngpus * 100;

    int num_gpus = 0;
    ASSERT_EQ(cudaGetDeviceCount(&num_gpus), cudaSuccess);

    ASSERT_GE(num_gpus, 1);

    std::vector<unsigned int> gpuids;
    for (int i = 0; i < ngpus; ++i)
        gpuids.push_back(i % num_gpus);

    // Initialize
    maps::multi::Scheduler sched(gpuids);

    maps::multi::Vector<uint8_t> vec (arrsize);

    sched.AnalyzeCall(dim3(arrsize), dim3(1, 1, 1),
                      maps::multi::Window<uint8_t, 1, 1, 1, 1, 0>(vec));
    sched.Fill(vec, FILL_VALUE);
    sched.WaitAll();

    std::vector<uint8_t> hvec(arrsize, 0);
    vec.Bind(&hvec[0]);

    ASSERT_EQ(sched.GatherReadOnly<false>(vec), true);

    for (unsigned int i = 0; i < arrsize; ++i)
    {
        ASSERT_EQ(hvec[i], FILL_VALUE) << "Index " << i << " not filled properly";
    }
}

TEST(Scheduler, Fill_1GPU)
{
    FillTest(1);
}

TEST(Scheduler, Fill_3GPUs)
{
    FillTest(3);
}

TEST(Scheduler, FillBeforeAnalyze)
{
    maps::multi::Scheduler sched;

    maps::multi::Vector<int> vec(10);

    ASSERT_EQ(sched.Fill(vec, FILL_VALUE), false);
}

struct u128_t
{
    uint64_t v[2];
};

void InternalGCP(const std::vector<void *>& parameters)
{
    unsigned char ucval = 129;
    float fval = 0.5f;
    double dval = 0.75;
    u128_t qval; qval.v[0] = 7777; qval.v[1] = 8888;

    unsigned char gucval;
    float gfval;
    double gdval;
    u128_t gqval;
    maps::multi::GetConstantParameter(parameters[1], gucval);
    maps::multi::GetConstantParameter(parameters[2], gfval);
    maps::multi::GetConstantParameter(parameters[3], gdval);
    maps::multi::GetConstantParameter(parameters[4], gqval);

    ASSERT_EQ(gucval, ucval);
    ASSERT_EQ(gfval, fval);
    ASSERT_EQ(gdval, dval);
    ASSERT_EQ(gqval.v[0], qval.v[0]);
    ASSERT_EQ(gqval.v[1], qval.v[1]);
}

bool GCPRoutine(void *context, int deviceIdx, cudaStream_t stream,
                const maps::multi::GridSegment& atom_segment,
                const std::vector<void *>& parameters,
                const std::vector<maps::multi::DatumSegment>& container_segments,
                const std::vector<maps::multi::DatumSegment>& container_allocation)
{
    InternalGCP(parameters);
    return true;
}

TEST(Scheduler, GetConstantParameter)
{
    maps::multi::Scheduler sched;
    maps::multi::Vector<int> vec(10);

    unsigned char ucval = 129;
    float fval = 0.5f;
    double dval = 0.75;
    u128_t qval; qval.v[0] = 7777; qval.v[1] = 8888;

    sched.AnalyzeCall(dim3(10), dim3(), maps::multi::StructuredInjectiveOutput<int, 1>(vec));
    sched.InvokeUnmodified(&GCPRoutine, nullptr, dim3(10),
                           maps::multi::StructuredInjectiveVectorO<int>(vec),
                           ucval, fval, dval, qval);
}
