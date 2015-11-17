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

#include <maps/multi/multi.cuh>

#include <thread>

// Testing multi-threaded Invocation through the unmodified routine API
// verifying distribution to a thread per device and correct barrier synchronization 

struct PerDeviceSubContext
{
    std::vector<std::thread::id> thread_ids;
    std::vector<std::chrono::system_clock::time_point> times;

    std::vector<std::chrono::milliseconds> sleep_times;
    unsigned int iteration;

    PerDeviceSubContext() : thread_ids(), times(), iteration(0) { }

    void Sleep()
    {
        if (sleep_times.size() == 0) return;
        std::this_thread::sleep_for(sleep_times[(iteration++) % sleep_times.size()]);
    }
};

struct InvokeTestContext
{
    std::vector<PerDeviceSubContext> sub_contexts;

    InvokeTestContext(int ngpus, const std::vector<std::chrono::milliseconds>& sleepTimes)
        : sub_contexts(ngpus)
    {

        // Distributing the sleep times array equally to all sub-contexts 
        for (size_t i = 0; i < sleepTimes.size();)
        {
            for (size_t j = 0; j < ngpus && i < sleepTimes.size(); j++, i++)
            {
                sub_contexts[j].sleep_times.push_back(sleepTimes[i]);
            }
        }
    }
};

bool InvokeTestRoutine(void *context, int deviceIdx, cudaStream_t stream,
    const maps::multi::GridSegment& task_segment,
    const std::vector<void *>& parameters,
    const std::vector<maps::multi::DatumSegment>& container_segments,
    const std::vector<maps::multi::DatumSegment>& container_allocation)
{
    InvokeTestContext *c = (InvokeTestContext *)context;
    if (!c)
        return false;

    auto& sc = c->sub_contexts[deviceIdx];

    sc.times.push_back(std::chrono::high_resolution_clock::now());
    sc.thread_ids.push_back(std::this_thread::get_id());
    sc.Sleep();
    return true;
}

void VerifyInvocation(const InvokeTestContext& context, unsigned int iterations)
{
    // Verify basic invocation 

    size_t ngpus = context.sub_contexts.size();

    for (size_t i = 0; i < ngpus ; i++)
    {
        auto& sc = context.sub_contexts[i];
        ASSERT_EQ(iterations, sc.times.size())
            << "Routine was expected to be called "
            << iterations << " times, but was called"
            << sc.times.size() << " times";
    }
}

void VerifyDistribution(const InvokeTestContext& context)
{
    // Verify Distribution

    // The thread ids involved in routine invocation
    std::set<std::thread::id> cids;
    size_t ngpus = context.sub_contexts.size();

    for (size_t i = 0; i < ngpus; i++)
    {
        auto& sc = context.sub_contexts[i];
        std::set<std::thread::id> scids;

        for (auto& id : sc.thread_ids) {
            scids.insert(id);
        }

        // Expecting at least one thread id from the sub context
        EXPECT_GT(scids.size(), 0) << "No invocations of routine were determined";

        // Expecting a single thread id from the sub context
        EXPECT_EQ(1, scids.size())
            << "A single device was invoked by more then one thread";

        // Add the single thread to the context ids set
        cids.insert(*scids.begin());
    }

    // Expecting <ngpus> thread ids from the context
    EXPECT_EQ(ngpus, cids.size())
        << "Expecting exacly one invoker thread per device";

    // Expecting this main thread (scheduling thread) to not be involved in actual invocation
    EXPECT_EQ(cids.end(), cids.find(std::this_thread::get_id()))
        << "Expecting the main scheduling thread to stay out of actual invocations";
}

void VerifySynchronization(
    const InvokeTestContext& context, 
    unsigned int iterations, 
    std::chrono::system_clock::time_point start
    )
{
    // Verify Synchronization

    // Expecting barrier synchronization
    // This means all times of sub-contexts for each iteration should be 
    // in a non-overlapped time frame

    size_t ngpus = context.sub_contexts.size();
    auto prev_iteration_max = start;

    for (size_t j = 0; j < iterations; j++)
    {
        auto iteration_max = prev_iteration_max;

        for (size_t i = 0; i < ngpus; i++)
        {
            auto& sc = context.sub_contexts[i];
            auto time = sc.times[j];

            EXPECT_GE(time.time_since_epoch(), prev_iteration_max.time_since_epoch())
                << "Expecting all times of current iteration to be "
                << "after the maximal time of the previous iteration";

            if (time > iteration_max) {
                iteration_max = time;
            }
        }

        prev_iteration_max = iteration_max;
    }
}

void TestInvocation(int ngpus, unsigned int iterations, std::vector<std::chrono::milliseconds> sleepTimes)
{
    // Tests the multi-threaded invocation through the unmodified routine API

    ASSERT_GT(ngpus, 0);
    ASSERT_GT(iterations, 0);

    // An unused output vector 
    std::vector<int> hostO(ngpus);

    // Create GPU list
    int num_gpus;
    CUASSERT_NOERR(cudaGetDeviceCount(&num_gpus));

    std::vector<unsigned int> gpuids;
    for (int i = 0; i < ngpus; ++i)
        gpuids.push_back(i % num_gpus);

    // Create test context
    InvokeTestContext context(ngpus, sleepTimes);

    // Keeping start time for sync verify
    auto start = std::chrono::high_resolution_clock::now();

    // Create scheduler
    maps::multi::Scheduler sched(gpuids);

    // Define data structure to be used
    maps::multi::Vector<int> O(ngpus);

    sched.AnalyzeCall(dim3(), dim3(), maps::multi::StructuredInjectiveVectorO<int>(O));

    for (size_t i = 0; i < iterations; i++)
    {
        // Invoke
        maps::multi::taskHandle_t task = 0;
        task = sched.InvokeUnmodified(
            InvokeTestRoutine,
            &context,
            dim3(),
            maps::multi::StructuredInjectiveVectorO<int>(O)
            );
        ASSERT_NE(task, 0) << "Invocation failed";
    }

    sched.WaitAll();
    for (int i = 0; i < num_gpus; i++)
    {
        CUASSERT_NOERR(cudaSetDevice(i));
        CUASSERT_NOERR(cudaDeviceSynchronize());
    }

    // Gather back to host (the data is not used for the test)
    O.Bind(&hostO[0]);
    EXPECT_TRUE(maps::multi::Gather(sched, O)) << "Gather failed";

    VerifyInvocation(context, iterations);
    VerifyDistribution(context);
    VerifySynchronization(context, iterations, start);
}

TEST(Invocation, Invoker1Distribution)
{
    TestInvocation(1, 10, std::vector<std::chrono::milliseconds>());
}

TEST(Invocation, Invokers2Distribution)
{
    TestInvocation(2, 10, std::vector<std::chrono::milliseconds>());
}

TEST(Invocation, Invokers3Distribution)
{
    TestInvocation(3, 10, std::vector<std::chrono::milliseconds>());
}

TEST(Invocation, Invokers4Distribution)
{
    TestInvocation(4, 10, std::vector<std::chrono::milliseconds>());
}

TEST(Invocation, InvokersBarrierSynchronization)
{
    std::vector<std::chrono::milliseconds> sleepTimes = 
    {
        // Will be used by the 4 invokers to sleep for on iteration 1
        std::chrono::milliseconds(1),
        std::chrono::milliseconds(2),
        std::chrono::milliseconds(3),
        std::chrono::milliseconds(4),

        // Will be used by the 4 invokers to sleep for on iteration 2
        std::chrono::milliseconds(1),
        std::chrono::milliseconds(2),
        std::chrono::milliseconds(3),
        std::chrono::milliseconds(4),

        // And so on
        std::chrono::milliseconds(1),
        std::chrono::milliseconds(1),
        std::chrono::milliseconds(1),
        std::chrono::milliseconds(1),

        std::chrono::milliseconds(1),
        std::chrono::milliseconds(2),
        std::chrono::milliseconds(3),
        std::chrono::milliseconds(4)
    };

    TestInvocation(4, 100, std::move(sleepTimes));
}
