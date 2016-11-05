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

#ifndef __MAPS_COMMON_CUH_
#define __MAPS_COMMON_CUH_

#include <cuda_runtime.h>
#include <iterator>
#include "common.h"
#include "macro_helpers.h"

namespace maps
{
    template <typename T>
    static __device__ __forceinline__ T LDG(T *ptr)
    {
#if (__CUDA_ARCH__ >= 320) // Kepler-based devices and above
        return __ldg(ptr);
#else
        return *ptr;
#endif
    }
    
    /// @brief Dynamic shared memory wrapper for general use.
    template<typename T>
    struct DynamicSharedMemory
    {
        ptrdiff_t m_offset;
        
        /**
         * @brief Initializes dynamic shared memory pointer with offset.
         * @param offset The offset (in bytes) where this buffer starts.
         */
        __device__ __forceinline__ void init(ptrdiff_t offset)
        {
            m_offset = offset;
        }

        __device__ __forceinline__ T *ptr()
        {
            extern __shared__ unsigned char __smem[];
            return (T *)(__smem + m_offset);
        }

        __device__ __forceinline__ const T *ptr() const
        {
            extern __shared__ unsigned char __smem[];
            return (const T *)(__smem + m_offset);
        }
    };

    /**
     * @brief A shared-memory array wrapper that can allocate both static 
     *        and dynamic shared memory. 
     * 
     * @note The struct must be designated "__shared__" on declaration,
     *       or part of a shared class.
     */
    template<typename T, size_t ARRAY_SIZE = DYNAMIC_SMEM>
    struct SharedMemoryArray
    {
        T smem[ARRAY_SIZE];

        __device__ __forceinline__ void init(ptrdiff_t offset = 0)
        {
            // Do nothing
        }
    };

    // Specialization for dynamic shared memory
    template<typename T>
    struct SharedMemoryArray<T, 0>
    {        
        T *smem;

        __device__ __forceinline__ void init(ptrdiff_t offset = 0)
        {
            extern __shared__ unsigned char __smem[];
            smem = (T *)(__smem + offset);
        }
    };

    namespace internal
    {
        static __device__ __forceinline__ bool __NextChunkAsync()
        {
            return false;
        }

        template <typename T>
        static __device__ __forceinline__ bool __NextChunkAsync(T& container)
        {
            container.nextChunkAsync();
            return T::SYNC_AFTER_NEXTCHUNK;
        }

        template <typename First, typename... Rest>
        static __device__ __forceinline__ bool __NextChunkAsync(First& first, Rest&... rest)
        {
            first.nextChunkAsync();
            return __NextChunkAsync(rest...) || First::SYNC_AFTER_NEXTCHUNK;
        }
    }  // namespace internal

    template <typename... Args>
    static __device__ __forceinline__ void NextChunkAll(Args&... args) {
        __syncthreads();
        bool bSync = internal::__NextChunkAsync(args...);
        if(bSync)
            __syncthreads();
    }    


    // Helper macros for MAPS_INIT
    #define _MAPS_DEF_INIT(INDEX, arg)                                   \
        __shared__ typename decltype(arg)::SharedData arg##_sdata;       \
        arg.init_async(arg##_sdata);                                     \

    #define _MAPS_DEF_COND(INDEX, arg) decltype(arg)::SYNC_AFTER_INIT || 
    #define _MAPS_DEF_POST(INDEX, arg) arg.init_async_postsync();


    /// @brief Initializes a list of containers simultaneously.
    #define MAPS_INIT(...)                                               \
        __MAPS_PP_FOR_EACH(_MAPS_DEF_INIT, __VA_ARGS__);                 \
        if(__MAPS_PP_FOR_EACH(_MAPS_DEF_COND, __VA_ARGS__) false)        \
            __syncthreads();                                             \
        __MAPS_PP_FOR_EACH(_MAPS_DEF_POST, __VA_ARGS__);

    // Internal macros for MAPS_FOREACH and MAPS_FOREACH_ALIGNED
    #define MAPS_FOREACH_NOUNROLL(iter, container) for(auto iter = container.begin(); iter.index() < decltype(container)::ELEMENTS; ++iter)
    #define MAPS_FOREACH_ALIGNED_NOUNROLL(input_iter, input_container, output_iter) for(auto input_iter = input_container.align(output_iter); input_iter.index() < decltype(input_container)::ELEMENTS; ++input_iter)

    // Define "MAPS_NOUNROLL" to disable automatic loop unrolling
    #ifndef MAPS_NOUNROLL
        #define MAPS_PRAGMA_UNROLL MAPS_PRAGMA(unroll)
    #else
        #define MAPS_PRAGMA_UNROLL
    #endif

    /**
     * @brief Loops over a container iterator.
     * @param [out] iter Resulting container iterator.
     * @param [in]  container Container to traverse.
     * @note Unrolls the loop by default. To disable, define "MAPS_NOUNROLL" prior to including MAPS.
     */
    #define MAPS_FOREACH(iter, container) MAPS_PRAGMA_UNROLL MAPS_FOREACH_NOUNROLL(iter, container)
    
    /**
    * @brief Loops over a container iterator, aligning the address according to another iterator.
    * @param [out] input_iter Resulting container iterator.
    * @param [in]  input_container Container to traverse.
    * @param [in]  output_iter Container to align to.
    * @note Unrolls the loop by default. To disable, define "MAPS_NOUNROLL" prior to including MAPS.
    */
    #define MAPS_FOREACH_ALIGNED(input_iter, input_container, output_iter) MAPS_PRAGMA_UNROLL MAPS_FOREACH_ALIGNED_NOUNROLL(input_iter, input_container, output_iter)


}  // namespace maps

#endif  // __MAPS_COMMON_CUH_
