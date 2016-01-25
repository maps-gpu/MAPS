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

#ifndef __MAPS_IO_COMMON_CUH_
#define __MAPS_IO_COMMON_CUH_

#include <type_traits>
#include <cuda_runtime.h>

#include "../../internal/common.cuh"
#include "../../internal/cuda_utils.hpp"
#include "../../internal/type_traits.hpp"

// NOTE: MAKE SURE THAT THERE ARE NO "blockIdx" REFERENCES IN THIS FILE.
//       IT IS OVERWRITTEN IN MAPS-MULTI FOR MULTI-GPU PURPOSES.

namespace maps
{
    
    ////////////////////////////////////////////////////////////////////////
    // Global to register reads
    
    struct IGlobalIOScheme
    {
        template <typename T>
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      T& value);
        template <typename T>
        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int stride, int offy, 
                                                      T& value);
        template <typename T>
        static __device__ __forceinline__ bool Read3D(const T *ptr, int offx,
                                                      int stride, int offy,
                                                      int height, int offz,
                                                      T& value);
        template <typename T>
        static __device__ __forceinline__ void Write(T *ptr, int offset,
                                                     const T& value);
    };
  

    ////////////////////////////////////////////////////////////////////////
    // Global to shared reads

    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int BLOCK_DEPTH, int XSHARED, int XSTRIDE, int YSHARED, 
              int ZSHARED, bool ASYNC, typename BoundaryConditions, 
              typename GlobalIOScheme>
    struct GlobalToShared
    {
        static __device__ __forceinline__ bool Read(const T *ptr, 
                                                    int dimensions[DIMS], 
                                                    int stride, int xoffset,
                                                    int yoffset, int zoffset, 
                                                    T *smem, int chunkID, 
                                                    int num_chunks);
    };

    ////////////////////////////////////////////////////////////////////////
    // Global to register array reads

    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT,
              int BLOCK_DEPTH, int DIMX, int DIMY, int DIMZ,
              typename BoundaryConditions, typename GlobalIOScheme>
    struct GlobalToArray
    {
        static __device__ __forceinline__ bool Read(const T *ptr, 
                                                    int dimensions[DIMS], 
                                                    int stride, int xoffset,
                                                    int yoffset, int zoffset, 
                                                    T (&regs)[DIMX*DIMY*DIMZ], 
                                                    int chunkID, 
                                                    int num_chunks);
    };

    //////////////////////////////////////////////////////////////////////////
    // Register array to global write + helper structure

    template <typename T, int DIMS, int BLOCK_WIDTH, int BLOCK_HEIGHT, 
              int BLOCK_DEPTH, int DIMX, int DIMY, int DIMZ, 
              typename GlobalIOScheme>
    struct ArrayToGlobal
    {
        static __device__ __forceinline__ bool Write(
          const T (&regs)[DIMX*DIMY*DIMZ], int dimensions[DIMS], int stride, 
          int xoffset, int yoffset, int zoffset, T *ptr);
    };

    //////////////////////////////////////////////////////////////////////////
    // Dimension ordering structures

    struct IDimensionOrdering
    {
        enum
        {
            DIMX = 0,
            DIMY = 1,
            DIMZ = 2
        };
            
        template<unsigned int DIM>
        static __host__ __device__ __forceinline__ unsigned int get(const dim3& blockId);
        
        template<unsigned int DIM>
        static __host__ __device__ __forceinline__ unsigned int get(const uint3& threadId);
    };

    template<int FIRST_DIM = 0, int SECOND_DIM = 1, int THIRD_DIM = 2>
    struct CustomOrdering : public IDimensionOrdering
    {
        enum
        {
            DIMX = FIRST_DIM,
            DIMY = SECOND_DIM,
            DIMZ = THIRD_DIM
        };

        template<typename T, unsigned int DIM>
        static __host__ __device__ __forceinline__ unsigned int get_internal(const T& a)
        {
            switch (DIM)
            {
              default:
              case 0:
                  return a.x;
              case 1:
                  return a.y;
              case 2:
                  return a.z;
            };
        }

        template<unsigned int DIM, typename T>
        static __host__ __device__ __forceinline__ unsigned int get(const T& a)
        {
            switch (DIM)
            {
              default:
              case 0:
                  return get_internal<T, FIRST_DIM>(a);
              case 1:
                  return get_internal<T, SECOND_DIM>(a);
              case 2:
                  return get_internal<T, THIRD_DIM>(a);
            }
        }
    };

    /// @brief Default dimension ordering (x,y,z)
    struct DefaultOrdering : public CustomOrdering<0,1,2>
    { };

    /// @brief Transposed dimension ordering (y,x,z)
    struct TransposedOrdering : public CustomOrdering<1,0,2>
    { };

}  // namespace maps

#endif  // __MAPS_IO_COMMON_CUH_



