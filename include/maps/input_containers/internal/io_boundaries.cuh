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

#ifndef __MAPS_IO_BOUNDARIES_CUH_
#define __MAPS_IO_BOUNDARIES_CUH_

#include "io_common.cuh"

namespace maps
{
    struct NoBoundaries : public IBoundaryConditions
    {
        virtual __host__ int64_t ComputeIndex(int64_t index, size_t dimsize) const override
        {
            return index;
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value)
        {
            return GlobalIOScheme::Read1D<T>(ptr, offset, value);
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride, 
                                                      int offy, int height, 
                                                      T& value)
        {
            return GlobalIOScheme::Read2D<T>(ptr, offx, stride, offy, value);
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read3D(const T *ptr, int offx,
                                                      int width, int stride, 
                                                      int offy, int height, 
                                                      int offz, int depth, 
                                                      T& value)
        {
            return GlobalIOScheme::Read3D<T>(ptr, offx, stride, offy, height, 
                                             offz, value);
        }
    };

    struct ZeroBoundaries : public IBoundaryConditions
    {
        virtual __host__ int64_t ComputeIndex(int64_t index, size_t dimsize) const override
        {
            return index;
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value)
        {
            if (offset < 0 || offset >= width)
            {
                value = T(0);
                return true;
            }

            return GlobalIOScheme::Read1D<T>(ptr, offset, value);
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride, 
                                                      int offy, int height, 
                                                      T& value)
        {
            if (offx < 0 || offy < 0 || offx >= width || offy >= height)
            {
                value = T(0);
                return true;
            }
            return GlobalIOScheme::Read2D<T>(ptr, offx, stride, offy, value);
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read3D(const T *ptr, int offx,
                                                      int width, int stride,
                                                      int offy, int height,
                                                      int offz, int depth,
                                                      T& value)
        {
            if (offx < 0 || offy < 0 || offz < 0 ||  
                offx >= width || offy >= height || offz >= depth)
            {
                value = T(0);
                return true;
            }
            return GlobalIOScheme::Read3D<T>(ptr, offx, stride, offy, height, 
                                             offz, value);
        }
    };

    struct ClampBoundaries : public IBoundaryConditions
    {
        virtual __host__ int64_t ComputeIndex(int64_t index, size_t dimsize) const override
        {
            return Clamp(index, (int64_t)0, (int64_t)(dimsize - 1));
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value)
        {
            return GlobalIOScheme::Read1D<T>(ptr,
                                             Clamp(offset, 0, width - 1), value);
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride, 
                                                      int offy, int height, 
                                                      T& value)
        {
            return GlobalIOScheme::Read2D<T>(ptr,               
                                             Clamp(offx, 0, width - 1),
                                             stride, Clamp(offy, 0, height - 1),
                                             value);
        }
        
        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read3D(const T *ptr, int offx,
                                                      int width, int stride,
                                                      int offy, int height,
                                                      int offz, int depth,
                                                      T& value)
        {
            return GlobalIOScheme::Read3D<T>(ptr,
                                             Clamp(offx, 0, width - 1),
                                             stride, Clamp(offy, 0, height - 1),
                                             height, Clamp(offz, 0, depth - 1),
                                             value);
        }
    };

    struct WrapBoundaries : public IBoundaryConditions
    {
        virtual __host__ int64_t ComputeIndex(int64_t index, size_t dimsize) const override
        {
            if (dimsize == 0)
                return 0;
            while (index < 0)
                index += (int64_t)dimsize;
            while (index >= (int64_t)dimsize)
                index -= (int64_t)dimsize;
            return index;
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read1D(const T *ptr, int offset,
                                                      int width, T& value)
        {
            return GlobalIOScheme::Read1D<T>(ptr, Wrap(offset, width), value);
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx, 
                                                      int width, int stride,
                                                      int offy, int height, 
                                                      T& value)
        {
            return GlobalIOScheme::Read2D<T>(ptr,
                                             Wrap(offx, width), stride,
                                             Wrap(offy, height), value);
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read3D(const T *ptr, int offx,
                                                      int width, int stride,
                                                      int offy, int height,
                                                      int offz, int depth,
                                                      T& value)
        {
            return GlobalIOScheme::Read3D<T>(ptr,
                                             Wrap(offx, 0, width - 1),
                                             stride, Wrap(offy, 0, height - 1),
                                             height, Wrap(offz, 0, depth - 1),
                                             value);
        }
    };

}  // namespace maps

#endif  // __MAPS_IO_BOUNDARIES_CUH_
