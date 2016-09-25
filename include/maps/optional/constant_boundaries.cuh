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

#ifndef __MAPS_CONSTANT_BOUNDARIES_CUH_
#define __MAPS_CONSTANT_BOUNDARIES_CUH_

#include "../input_containers/internal/io_common.cuh"

// Allocate 256 bytes for constant boundary values. If more values are necessary, more can be allocated here.
#define __MAPS_CONSTANT_BOUNDARY_BYTES 256

namespace maps
{
    __constant__ char __MAPS_CONSTANT_BOUNDARIES_VALUES[__MAPS_CONSTANT_BOUNDARY_BYTES];

    // Defines a boundary condition where the value is taken from a constant number (determined at runtime)
    template <int CONSTANT_UID = 0>
    struct ConstantBoundaries : public IBoundaryConditions
    {
        // Set the constant value on all available GPU devices
        template <typename T>
        static __host__ void SetConstant(const T& value, cudaStream_t stream = nullptr, int gpuid = -1)
        {
            static_assert((CONSTANT_UID + 1) * sizeof(T) <= __MAPS_CONSTANT_BOUNDARY_BYTES,
                          "Too many constants, increase __MAPS_CONSTANT_BOUNDARY_BYTES");

            int ndevs = 0, lastdev = 0;
            MAPS_CUDA_CHECK(cudaGetDeviceCount(&ndevs));
            MAPS_CUDA_CHECK(cudaGetDevice(&lastdev));

            // Set on all devices
            if (gpuid < 0)
            {
                for (int i = 0; i < ndevs; ++i) {
                    MAPS_CUDA_CHECK(cudaSetDevice(i));
                    MAPS_CUDA_CHECK(cudaMemcpyToSymbolAsync(__MAPS_CONSTANT_BOUNDARIES_VALUES, &value, sizeof(T),
                                                            sizeof(T) * CONSTANT_UID, cudaMemcpyHostToDevice, stream));
                }
            }
            else // Set on specific device
            {
                MAPS_CUDA_CHECK(cudaSetDevice(gpuid));
                MAPS_CUDA_CHECK(cudaMemcpyToSymbolAsync(__MAPS_CONSTANT_BOUNDARIES_VALUES, &value, sizeof(T),
                                                        sizeof(T) * CONSTANT_UID, cudaMemcpyHostToDevice, stream));
            }
            MAPS_CUDA_CHECK(cudaSetDevice(lastdev));
        }



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
                value = ((T*)__MAPS_CONSTANT_BOUNDARIES_VALUES)[CONSTANT_UID];
                return true;
            }

            return GlobalIOScheme::template Read1D<T>(ptr, offset, value);
        }

        template <typename T, typename GlobalIOScheme>
        static __device__ __forceinline__ bool Read2D(const T *ptr, int offx,
                                                      int width, int stride,
                                                      int offy, int height,
                                                      T& value)
        {
            if (offx < 0 || offy < 0 || offx >= width || offy >= height)
            {
                value = ((T*)__MAPS_CONSTANT_BOUNDARIES_VALUES)[CONSTANT_UID];
                return true;
            }
            return GlobalIOScheme::template Read2D<T>(ptr, offx, stride, offy, value);
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
                value = ((T*)__MAPS_CONSTANT_BOUNDARIES_VALUES)[CONSTANT_UID];
                return true;
            }
            return GlobalIOScheme::template Read3D<T>(ptr, offx, stride, offy, height,
                                                      offz, value);
        }
    };
}  // namespace maps

#endif  // __MAPS_CONSTANT_BOUNDARIES_CUH_
