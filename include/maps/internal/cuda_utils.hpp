/*
 *	MAPS: GPU device level memory abstraction library
 *	Based on the paper: "MAPS: Optimizing Massively Parallel Applications 
 *	Using Device-Level Memory Abstraction"
 *	Copyright (C) 2014  Amnon Barak, Eri Rubin, Tal Ben-Nun
 *
 *	This program is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *	
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *	
 *	You should have received a copy of the GNU General Public License
 *	along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __MAPS_CUDA_UTILS_HPP_
#define __MAPS_CUDA_UTILS_HPP_

#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>

namespace maps
{
	// CUDA assertions
	#define MAPS_CUDA_CHECK(err) do {													\
	cudaError_t errr = (err);															\
	if(errr != cudaSuccess)																\
	{																					\
		printf("ERROR in line %d: %s (%d)\n", __LINE__, cudaGetErrorString(errr), errr);\
		exit(1);																		\
	} 																					\
	} while(0)

	static inline int RoundUp(int gridSize, int blockSize) 
	{ 
		return (gridSize + blockSize - 1) / blockSize; 
	}

	static inline void CudaAlloc(void** d_ptr, unsigned int size)
	{
		MAPS_CUDA_CHECK(cudaMalloc(d_ptr, size));
	}

	static inline void CudaAllocAndClear(void** d_ptr, unsigned int size)
	{
		MAPS_CUDA_CHECK(cudaMalloc(d_ptr, size));
		MAPS_CUDA_CHECK(cudaMemset(*d_ptr, 0, size));
	}

	static inline void CudaAllocAndCopy(void** d_ptr, void* h_ptr, unsigned int size)
	{
		CudaAlloc(d_ptr,size);
		MAPS_CUDA_CHECK(cudaMemcpy(*d_ptr,h_ptr,size, cudaMemcpyHostToDevice ));
	}

	static inline void CudaSafeFree(void* d_ptr)
	{
		if (d_ptr)
			MAPS_CUDA_CHECK(cudaFree(d_ptr));
	}

}  // namespace maps

#endif  // __MAPS_CUDA_UTILS_HPP_
