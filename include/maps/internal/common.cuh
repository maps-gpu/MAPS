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

#ifndef __MAPS_COMMON_CUH_
#define __MAPS_COMMON_CUH_

#include <cuda_runtime.h>
#include <iterator>

namespace maps
{
	/**
	 * @brief Determines the behavior when accessing data beyond the 
	 *        dimensions of the input data.
	 */
	enum BorderBehavior
	{
		WB_NOCHECKS,	///< Assume input is allocated beyond the boundaries and do not perform the checks.
		WB_ZERO,		///< Return a constant value of T(0).
		WB_COPY,		///< Copy the closest value at the border.
		WB_WRAP,		///< Wrap the results around the input data.
	};
	
	// The use of enum ensures compile-time evaluation (pre-C++11 "constexpr").
	enum
	{
		/// @brief If given as template parameter, determines shared memory size at runtime
		DYNAMIC_SMEM = 0,
	};

	template <typename T>
	static __host__ __device__ inline T Clamp(const T& value, const T& minValue, const T& maxValue)
	{
		if(value < minValue)
			return minValue;
		if(value > maxValue)
			return maxValue;
		return value;
	}

	template<typename T>
	__device__ __forceinline__ T ldg(const T* ptr) 
	{
#if __CUDA_ARCH__ >= 350
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


	// Static assertions
#if (_MSC_VER >= 1600) || (__cplusplus >= 201103L)
	#define MAPS_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
	#ifdef __MAPS_CAT
	#error Using disallowed macro name __MAPS_CAT
	#endif
	#ifdef __MAPS_CAT_
	#error Using disallowed macro name __MAPS_CAT_
	#endif

	// Workaround for static assertions pre-C++11
	#define __MAPS_CAT_(a, b) a ## b
	#define __MAPS_CAT(a, b) __MAPS_CAT_(a, b)
	#define MAPS_STATIC_ASSERT(cond, msg) typedef int __MAPS_CAT(maps_static_assert, __LINE__)[(cond) ? 1 : -1]
#endif


}  // namespace maps

#endif  // __MAPS_COMMON_CUH_
