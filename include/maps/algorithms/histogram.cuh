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

#ifndef __MAPS_HISTOGRAM_CUH_
#define __MAPS_HISTOGRAM_CUH_

#include "../internal/common.cuh"
#include "../internal/type_traits.hpp"

namespace maps
{
	/**
	 * @brief Computes device-level histograms using shared-memory atomic operations.
	 */
	template<typename BinType, unsigned int BINS>
	class HistogramSharedAtomic
	{
		MAPS_STATIC_ASSERT(IsIntegral<BinType>::value == true, "Bin type must be integral");
	protected:
		BinType *m_outHist;		///< Output histogram (pointer to global memory).
		BinType m_sHist[BINS];	///< Shared temporary histogram.
	public:
		__device__ __forceinline__ HistogramSharedAtomic()
		{
		}
	
		/**
		 * @brief Initializes the device-level histogram algorithm.
		 * @param[in] outHist The global-memory histogram pointer to output to.
		 */
		__device__ __forceinline__ void init(BinType *outHist)
		{
			int tid = threadIdx.x+threadIdx.y*blockDim.x;

			m_outHist = outHist;

			for (; tid < BINS; tid += blockDim.x*blockDim.y)
			{
				m_sHist[tid] = 0;
			}

			__syncthreads();
		}

		/**
		 * @brief Increments histogram at the given index ( range: [0,BINS) ).
		 * @param[in] index The index to increment.
		 */
		__device__ __forceinline__ void compute(unsigned int index)
		{		
			atomicAdd(m_sHist + index, 1);
		}

		/**
		 * @brief Commits the computed histogram to global memory.
		 */
		__device__ __forceinline__ void commit()
		{		
			int tid = threadIdx.x+threadIdx.y*blockDim.x;

			__syncthreads();

			for (; tid < BINS; tid += blockDim.x*blockDim.y)
			{
				atomicAdd(m_outHist + tid, m_sHist[tid]);
			}
		}
	};

	/**
	 * @brief Computes device-level histograms using global-memory atomic operations.
	 */
	template<typename BinType, unsigned int BINS>
	class HistogramGlobalAtomic
	{
		MAPS_STATIC_ASSERT(IsIntegral<BinType>::value == true, "Bin type must be integral");
	protected:
		BinType *m_outHist;	///< Output histogram (pointer to global memory).
	public:
		__device__ __forceinline__ HistogramGlobalAtomic()
		{
		}
	
		/**
		 * @brief Initializes the device-level histogram algorithm.
		 * @param[in] outHist The global-memory histogram pointer to output to.
		 */
		__device__ __forceinline__ void init(BinType *outHist)
		{
			m_outHist = outHist;
		}

		/**
		 * @brief Increments histogram at the given index ( range: [0,BINS) ).
		 * @param[in] index The index to increment.
		 */
		__device__ __forceinline__ void compute(unsigned int index)
		{		
			atomicAdd(m_outHist + index, 1);
		}

		/**
		 * @brief Commits the computed histogram to global memory.
		 */
		__device__ __forceinline__ void commit()
		{		
			// Nothing to do in this case
		}
	};

	/**
	 * @brief Automatically chooses the most efficient histogram algorithm
	 *		  based on the GPU architecture.
	 *
	 * @note To use in kernels, define: 
	 * @code
	 * __shared__ typename maps::Histogram<BinT, BINS>::DevHistogram hist;
	 */
	template<typename BinType, unsigned int BINS>
	struct Histogram
	{
	#if (__CUDA_ARCH__ >= 500)		// Maxwell-based devices
		typedef HistogramSharedAtomic<BinType, BINS> DevHistogram;
	#elif (__CUDA_ARCH__ >= 200)	// Fermi and Kepler-based devices
		typedef HistogramGlobalAtomic<BinType, BINS> DevHistogram;
	#elif (__CUDA_ARCH__ > 0)		// Older architectures
		// Histogram currently not implemented for these architectures
		MAPS_STATIC_ASSERT(false, "Histogram not implemented for architectures under SM 2.0");
	#else
		// Host
	#endif
	};

}  // namespace maps

#endif  // __MAPS_HISTOGRAM_CUH_
