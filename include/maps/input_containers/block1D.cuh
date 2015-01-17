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

#ifndef __MAPS_BLOCK1D_CUH_
#define __MAPS_BLOCK1D_CUH_

#include "../internal/common.cuh"

namespace maps
{
	#ifdef __MAPS_WRAP
	#error Using disallowed macro name __MAPS_WRAP
	#endif

	// Mod without divide, works on values from 0 up to 2m
	#define __MAPS_WRAP(x,m) (((x)<(m))?(x):((x)-(m)))  

	/**
	 * @brief The Block1D dwarf container
	 *
	 * @note The total number of elements in the array must be
	 *       divisible by the block size.
	 */
	template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT>
	class Block1D
	{
		MAPS_STATIC_ASSERT(BLOCK_WIDTH > 0, "Block width must be positive");
		MAPS_STATIC_ASSERT(BLOCK_HEIGHT > 0, "Block height must be positive");

	protected:
		/// The data loaded onto shared memory
		T m_sdata[BLOCK_WIDTH*BLOCK_HEIGHT];

		/// The original data
		const T *m_gdata;
		
		/// The chunk that will be loaded next (offsetted per block)
		size_t m_nextChunk;

		/// The total number of chunks to load
		size_t m_numChunks;

	public:

		/// @brief Block 1D iterator class
		class iterator : public std::iterator<std::input_iterator_tag, T>
		{
		protected:
			unsigned int m_pos;
			const T *m_sParentData;
			
			__device__  __forceinline__ void next() 
			{
				++m_pos;
			}
		public:
			__device__ iterator(unsigned int pos, const Block1D *parent)
			{
				m_pos = pos;
				m_sParentData = parent->m_sdata;
			}

			__device__ iterator(const iterator& other)
			{
				m_pos = other.m_pos;
				m_sParentData = other.m_sParentData;
			}

			__device__  __forceinline__ void operator=(const iterator &a)
			{
				m_pos = a.m_pos;
				m_sParentData = a.m_sParentData;
			}

			__device__ __forceinline__ const T& operator*() const
			{
				return m_sParentData[m_pos];
			}

			__device__  __forceinline__ iterator& operator++() // Prefix
			{
				next();
				return *this;
			}

			__device__  __forceinline__ iterator operator++(int) // Postfix
			{
				iterator temp(*this);
				next();
				return temp;
			}
			
			__device__  __forceinline__ bool operator==(const iterator &a) const
			{
				return m_pos == a.m_pos;
			}
			__device__  __forceinline__ bool operator!=(const iterator &a) const
			{
				return m_pos != a.m_pos;
			}
		};

		/**
		 * Initializes the container.
		 */
		__device__ __forceinline__ void init(const T* data, size_t dataSize)
		{
			size_t numChunks = dataSize / (BLOCK_WIDTH*BLOCK_HEIGHT);

			// If threadIdx == (0,0). This works because thread indices are nonnegative.
			if ((threadIdx.x + threadIdx.y) == 0)
			{
				// Load parameters to shared object
				m_gdata = data;
			
				m_nextChunk = 0;
				m_numChunks = numChunks;
			}

			__syncthreads();
		}

		/**
		 * Creates a thread-level iterator that points to the beginning of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator begin() const
		{
			return iterator(0, this);
		}

		/**
		 * Creates a thread-level iterator that points to the end of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator end() const
		{
			return iterator(BLOCK_WIDTH*BLOCK_HEIGHT, this);
		}
		
		/**
		 * Progresses to process the next chunk.
		 */
		__device__ __forceinline__ void nextChunk() 
		{
			int tid = threadIdx.x + BLOCK_WIDTH*threadIdx.y;			
			
			// Signal compiler to not optimize this out
			volatile size_t nextChunk = m_nextChunk;

			__syncthreads();

			if(nextChunk < m_numChunks)
			{
				// Load next chunk to shared memory
				m_sdata[tid] = m_gdata[__MAPS_WRAP(blockIdx.x + nextChunk, m_numChunks) * BLOCK_WIDTH*BLOCK_HEIGHT + tid];
			}			

			if (tid == 0)
				m_nextChunk++;

			__syncthreads();
		}

		/**
		 * Progresses to process the next chunk without calling __syncthreads().
		 * @note This is an advanced function that should be used carefully.
		 */
		__device__ __forceinline__ void nextChunkAsync() 
		{
			int tid = threadIdx.x + BLOCK_WIDTH*threadIdx.y;			
			
			// Signal compiler to not optimize this out
			volatile size_t nextChunk = m_nextChunk;

			if(nextChunk < m_numChunks)
			{
				// Load next chunk to shared memory
				m_sdata[tid] = m_gdata[__MAPS_WRAP(blockIdx.x + nextChunk, m_numChunks) * BLOCK_WIDTH*BLOCK_HEIGHT + tid];
			}			

			if (tid == 0)
				m_nextChunk++;
		}

		/**
		 * @brief Returns false if there are more chunks to process.
		 */
		__device__ __forceinline__ bool isDone() 
		{ 
			return (m_nextChunk > m_numChunks);
		}
	};

	#undef __MAPS_WRAP

}  // namespace maps

#endif  // __MAPS_BLOCK1D_CUH_
