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

#ifndef __MAPS_BLOCK2D_CUH_
#define __MAPS_BLOCK2D_CUH_

#include "../internal/common.cuh"

namespace maps
{
	/**
	 * @brief Shared data container for the Block2D class
	 */
	template <bool TRANS, typename T, int BLOCK_SIZE>
	struct Block2DData
	{
		T data[(BLOCK_SIZE+(int)TRANS) * BLOCK_SIZE];
	};

	/**
	 * @brief The Block2D dwarf container
	 *
	 * @note The total number of elements in the array must be
	 *       divisible by the block size.
	 */
	template <bool TRANS, typename T, int BLOCK_SIZE>
	class Block2D
	{
	public:
		/// Total matrix width (or height in transposed mode)
		unsigned int width;

		/// Block processing step size
		unsigned int step;
		
		/// Block processing end index
		unsigned int endInd;

		/// The current chunk index
		unsigned int index;

		/// Pointer to the data loaded onto shared memory
		T *smem;
		
		/// The original data
		const T *pData;

		/**
		 * Initializes the container.
		 */
		__device__ __forceinline__ void init(const T *data, unsigned int _width, 
											 Block2DData<TRANS, T, BLOCK_SIZE>& sdata)
		{
			width = _width;

			pData = data;
			smem = sdata.data;

			index   = TRANS ? (BLOCK_SIZE * blockIdx.x) : (width * BLOCK_SIZE * blockIdx.y);
			step    = TRANS ? (BLOCK_SIZE * width)      : (BLOCK_SIZE);
			endInd  = index + width;
		}

		/**
		 * Progresses to process the next chunk.
		 */
		__device__  __forceinline__ void nextChunk()
		{
			__syncthreads();

			// load data to shared memory
			smem[(BLOCK_SIZE+(int)TRANS)*threadIdx.y+threadIdx.x] = pData[index + width * threadIdx.y + threadIdx.x];

			index += step;

			__syncthreads();
		}

		/**
		 * Progresses to process the next chunk without calling __syncthreads().
		 * @note This is an advanced function that should be used carefully.
		 */
		__device__ __forceinline__ void nextChunkAsync() 
		{
			// load data to shared memory
			smem[(BLOCK_SIZE+(int)TRANS)*threadIdx.y+threadIdx.x] = pData[index + width * threadIdx.y + threadIdx.x];

			index += step;
		}

		/**
		 * @brief Returns false if there are more chunks to process.
		 */
		__device__ __forceinline__ bool isDone()
		{
			return index >= endInd;
		}

		/**
		 * Returns an internal item from this chunk, without creating iterators.
		 * @note This is an advanced function that should be used carefully.
		 */
		__device__  __forceinline__ T getItem(const int ind)
		{
			if (TRANS)
			{
				return smem[ind*(BLOCK_SIZE+TRANS)+threadIdx.x];
			}
			else
			{
				return smem[threadIdx.y*BLOCK_SIZE+ind];
			}
		}

		/// @brief Block 2D iterator class
		class iterator : public std::iterator<std::input_iterator_tag, T>
		{
		public:
			T* _pAs;
			int _k;

		public:
			__device__ __forceinline__ iterator()
			{
				_k = -1;
			}

			__device__ __forceinline__ ~iterator() {}

			__device__ __forceinline__ iterator(const int k, Block2D *pBlock2D)
			{
				init(k,pBlock2D);
			}

			__device__ __forceinline__ void init(const int k, Block2D *pBlock2D)
			{
				_k = k;
				_pAs = pBlock2D->smem;
			}

			__device__ __forceinline__ int k() const
			{
				return _k;
			}

			__device__ __forceinline__ T operator* () const
			{
				if (TRANS)
					return _pAs[(BLOCK_SIZE+1)*_k+threadIdx.x];
				else
					return _pAs[(BLOCK_SIZE)*threadIdx.y+_k];
			}


			__device__ __forceinline__ iterator operator++() // Prefix
			{
				next();
				return *this;
			}

			__device__ __forceinline__ iterator operator++(int) // Postfix
			{
				iterator temp(*this);
				next();
				return temp;
			}

			__device__ __forceinline__ void next() 
			{
				_k++;
			}

			__device__ __forceinline__ void operator=(const iterator &a)
			{
				_k = a._k;
				_pAs = a._pAs;
			}
			__device__ __forceinline__ bool operator==(const iterator &a)
			{
				return (_k == a._k);
			}
			__device__ __forceinline__ bool operator!=(const iterator &a)
			{
				return !(_k == a._k);
			}
		};  // class iterator


		/**
		 * Creates a thread-level iterator that points to the beginning of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator begin()
		{
			return iterator(0,this);
		}

		/**
		 * Creates a thread-level iterator that points to the end of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator end()
		{
			return iterator(BLOCK_SIZE,this);
		}

	};

}  // namespace maps

#endif  // __MAPS_BLOCK2D_CUH_
