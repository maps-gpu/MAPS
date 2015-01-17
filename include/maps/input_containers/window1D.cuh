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

#ifndef __MAPS_WINDOW1D_CUH_
#define __MAPS_WINDOW1D_CUH_

#include "../internal/common.cuh"
#include "../internal/texref.cuh"

namespace maps
{
	#ifdef __MAPS_WIND_WIDTH
	#error Using disallowed macro name __MAPS_WIND_WIDTH
	#endif
	#ifdef __MAPS_XSHARED
	#error Using disallowed macro name __MAPS_XSHARED
	#endif

	#define __MAPS_WIND_WIDTH (WINDOW_APRON*2+1)
	#define __MAPS_XSHARED ((BLOCK_WIDTH > 0) ? (BLOCK_WIDTH+WINDOW_APRON*2) : (0))

	template<typename T, int BLOCK_WIDTH, int WINDOW_APRON, BorderBehavior BORDER = WB_NOCHECKS>
	class Window1D
	{
		MAPS_STATIC_ASSERT(BLOCK_WIDTH > 0, "Block width must be positive");
		MAPS_STATIC_ASSERT(WINDOW_APRON > 0, "Window apron must be positive");
		MAPS_STATIC_ASSERT(BLOCK_WIDTH >= 2 * WINDOW_APRON, "Block width must be at least twice the size of the apron");

	protected:
		/// The data loaded onto shared memory
		T m_sdata[__MAPS_XSHARED];
		
		__device__ __forceinline__ T at(const T *ptr, int x, size_t width)
		{
			// Boundary conditions
			if(BORDER != WB_NOCHECKS)
			{
				if(x < 0 || x >= width)
				{					
					if(BORDER == WB_COPY)
						return ptr[Clamp(x, 0, (int)width)];
					else if(BORDER == WB_WRAP)
						return ptr[(x % width)];
					else // if(BORDER == WB_ZERO) or anything else
						return T(0);
				}
			}
			return ptr[x];
		}
	public:

		/// @brief Internal Window 1D iterator class
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
			__device__ iterator(unsigned int pos, const Window1D *parent)
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
		 * @brief Initializes the container.
		 * @param[in] ptr Pointer to input data.
		 * @param[in] width Total width of input data.
		 */
		__device__ void init(const T *ptr, size_t width)
		{
			// Load data to shared memory

			// Calculate begin and end indices
			int x = BLOCK_WIDTH * blockIdx.x + threadIdx.x;

			// Copy data: [left_apron, blocksize)
			m_sdata[threadIdx.x] = at(ptr, x - WINDOW_APRON, width);
		
			// Copy right apron data [blocksize + 1, blocksize + 2*WINDOW_APRON)
			if (threadIdx.x < 2*WINDOW_APRON)
			{
				m_sdata[BLOCK_WIDTH + threadIdx.x] = at(ptr, x - WINDOW_APRON + BLOCK_WIDTH, width);
			}

			__syncthreads();
		}

		/**
		 * @brief Creates a thread-level iterator that points to the beginning of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator begin() const
		{
			return iterator(threadIdx.x, this);
		}

		/**
		 * @brief Creates a thread-level iterator that points to the end of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator end() const
		{
			return iterator(threadIdx.x + __MAPS_WIND_WIDTH, this);
		}
		
		/**
		 * @brief Progresses to process the next chunk (does nothing).
		 */
		__device__ __forceinline__ void nextChunk() { }

		/**
		 * Progresses to process the next chunk without calling __syncthreads() (does nothing).
		 * @note This is an advanced function that should be used carefully.
		 */
		__device__ __forceinline__ void nextChunkAsync() { }

		/**
		 * @brief Returns false if there are more chunks to process.
		 */
		__device__ __forceinline__ bool isDone() { return true; }
	};

	template<typename T, int TEXTURE_UID, int BLOCK_WIDTH, int WINDOW_APRON>
	class Window1DTexture
	{
		MAPS_STATIC_ASSERT(BLOCK_WIDTH > 0, "Block width must be positive");
		MAPS_STATIC_ASSERT(WINDOW_APRON > 0, "Window apron must be positive");
		MAPS_STATIC_ASSERT(BLOCK_WIDTH >= 2 * WINDOW_APRON, "Block width must be at least twice the size of the apron");

	protected:
		/// The data loaded onto shared memory
		T m_sdata[__MAPS_XSHARED];
		
		__device__ __forceinline__ T at(const T *ptr, int x)
		{
			typedef typename UniqueTexRef1D<T>::template TexId<TEXTURE_UID> TexId;
			return TexId::read(x);
		}
	public:

		/// @brief Internal Window 1D iterator class
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
			__device__ iterator(unsigned int pos, const Window1DTexture *parent)
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
		__device__ void init()
		{
			// Load data to shared memory

			// Calculate begin and end indices
			int x = BLOCK_WIDTH * blockIdx.x + threadIdx.x;

			// Copy data: [left_apron, blocksize)
			m_sdata[threadIdx.x] = at(NULL, x - WINDOW_APRON);
		
			// Copy right apron data [blocksize + 1, blocksize + 2*WINDOW_APRON)
			if (threadIdx.x < 2*WINDOW_APRON)
			{
				m_sdata[BLOCK_WIDTH + threadIdx.x] = at(NULL, x - WINDOW_APRON + BLOCK_WIDTH);
			}

			__syncthreads();			
		}

		/**
		 * Creates a thread-level iterator that points to the beginning of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator begin() const
		{
			return iterator(threadIdx.x, this);
		}

		/**
		 * Creates a thread-level iterator that points to the end of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator end() const
		{
			return iterator(threadIdx.x + __MAPS_WIND_WIDTH, this);
		}
		
		/**
		 * Progresses to process the next chunk (does nothing).
		 */
		__device__ __forceinline__ void nextChunk() { }

		/**
		 * Progresses to process the next chunk without calling __syncthreads() (does nothing).
		 * @note This is an advanced function that should be used carefully.
		 */
		__device__ __forceinline__ void nextChunkAsync() { }

		/**
		 * @brief Returns false if there are more chunks to process.
		 */
		__device__ __forceinline__ bool isDone() { return true; }
	};

	#undef __MAPS_WIND_WIDTH
	#undef __MAPS_XSHARED

}  // namespace maps

#endif  // __MAPS_WINDOW1D_CUH_
