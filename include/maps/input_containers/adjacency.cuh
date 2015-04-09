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

#ifndef __MAPS_ADJACENCY_CUH_
#define __MAPS_ADJACENCY_CUH_

#include <maps/index_mappers/graph_mapper.hpp>

namespace maps
{

	template <typename T, bool SYMMETRIC = true>
	class Adjacency
	{
	public:
		float *_sdata;

		__device__ Adjacency() { }

		__device__ void init(int localThreadInd, int numThreadsInBlock,  const T *g_graphData, float *sdata,
			const GraphMapper::gpuData GraphGPUData, /*const uint2* g_constVecsBlockData, const unsigned int* g_data_block_ind,*/
			const unsigned int MaxNumOfItemsInBlock, const int &globalIndex,/*const uint2* g_num_const, const int* g_const_ind,*/ const unsigned int &numPartRoundUp)
		{
			_sdata = sdata;
			loadGraphDataToSharedMem(localThreadInd, numThreadsInBlock, _sdata, g_graphData, GraphGPUData._d_b_c_data_map, GraphGPUData._d_b_c_ind_map, MaxNumOfItemsInBlock);
			_globalIndex = globalIndex;
			_numOfConst = GraphGPUData._d_p_c_count_map[globalIndex].x;
			_g_const_ind = GraphGPUData._d_p_c_s_ind_map;
			_numPartRoundUp = numPartRoundUp;

			_MaxNumOfItemsInBlock = MaxNumOfItemsInBlock;		
		}

		// TODO(later): Implement chunk-based computations
		//__device__ void initChunky(const int &localThreadInd, const int &numThreadsInBlock, const T *g_corr_half_vec , float *sdata,const uint2* g_constVecsBlockData, const unsigned int* g_half_vec_block_ind, const unsigned int chunkSize)
		//{
		//	m_localThreadInd = localThreadInd;
		//	m_numThreadsInBlock = numThreadsInBlock;
		//	m_g_corr_half_vec = g_corr_half_vec;
		//	m_sdata = sdata;
		//	m_g_constVecsBlockData = g_constVecsBlockData;
		//	m_g_half_vec_block_ind = g_half_vec_block_ind;
		//	m_chunkSize = chunkSize;
		//	m_chunkNum = 0;

		//	loadGraphDataChunkToSharedMem(m_localThreadInd, m_numThreadsInBlock, m_sdata, m_g_corr_half_vec ,m_g_constVecsBlockData, m_g_half_vec_block_ind, m_chunkNum, m_chunkSize);

		//	m_chunkNum ++;
		//}

		//__device__ void loadNextChunk()
		//{
		//	loadGraphDataChunkToSharedMem(m_localThreadInd, m_numThreadsInBlock, m_sdata, m_g_corr_half_vec ,m_g_constVecsBlockData, m_g_half_vec_block_ind, m_chunkNum, m_chunkSize);

		//	m_chunkNum ++;
		//}

		__device__ __forceinline__ void loadGraphDataToSharedMem(int localThreadInd, int numThreadsInBlock, float *sdata, const T *g_graph_data ,const uint2* g_constIndexBlockData, const unsigned int* g_graph_data_block_ind, const unsigned int MaxNumOfItemsInBlock)
		{
			// first load all needed data to shared memory and then use them
			int loadIndex = localThreadInd;

			uint2 numOfItemsForBlock = g_constIndexBlockData[blockIdx.x];

			while (loadIndex < numOfItemsForBlock.x)
			{
				int half_vec_ind = ldg<unsigned int>(g_graph_data_block_ind + numOfItemsForBlock.y + loadIndex);
				T tmpVec = ldg<T>(g_graph_data + half_vec_ind);

				sdata[loadIndex] = tmpVec;
				//sdata[vecLoad]							= tmpVec.x;
				//sdata[vecLoad+MaxNumOfConstVecsInBlock]	= tmpVec.y;
				//sdata[vecLoad+2*MaxNumOfConstVecsInBlock]	= tmpVec.z;

				loadIndex += numThreadsInBlock;
			}

			__syncthreads();
		}

		// TODO(later): Implement chunk-based computations
		//__device__ void loadGraphDataChunkToSharedMem(int localThreadInd, int numThreadsInBlock, float *sdata, const T *g_corr_half_vec ,const uint2* g_constVecsBlockData, const unsigned int* g_half_vec_block_ind, const unsigned int chunkNum, const unsigned int chunkSize)
		//{
		//	// first load all needed data to shared memory and then use them
		//	int vecLoad = localThreadInd;
		//	const int chunkOffset = chunkNum*chunkSize;

		//	uint2 constVecsBlockData = g_constVecsBlockData[blockIdx.x];

		//	while (vecLoad + chunkOffset < constVecsBlockData.x && vecLoad < chunkSize)
		//	{
		//		int half_vec_ind = g_half_vec_block_ind[constVecsBlockData.y+vecLoad];
		//		T tmpVec = g_corr_half_vec[half_vec_ind];


		//		sdata[vecLoad]							= tmpVec.x;
		//		sdata[vecLoad+chunkSize]	= tmpVec.y;
		//		sdata[vecLoad+2*chunkSize]	= tmpVec.z;

		//		vecLoad += numThreadsInBlock;
		//	}

		//	__syncthreads();
		//}

		class iterator
		{
		public:
			__device__ iterator(){};

			__device__ __forceinline__ iterator(const Adjacency *pAdjacency, const unsigned int counter)//, const uint2* g_num_const, const int* g_const_ind, const unsigned int &MaxNumOfConstVecsInBlock, const unsigned int &numPartRoundUp)
			{
				_index = pAdjacency->_localThreadInd;
				_c_ind = pAdjacency->_globalIndex;
				_counter = counter;
				_sdata = pAdjacency->_sdata;
				_g_const_ind = pAdjacency->_g_const_ind;
				_sharedBlockSize = pAdjacency->_MaxNumOfItemsInBlock;
				_numPartRoundUp = pAdjacency->_numPartRoundUp;
			}

			__device__ __forceinline__ void loadData()
			{
				int c_r_ind = ldg<int>(_g_const_ind+_c_ind);

				if (SYMMETRIC)
				{
					unsigned int abs_c_r_ind = abs(c_r_ind)-1;
					float sign = ((c_r_ind>0) - 0.5f)*2.0f;

					_corr_half_vec = _sdata[abs_c_r_ind];
					//_corr_half_vec.x = _sdata[abs_c_r_ind];
					//_corr_half_vec.y = _sdata[abs_c_r_ind+_sharedBlockSize];
					//_corr_half_vec.z = _sdata[abs_c_r_ind+2*_sharedBlockSize];

					_corr_half_vec *= sign;
				}
				else
					_corr_half_vec = _sdata[c_r_ind];

			}

			__device__ __forceinline__ void getNext()
			{
				_c_ind+=_numPartRoundUp;
				_counter++;
			}

			__device__ __forceinline__ T operator* ()
			{
				loadData();
				return _corr_half_vec;
			}

			__device__ __forceinline__ iterator operator++() // Prefix
			{
				getNext();
				return *this;
			}

			__device__ __forceinline__ iterator operator++(int) // Postfix
			{
				iterator temp(*this);
				getNext();
				return temp;
			}

			__device__ __forceinline__ void operator=(const iterator &a)
			{
				_corr_half_vec = a._corr_half_vec;
				_index = a._index;
				_c_ind = a._c_ind;
				_counter = a._counter;
				_sdata = a._sdata;
				_g_const_ind = a._g_const_ind;
				_MaxNumOfConstVecsInBlock = a._MaxNumOfConstVecsInBlock;
				_numPartRoundUp = a._numPartRoundUp;
				_sharedBlockSize = a._sharedBlockSize;
			}

			__device__ __forceinline__ bool operator==(const iterator &a)
			{
				return _counter == a._counter;
			}

			__device__ __forceinline__ bool operator!=(const iterator &a)
			{
				return _counter != a._counter;
			}

			T _corr_half_vec;
			unsigned int _c_ind;
		protected:
			unsigned int _index;
			unsigned int _sharedBlockSize;
			unsigned int _numPartRoundUp;
			unsigned int _counter;
			const int* _g_const_ind;
			unsigned int _MaxNumOfConstVecsInBlock;

			float *_sdata;
		}; // end of class iterator

		/**
		 * Creates a thread-level iterator that points to the beginning of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator begin() const
		{
			iterator itBegin(this, 0);
			//itBegin.loadData();
			return itBegin;
		}

		/**
		 * Creates a thread-level iterator that points to the end of the current chunk.
		 * @return Thread-level iterator.
		 */
		__device__ __forceinline__ iterator end() const
		{
			return iterator(this, _numOfConst);
		}

	protected:
		int _localThreadInd;
		int _numThreadsInBlock;
		const T *_g_corr_half_vec;
		const uint2* _g_constVecsBlockData;
		const unsigned int* _g_half_vec_block_ind;
		unsigned int _chunkSize;
		unsigned int _chunkNum;

		// iterator data
		unsigned int _globalIndex;
		unsigned int _numOfConst;
		unsigned int _numPartRoundUp;
		const int* _g_const_ind;
		unsigned int _MaxNumOfItemsInBlock;
	};

}  // namespace maps

#endif  // __MAPS_ADJACENCY_CUH_
