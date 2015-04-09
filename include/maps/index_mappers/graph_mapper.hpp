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

#ifndef __MAPS_GRAPH_MAPPER_H_
#define __MAPS_GRAPH_MAPPER_H_

// stl includes
#include <vector>
#include <unordered_set>

//cuda includes
#include <vector_types.h>

#include "../internal/cuda_utils.hpp"

//#define MAX_NODE_RANK_SIZE 20


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

namespace maps
{
	/**
	 * @brief The GraphMapper host side pre processor 
	 *
	 * @note This class lets the user define the topology of the graph
	 *       It then analyzes the topology and produces index arrayes 
	 *       for efficient processing on the GPU
	 */
	class GraphMapper
	{
	protected:
		class Node;
		class Edge;
	public:
		GraphMapper(unsigned int blockSize, bool symmetric = true)
		{
			_nodeCount=0;_edgeCount=0;_blockSize = blockSize;_gpuData._d_p_c_count_map=0;
			_gpuData._d_p_c_ind_map=0;_gpuData._d_b_c_data_map=0;_gpuData._d_b_c_ind_map=0;
			_gpuData._d_p_c_s_ind_map=0;_MaxNumOfConstVecsInBlock=0;
			_symmetric = symmetric;
		};

		~GraphMapper(){if (_gpuData._d_p_c_ind_map) releaseIndexMap();};

		int addNode() 
		{
			GraphMapper::Node tmpNode(this,_nodeCount);
			_nodeList.push_back(tmpNode);
			_nodeCount++;
			return _nodeCount-1;
		}

		void addNodes(const unsigned int numOfNodes)
		{
			for (unsigned int nodeNum = 0; nodeNum < numOfNodes; nodeNum++)
				addNode();
		}

		/**
		 * Initializes the graph with a fixed number of nodes
		 *
		 */
		bool init(size_t numNodes)
		{
			// NOTE: add check and cleanup if needed
			addNodes(numNodes);
			return true;
		}

		void setMaxNodeRankSize(size_t maxNodeRankSize) 
		{
			_maxNodeRankSize = maxNodeRankSize+1;
		}

		/**
		 * adds the option of excluding specific nodes from being
		 * processed on the device
		 */
		void excludeNode(const unsigned int nodeInd)
		{
			_nodeList[nodeInd]._exclude = true;
		}

		/**
		 * This function connects to nodes in the graph with an edge
		 * Calling this function in the user code were he builds his
		 * data can be an easy way to build the topology
		 */
		int addEdge(unsigned int node1Ind,unsigned int node2Ind)
		{
			// We increment edgecount in the symmetric case before we get the index
			// to support negative edges (e.g. 1,-1)
			if (_symmetric)
				_edgeCount++;

			int edgeInd = _edgeCount;

			if (!_symmetric)
				_edgeCount++;

			GraphMapper::Edge tmpEdge(this,node1Ind,node2Ind,edgeInd);
			_edgeList.push_back(tmpEdge);
			_nodeList[node1Ind].addEdge(edgeInd);
			// using the sign to know the direction of the edge
			if (_symmetric)
				_nodeList[node2Ind].addEdge(-edgeInd);
			return edgeInd;
		}

		/**
		 * The main function which processes the topology of the 
		 * graph and builds the index maps.
		 * 
		 * @note  this function already copies the index maps
		 * to the GPU
		 */
		bool createIndexMap()
		{
			_nodeEdgeIndOffsetMap.resize(_nodeCount);

			//printf("Node count: %d\n", _nodeCount);

			unsigned int count=0;
			//build node to edge map

			//printf("graph has %d nodes\n",_nodeCount);

			// compose a list of edges for each node
			for (unsigned int _nInd=0; _nInd< _nodeCount; _nInd++)
			{
				Node* pNode = &_nodeList[_nInd];

				if (pNode->_exclude == true)
				{
					_nodeEdgeIndOffsetMap[_nInd].x = 0;
					_nodeEdgeIndOffsetMap[_nInd].y = count;
				}
				else
				{
					_nodeEdgeIndOffsetMap[_nInd].x = pNode->_edgeIndList.size();
					_nodeEdgeIndOffsetMap[_nInd].y = count;
					count += pNode->_edgeIndList.size();

					for (unsigned int eCount=0; eCount < pNode->_edgeIndList.size(); eCount++)
					{
						int ind = 0;
						if (_symmetric)
						{
							if (pNode->_edgeIndList[eCount] > 0)
								ind = pNode->_edgeIndList[eCount]-1;
							else
								ind = pNode->_edgeIndList[eCount]+1;
						}
						else
							ind = pNode->_edgeIndList[eCount];
						//_nodeEdgeIndexList.push_back(ind);
						_nodeEdgeIndexList.push_back(_edgeList[ind]._iNode2Ind);
					}
				}

			}

			
			//printf("Node edge index offset map: ");
			//for(auto iter : _nodeEdgeIndOffsetMap)
			//	printf("(%d, %d),  ", iter.x, iter.y);
			//printf("\n");

			//printf("Node edge index list: ");
			//for(auto iter : _nodeEdgeIndOffsetMap)
			//	printf("%d, ", iter);
			//printf("\n");
			

			CudaAllocAndCopy((void**)&_gpuData._d_p_c_count_map,(void*)&(*_nodeEdgeIndOffsetMap.begin()),_nodeEdgeIndOffsetMap.size()*sizeof(uint2));
			CudaAllocAndCopy((void**)&_gpuData._d_p_c_ind_map,(void*)&(*_nodeEdgeIndexList.begin()),_nodeEdgeIndexList.size()*sizeof(int));

			//for each block find the needed indexes 
			unsigned int numPartInBlock = _blockSize;
			unsigned int numBlocks = RoundUp(_nodeCount,numPartInBlock);
			unsigned int nInd=0;
			//unsigned int maxBlockSize=0;
			_maxBlockSize=0;
			unsigned int maxOverlap=0;
			unsigned int totalOverLap = 0;
			unsigned int overlapCount = 0;
			std::vector<std::unordered_set<unsigned int>> _blockEdgeIndSets;
			_blockEdgeIndSets.resize(numBlocks);

			//serialize block data
			_BlockSharedEdgeNumAndOffset.resize(numBlocks);

			unsigned int offset = 0;

			unsigned int maxRankOfNode = 0;
			unsigned int numPartRoundUp = RoundUp(_nodeCount,512)*512;
			_sharedNodeEdgeIndexBigList.resize(numPartRoundUp*_maxNodeRankSize);

			for (unsigned int blockId=0; blockId < numBlocks; blockId++)
			{
				std::unordered_set<unsigned int> *curBlockSet = &(_blockEdgeIndSets[blockId]);
				unsigned int overlap = 0;
				unsigned int blockStartPartId = nInd;
				for (;nInd<(blockId+1)*numPartInBlock;nInd++)
				{
					if (nInd < _nodeCount)
					{
						Node* pNode = &_nodeList[nInd];
						if (pNode->_exclude != true)
						{

							for (unsigned int eCount=0; eCount < pNode->_edgeIndList.size(); eCount++)
							{
								unsigned int nnInd;

								if (_symmetric)
									nnInd = abs(_edgeList[pNode->_edgeIndList[eCount]-1]._iNode2Ind);
								else
									nnInd = abs(_edgeList[pNode->_edgeIndList[eCount]]._iNode2Ind);

								if (curBlockSet->find(nnInd) != curBlockSet->end())
									overlap++;
								else
									curBlockSet->insert(nnInd);
							}
						}
					}
				}// end of block loop

				//TODO: SHOULD SORT THE INDEXES

				//printf("block %d has %d unique elements %d overlap\n",blockId,curBlockSet->size(),overlap);
				_maxBlockSize = max(curBlockSet->size(),_maxBlockSize);
				maxOverlap = max(overlap,maxOverlap);
				totalOverLap += overlap;
				overlapCount ++;

				_BlockSharedEdgeNumAndOffset[blockId].x = curBlockSet->size();
				_BlockSharedEdgeNumAndOffset[blockId].y = offset;
				//printf("\tcurBlockSet size %d\n\toffset %d\n",curBlockSet->size(),offset);

				offset += curBlockSet->size();

				std::unordered_set<unsigned int>::iterator curBlockSetIt = curBlockSet->begin();

				for (unsigned int blockSetInd=0; blockSetInd< curBlockSet->size(); blockSetInd++)
				{
					if (_symmetric)
						_BlockEdgeIndexMap.push_back(*curBlockSetIt-1);
					else
						_BlockEdgeIndexMap.push_back(*curBlockSetIt);
					curBlockSetIt++;
				}


				//prepare internal shared memory indexes for the constraints connected to a particle
				for (unsigned int nInd__ = blockStartPartId; nInd__<(blockId+1)*numPartInBlock; nInd__++)
				{
					if (nInd__ < _nodeCount)
					{
						Node* pNode = &_nodeList[nInd__];
						if (pNode->_exclude != true)
						{
							unsigned int iCount=0;
							if (_symmetric)
							{
								for (curBlockSetIt = curBlockSet->begin(); curBlockSetIt != curBlockSet->end(); curBlockSetIt++)
								{
										if (iCount >= _maxNodeRankSize)
										{
											if ((_edgeList[*curBlockSetIt-1]._iNode1Ind == nInd__) || (_edgeList[*curBlockSetIt-1]._iNode2Ind == nInd__) )
											{
												//printf("too many edges connected to node %d max %d cur count %d\n",nInd__, _maxNodeRankSize, iCount);
												iCount++;
											}
										}
										else
										{
											if (_edgeList[*curBlockSetIt-1]._iNode1Ind == nInd__)
											{
												_sharedNodeEdgeIndexBigList[iCount*numPartRoundUp+nInd__] = (std::distance(curBlockSet->begin(),curBlockSet->find(*curBlockSetIt))+1);
												iCount++;
											}

											if (_symmetric && _edgeList[*curBlockSetIt-1]._iNode2Ind == nInd__) 
											{
												_sharedNodeEdgeIndexBigList[iCount*numPartRoundUp+nInd__] = (-(std::distance(curBlockSet->begin(),curBlockSet->find(*curBlockSetIt)))-1);
												iCount++;
											}
										}
								}
									//else
									//{
									//	if (iCount >= _maxNodeRankSize)
									//	{
									//		//return false;
									//		if ((_edgeList[*curBlockSetIt]._iNode1Ind == nInd__))
									//		{
									//			//printf("too many edges connected to node %d max %d cur count %d\n",nInd__, _maxNodeRankSize, iCount);
									//			iCount++;
									//		}
									//	}
									//	else
									//	{
									//		//if (_edgeList[*curBlockSetIt]._iNode1Ind == nInd__)
									//		if(IsConnected(nInd__, *curBlockSetIt))										
									//		{
									//			_sharedNodeEdgeIndexBigList[iCount*numPartRoundUp+nInd__] = (std::distance(curBlockSet->begin(),curBlockSet->find(*curBlockSetIt)));
									//			iCount++;
									//		}
									//	}
									//}
							}
							else
							{
								Node &pCurNode = _nodeList[nInd__];
								for (iCount = 0; iCount < pCurNode._edgeIndList.size(); iCount++)
								{
									_sharedNodeEdgeIndexBigList[iCount*numPartRoundUp+nInd__] = (std::distance(curBlockSet->begin(),curBlockSet->find(_edgeList[pCurNode._edgeIndList[iCount]]._iNode2Ind)));
								}
							}

							maxRankOfNode = max(maxRankOfNode,iCount);
						}
					}
				}
			}

			//printf("maxRank is %d\n",maxRankOfNode);

			//printf("Block shared edge (num, offset): ");
			//for(auto iter : _BlockSharedEdgeNumAndOffset)
			//	printf("(%d, %d),  ", iter.x, iter.y);
			//printf("\n");

			//printf("Block edge index map: ");
			//for(auto iter : _BlockEdgeIndexMap)
			//	printf("%d, ", iter);
			//printf("\n");

			//printf("Shared node edge index big list: ");
			//for(auto iter : _sharedNodeEdgeIndexBigList)
			//	printf("%d, ", iter);
			//printf("\n");

			//printf("num const %d num part %d, max Block %d max Overlap %d avg overlap %.2f\n",_edgeCount,_nodeCount,_maxBlockSize,maxOverlap,(float)totalOverLap/(float)overlapCount);

			CudaAllocAndCopy((void**)&_gpuData._d_b_c_data_map,(void*)&(*_BlockSharedEdgeNumAndOffset.begin()),_BlockSharedEdgeNumAndOffset.size()*sizeof(uint2));
			CudaAllocAndCopy((void**)&_gpuData._d_b_c_ind_map,(void*)&(*_BlockEdgeIndexMap.begin()),_BlockEdgeIndexMap.size()*sizeof(unsigned int));
			CudaAllocAndCopy((void**)&_gpuData._d_p_c_s_ind_map,(void*)&(*_sharedNodeEdgeIndexBigList.begin()),maxRankOfNode*numPartRoundUp*sizeof(unsigned int));

			_MaxNumOfConstVecsInBlock = _maxBlockSize;

			return true;
		}

		/**
		 * Cleanup function
		 */
		void releaseIndexMap()
		{
			CudaSafeFree(_gpuData._d_p_c_count_map);
			CudaSafeFree(_gpuData._d_p_c_ind_map);

			CudaSafeFree(_gpuData._d_b_c_data_map);
			CudaSafeFree(_gpuData._d_b_c_ind_map);
			CudaSafeFree(_gpuData._d_p_c_s_ind_map);	
		}


	protected:
		std::vector<Node> _nodeList;
		std::vector<Edge> _edgeList;
		unsigned int _nodeCount;
		unsigned int _edgeCount;
		size_t _maxNodeRankSize;

		unsigned int _blockSize;

	public:
		std::vector<uint2> _nodeEdgeIndOffsetMap;
		std::vector<int> _nodeEdgeIndexList;

		std::vector<uint2> _BlockSharedEdgeNumAndOffset;
		std::vector<int> _BlockEdgeIndexMap;
		//std::vector<int> _sharedNodeEdgeIndexList[_maxNodeRankSize];
		std::vector<int> _sharedNodeEdgeIndexBigList;

		unsigned int _maxBlockSize;

		struct gpuData
		{
			/// internal device index buffer, needs to be passed to the 
			/// device container
			uint2* _d_p_c_count_map;
			/// internal device index buffer, needs to be passed to the 
			/// device container
			int* _d_p_c_ind_map;

			/// internal device index buffer, needs to be passed to the 
			/// device container
			uint2* _d_b_c_data_map;
			/// internal device index buffer, needs to be passed to the 
			/// device container
			unsigned int* _d_b_c_ind_map;
			/// internal device index buffer, needs to be passed to the 
			/// device container
			int* _d_p_c_s_ind_map;
		};

		gpuData _gpuData;
		/// This member holds the maximum buffer needed in shared memory
		unsigned int _MaxNumOfConstVecsInBlock;

		/// Determine if this is a symmetric graph
		bool _symmetric;

		/**
		 * The user can use this function to pass the whole topology
		 * of the graph in one go to the class
		 */
		bool registerData(const uint2 *edges, size_t numEdges)
		{
			for(size_t i = 0; i < numEdges; i++)
				if(!addEdge(edges[i].x, edges[i].y))
					return false;

			return true;
		}

	protected:
		class Edge
		{
		public:
			friend class GraphMapper;
			Edge(GraphMapper *pParentGraph, unsigned int node1Ind, unsigned int node2Ind, unsigned int index)
			{
				_pGraph = pParentGraph;
				_iNode1Ind = node1Ind;
				_iNode2Ind = node2Ind;
				_iInd = index;
			}
			~Edge(){};
		protected:
			GraphMapper *_pGraph;
			int _iNode1Ind;
			int _iNode2Ind;
			int _iInd;
		};

		class Node
		{
		public:
			friend class GraphMapper;
			Node(GraphMapper *pParentGraph, unsigned int index)
			{
				_pGraph = pParentGraph;
				_iInd = index;
				_exclude = false;
			}
			~Node(){};

		protected:
			void addEdge(const int edgeInd) 
			{
				_edgeIndList.push_back(edgeInd);
			}

			GraphMapper *_pGraph;
			std::vector<int> _edgeIndList;
			int _iInd;
			bool _exclude;
		};
	};
}  // namespace maps

#endif  // __MAPS_GRAPH_MAPPER_H_
