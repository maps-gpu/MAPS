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

#define MAX_NODE_RANK_SIZE 20


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
		GraphMapper(unsigned int blockSize){_nodeCount=0;_edgeCount=0;_blockSize = blockSize;_d_p_c_count_map=0;_d_p_c_ind_map=0;_d_b_c_data_map=0;_d_b_c_ind_map=0;_d_p_c_s_ind_map=0;_MaxNumOfConstVecsInBlock=0;};
		~GraphMapper(){if (_d_p_c_ind_map) releaseIndexMap();};

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
			_edgeCount++;
			int edgeInd = _edgeCount;
			GraphMapper::Edge tmpEdge(this,node1Ind,node2Ind,edgeInd);
			_edgeList.push_back(tmpEdge);
			_nodeList[node1Ind].addEdge(edgeInd);
			// using the sign to know the dircetion of the edge
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
		void createIndexMap()
		{
			_nodeEdgeIndOffsetMap.resize(_nodeCount);

			unsigned int count=0;
			//build particle to constraint map

			//printf("graph has %d nodes\n",_nodeCount);

			// compose a list of edges for each node
			for (unsigned int nInd=0; nInd< _nodeCount; nInd++)
			{
				Node* pNode = &_nodeList[nInd];

				if (pNode->_exclude == true)
				{
					_nodeEdgeIndOffsetMap[nInd].x = 0;
					_nodeEdgeIndOffsetMap[nInd].y = count;
				}
				else
				{
					_nodeEdgeIndOffsetMap[nInd].x = pNode->_edgeIndList.size();
					_nodeEdgeIndOffsetMap[nInd].y = count;
					count += pNode->_edgeIndList.size();

					for (unsigned int eCount=0; eCount < pNode->_edgeIndList.size(); eCount++)
					{
						int ind = 0;
						if (pNode->_edgeIndList[eCount] > 0)
							ind = pNode->_edgeIndList[eCount]-1;
						else
							ind = pNode->_edgeIndList[eCount]+1;
						_nodeEdgeIndexList.push_back(ind);
					}
				}

			}

			CudaAllocAndCopy((void**)&_d_p_c_count_map,(void*)&(*_nodeEdgeIndOffsetMap.begin()),_nodeEdgeIndOffsetMap.size()*sizeof(uint2));
			CudaAllocAndCopy((void**)&_d_p_c_ind_map,(void*)&(*_nodeEdgeIndexList.begin()),_nodeEdgeIndexList.size()*sizeof(int));

			//for each block find the needed constraints 
			unsigned int numPartInBlock = _blockSize;
			unsigned int numBlocks = RoundUp(_nodeCount,numPartInBlock);
			unsigned int nInd=0;
			//unsigned int maxBlockSize=0;
			_maxBlockSize=0;
			unsigned int maxOverlap=0;
			unsigned int totalOverLap = 0;
			unsigned int overlapCount = 0;
			std::vector< std::unordered_set<unsigned int> > _blockEdgeIndSets;
			_blockEdgeIndSets.resize(numBlocks);

			//serialize block data
			_BlockSharedEdgeNumAndOffset.resize(numBlocks);

			unsigned int offset = 0;

			unsigned int maxRankOfNode = 0;
			unsigned int numPartRoundUp = RoundUp(_nodeCount,512)*512;
			_sharedNodeEdgeIndexBigList.resize(numPartRoundUp*MAX_NODE_RANK_SIZE);

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
								unsigned int eInd = abs(pNode->_edgeIndList[eCount]);
								if (curBlockSet->find(eInd) != curBlockSet->end())
									overlap++;
								else
									curBlockSet->insert(eInd);
							}
						}
					}
				}// end of block loop

				//printf("block %d has %d uniq elements %d overlap\n",blockId,curBlockSet->size(),overlap);
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
					_BlockEdgeIndexMap.push_back(*curBlockSetIt-1);
					curBlockSetIt++;
				}


				//prepare internal shared memory indexes for the constraints connected to a particle
				for (unsigned int nInd = blockStartPartId; nInd<(blockId+1)*numPartInBlock; nInd++)
				{
					if (nInd < _nodeCount)
					{
						Node* pNode = &_nodeList[nInd];
						if (pNode->_exclude != true)
						{
							unsigned int iCount=0;
							for (curBlockSetIt = curBlockSet->begin(); curBlockSetIt != curBlockSet->end(); curBlockSetIt++)
							{
								if (_edgeList[*curBlockSetIt-1]._iNode1Ind == nInd)
								{
									_sharedNodeEdgeIndexBigList[iCount*numPartRoundUp+nInd] = (std::distance(curBlockSet->begin(),curBlockSet->find(*curBlockSetIt))+1);
									iCount++;
								}

								if (_edgeList[*curBlockSetIt-1]._iNode2Ind == nInd) 
								{
									_sharedNodeEdgeIndexBigList[iCount*numPartRoundUp+nInd] = (-(std::distance(curBlockSet->begin(),curBlockSet->find(*curBlockSetIt)))-1);
									iCount++;
								}
							}

							maxRankOfNode = max(maxRankOfNode,iCount);
						}
					}
				}
			}



			//printf("num const %d num part %d, max Block %d max Overlap %d avg overlap %.2f\n",_edgeCount,_nodeCount,_maxBlockSize,maxOverlap,(float)totalOverLap/(float)overlapCount);

			CudaAllocAndCopy((void**)&_d_b_c_data_map,(void*)&(*_BlockSharedEdgeNumAndOffset.begin()),_BlockSharedEdgeNumAndOffset.size()*sizeof(uint2));
			CudaAllocAndCopy((void**)&_d_b_c_ind_map,(void*)&(*_BlockEdgeIndexMap.begin()),_BlockEdgeIndexMap.size()*sizeof(unsigned int));
			CudaAllocAndCopy((void**)&_d_p_c_s_ind_map,(void*)&(*_sharedNodeEdgeIndexBigList.begin()),maxRankOfNode*numPartRoundUp*sizeof(unsigned int));

			_MaxNumOfConstVecsInBlock = _maxBlockSize;

		}

		/**
		 * Cleanup function
		 */
		void releaseIndexMap()
		{
			CudaSafeFree(_d_p_c_count_map);
			CudaSafeFree(_d_p_c_ind_map);

			CudaSafeFree(_d_b_c_data_map);
			CudaSafeFree(_d_b_c_ind_map);
			CudaSafeFree(_d_p_c_s_ind_map);	
		}


	protected:
		std::vector<Node> _nodeList;
		std::vector<Edge> _edgeList;
		unsigned int _nodeCount;
		unsigned int _edgeCount;

		unsigned int _blockSize;

	public:
		std::vector<uint2> _nodeEdgeIndOffsetMap;
		std::vector<int> _nodeEdgeIndexList;

		std::vector<uint2> _BlockSharedEdgeNumAndOffset;
		std::vector<int> _BlockEdgeIndexMap;
		//std::vector<int> _sharedNodeEdgeIndexList[MAX_NODE_RANK_SIZE];
		std::vector<int> _sharedNodeEdgeIndexBigList;

		unsigned int _maxBlockSize;

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

		/// This member holds the maximum buffer needed in shared memory
		unsigned int _MaxNumOfConstVecsInBlock;

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
