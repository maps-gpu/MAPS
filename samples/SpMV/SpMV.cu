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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <maps/maps.cuh>
#include <maps/index_mappers/graph_mapper.hpp>

#include "cuda_common.h"


#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>

#include "mmio.h"

#define REPETITIONS 20


__global__ void SPmV_maps_kernel_naive (const int N, float* g_A_val, unsigned int* g_A_i_ind, unsigned int* g_A_lineStartInd, float* g_x, float* g_b)
{
	int global_ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (global_ind > N)
		return;

	int lineStartIndex = g_A_lineStartInd[global_ind];
	int nextLineStartInd = g_A_lineStartInd[global_ind+1];

	float res=0.f;

	for (int i=lineStartIndex; i<nextLineStartInd; i++)
	{
		res += g_A_val[i] * g_x[g_A_i_ind[i]];
	}

	g_b[global_ind] = res;
}

__global__ void SPmV_maps_kernel_maps(const int N, const float* g_A_val, const unsigned int* g_A_i_ind, const unsigned int* g_A_lineStartInd, const float* g_x, float* g_b,
									  const maps::GraphMapper::gpuData GraphGPUData, const unsigned int num_of_particles, const unsigned int MaxNumOfConstVecsInBlock, 
									  const unsigned int numPartRoundUp)
{
	int global_ind = threadIdx.x + blockIdx.x*blockDim.x;

	extern __shared__ float sdata[];

	maps::Adjacency<float,false> MyGraph;

	MyGraph.init(threadIdx.x, blockDim.x, g_x ,sdata , GraphGPUData, MaxNumOfConstVecsInBlock, global_ind,  numPartRoundUp);

	if (global_ind < N)
	{
		int lineStartIndex = g_A_lineStartInd[global_ind];
		int nextLineStartInd = g_A_lineStartInd[global_ind+1];
	
		float res=0.f;

		maps::Adjacency<float,false>::iterator gIter = MyGraph.begin();
		for (int i=lineStartIndex; i<nextLineStartInd; ++i,++gIter)
		{
			res += g_A_val[i] * (*gIter);
		}

		g_b[global_ind] = res;
	}
}

int main(int argc, char *argv[])
{
	MM_typecode matcode;
	FILE *f;
	int rows, cols, nz;   
	int i, *I, *J;
	double *val;
	bool symetric = false;

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
		exit(1);
	}
	else    
	{ 
		if ((f = fopen(argv[1], "r")) == NULL) 
			exit(1);
	}

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}


	/*  This is how one can screen matrix types if their application */
	/*  only supports a subset of the Matrix Market data types.      */

	if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
		mm_is_sparse(matcode) )
	{
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
	}

	symetric = mm_is_symmetric(matcode);

	/* find out size of sparse matrix .... */

	if ((mm_read_mtx_crd_size(f, &rows, &cols, &nz)) !=0)
		exit(1);

	printf("%s matrix has %d nz with dims of M (rows)=%d N (cols)=%d\n",argv[1],nz,rows,cols);

	/* reserve memory for matrices */

	I = (int *) malloc(nz * sizeof(int));
	J = (int *) malloc(nz * sizeof(int));
	val = (double *) malloc(nz * sizeof(double));

	/*
	 * Matrix market format: (from http://math.nist.gov/MatrixMarket/formats.html)
	 
		%%MatrixMarket matrix coordinate real general
		%=================================================================================
		%
		% This ASCII file represents a sparse MxN matrix with L 
		% nonzeros in the following Matrix Market format:
		%
		% +----------------------------------------------+
		% |%%MatrixMarket matrix coordinate real general | <--- header line
		% |%                                             | <--+
		% |% comments                                    |    |-- 0 or more comment lines
		% |%                                             | <--+         
		% |    M  N  L                                   | <--- rows, columns, entries
		% |    I1  J1  A(I1, J1)                         | <--+
		% |    I2  J2  A(I2, J2)                         |    |
		% |    I3  J3  A(I3, J3)                         |    |-- L lines
		% |        . . .                                 |    |
		% |    IL JL  A(IL, JL)                          | <--+
		% +----------------------------------------------+   
		%
		% Indices are 1-based, i.e. A(1,1) is the first element.
		%
		%=================================================================================
		5  5  8
		1     1   1.000e+00
		2     2   1.050e+01
		3     3   1.500e-02
		1     4   6.000e+00
		4     2   2.505e+02
		4     4  -2.800e+02
		4     5   3.332e+01
		5     5   1.200e+01

		This produces the following matrix:
		1    0      0       6      0     
		0   10.5    0       0      0     
		0    0    .015      0      0     
		0  250.5    0     -280    33.32  
		0    0      0       0     12     
	*/

	// TODO REMOVE TEMP DEBUG
	//std::vector<float> Adense(rows*cols, 0.0f);

	/* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
	/*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
	/*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	for (i=0; i<nz; i++)
	{
		fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
		I[i]--;  /* adjust from 1-based to 0-based */
		J[i]--;

		//Adense[I[i] * rows + J[i]] = val[i];
		//if(symetric)
		//	Adense[J[i] * rows + I[i]] = val[i];
	}

	if (f !=stdin) fclose(f);

	/************************/
	/* now write out matrix */
	/************************/

	struct matCell{
		int i,j;
		float val;
	};

	struct less_than_key
	{
		inline bool operator() (const matCell& c1, const matCell& c2)
		{
			if (c1.i == c2.i)
				return (c1.j < c2.j);

			return (c1.i < c2.i);
		}
	};

	printf("converting matrix with %d nz to CSR\n",nz);


	int symNZcount = 0;
	if (symetric)
	{
		for (i=0; i<nz; i++)
		{
			if (I[i] != J[i])
				symNZcount++;
		}
	}

	std::vector<matCell> spMat(nz+symNZcount);
	//std::vector<matCell> symSpMat(nz);

	int ii=0;

	for (i=0; i<nz; i++)
	{
		spMat[i].i = I[i];
		spMat[i].j = J[i];
		spMat[i].val = (float)val[i];
		if (symetric && I[i] != J[i])
		{
			spMat[nz+ii].i = J[i];
			spMat[nz+ii].j = I[i];
			spMat[nz+ii].val = (float)val[i];
			ii++;
		}
	}

	nz += symNZcount;

	if (symetric)
	{
		printf("filling symmetric matrix, new size is %d\n",nz);
	}


	std::sort(spMat.begin(),spMat.end(),less_than_key());

	//int line=0;

	std::vector<float> CSR_val(nz);
	std::vector<int> CSR_J(nz);
	std::vector<int> lineStartInd(rows+1);

	int counter=0;
	int curLine = spMat[0].i;

	//depends on your architecture, mainly on the size of the shared memory
	const int blockSize = 256;

	//build index arrays
	maps::GraphMapper indexMapper(blockSize,false);
	indexMapper.init(rows);

	unsigned int maxNRank = 0;
	unsigned int curLineSize = 0;
	
	lineStartInd[0] = 0;

	for (std::vector<matCell>::iterator matIt = spMat.begin(); matIt != spMat.end(); ++matIt)
	{
		// TODO: look at vv
		indexMapper.addEdge(matIt->i,matIt->j);

		if (curLine != matIt->i)
		{
			maxNRank = max(maxNRank,curLineSize);
			curLineSize = 0;

			lineStartInd[curLine+1] = counter;
			//printf("line %d has %d values\n",curLine,lineStartInd[lineStartInd.size()-1]-lineStartInd[lineStartInd.size()-2]);
			curLine = matIt->i;
		}

		curLineSize++;

		CSR_val[counter] = matIt->val;
		CSR_J[counter] = matIt->j;

		counter++;
	}

	lineStartInd[rows]=counter;

	std::vector<float> x(cols,1.0f);

	srand (static_cast <unsigned> (123));

	for (int stam=0; stam<cols; stam++)
	{
		x[stam] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	std::vector<float> b(rows,0.0f);

	printf("doing SpMV on the CPU\n");

	//int line =0;

	for (int ind=0; ind<nz; ind++)
	{
		//if (line != spMat[ind].j)
		//{
			//printf("calculated b[%d] = %f\n",line,b[line]);
		//	line = spMat[ind].j;
		//}
		b[spMat[ind].i]+= x[spMat[ind].j] * spMat[ind].val;		
	}

	// CUDA
	float* d_A_val;
	unsigned int* d_A_j_ind;
	unsigned int* d_A_lineStartInd;
	float* d_x; 
	float* d_b;

	std::vector<float> gpu_b(rows,0.0f);

	maps::CudaAllocAndCopy((void**)&d_A_val,(void*)&(*CSR_val.begin()),CSR_val.size()*sizeof(float));
	maps::CudaAllocAndCopy((void**)&d_A_j_ind,(void*)&(*CSR_J.begin()),CSR_J.size()*sizeof(int));
	maps::CudaAllocAndCopy((void**)&d_A_lineStartInd,(void*)&(*lineStartInd.begin()),lineStartInd.size()*sizeof(int));

	maps::CudaAllocAndCopy((void**)&d_x,(void*)&(*x.begin()),x.size()*sizeof(float));
	maps::CudaAllocAndClear((void**)&d_b,b.size()*sizeof(float));

	dim3 blockDim(blockSize,1,1);
	dim3 gridDim(maps::RoundUp(rows, blockDim.x),1,1);

	printf("launching kernel 1\n");

	
	for (int i = 0; i < REPETITIONS; i++)
	{
		SPmV_maps_kernel_naive <<<gridDim, blockDim>>> (rows, d_A_val, d_A_j_ind, d_A_lineStartInd, d_x, d_b);
	}

	MAPS_CUDA_CHECK(cudaGetLastError());
	MAPS_CUDA_CHECK(cudaDeviceSynchronize());

	compareGpuCpuVecs<float>(d_b, (float*)&(*b.begin()), b.size(), "SpMV", 0.00001f,true);
	

	printf("after kernel 1\n");

	//printf("A = \n");	
	//for(int i = 0; i < rows; i++)
	//{
	//	for(int j = 0; j < cols; j++)
	//		printf("%f ", Adense[i * cols + j]);
	//	printf("\n");
	//}

	indexMapper.setMaxNodeRankSize(maxNRank);

	indexMapper.createIndexMap();

	printf("max items in line %d\n",maxNRank);

	int sharedMemSize_con = sizeof(float)*indexMapper._MaxNumOfConstVecsInBlock;
	unsigned int numPartRoundUp = maps::RoundUp(cols,512)*512;

	MAPS_CUDA_CHECK(cudaMemset(d_b, 0, b.size()*sizeof(float)));

	for (int i = 0; i < REPETITIONS; i++)
	{
		SPmV_maps_kernel_maps <<<gridDim, blockDim, sharedMemSize_con>>> (rows, d_A_val, d_A_j_ind, d_A_lineStartInd, d_x, d_b, indexMapper._gpuData, cols, indexMapper._MaxNumOfConstVecsInBlock, numPartRoundUp);
	}

	MAPS_CUDA_CHECK(cudaGetLastError());
	MAPS_CUDA_CHECK(cudaDeviceSynchronize());

	compareGpuCpuVecs<float>(d_b, (float*)&(*b.begin()), b.size(), "SpMV MAPS", 0.00001f,true);

	return 0;
}

