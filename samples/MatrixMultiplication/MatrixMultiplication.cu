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

#define MATRIX_WIDTH 1024

#define BW 32

#define REPETITIONS 100

template<typename T>
__global__ void MatrixMultiplicationNaive(const T *A, size_t wA, const T *B, 
										  size_t hAwB, size_t hB, T *C)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float result = 0.0f;

	for (int i = 0; i < wA; ++i)
    {
 		result += A[y * wA + i] * B[i * wA + x];
    }

	C[y * wA + x] = result;
}

template<typename T, int BLOCK_SIZE>
__global__ void MatrixMultiplicationMAPS(const T *A, size_t wA, const T *B, 
										 size_t hAwB, size_t hB, T *C)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	typedef maps::Block2D<false, float, BLOCK_SIZE> B2D_t;
    typedef maps::Block2D<true,  float, BLOCK_SIZE> B2DT_t;

	__shared__ maps::Block2DData<false, float, BLOCK_SIZE> AData;
    __shared__ maps::Block2DData<true,  float, BLOCK_SIZE> BData;

	B2D_t  matConA;
    B2DT_t matConB;
     
    matConA.init(A, wA, AData);
    matConB.init(B, hB, BData);

	float result = 0.0f;

    typename B2D_t::iterator  matAIt, matAEnd;
    typename B2DT_t::iterator matBIt;
    do
    {		
		matConA.nextChunk();
		matConB.nextChunk();

		matAEnd = matConA.end();

		#pragma unroll
        for (matAIt = matConA.begin(), matBIt = matConB.begin();
             matAIt != matAEnd; ++matAIt, ++matBIt)
        {
            result += (*matAIt) * (*matBIt);
        }
    } while (!matConA.isDone());

	C[y * wA + x] = result;
}

template<typename T, int BLOCK_SIZE>
__global__ void MatrixMultiplicationMAPSAdvanced(const T *A, size_t wA, const T *B, 
												 size_t hAwB, size_t hB, T *C)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	typedef maps::Block2D<false, float, BLOCK_SIZE> B2D_t;
    typedef maps::Block2D<true,  float, BLOCK_SIZE> B2DT_t;

	__shared__ maps::Block2DData<false, float, BLOCK_SIZE> AData;
    __shared__ maps::Block2DData<true,  float, BLOCK_SIZE> BData;

	B2D_t  matConA;
    B2DT_t matConB;
     
    matConA.init(A, wA, AData);
    matConB.init(B, hB, BData);

	float result = 0.0f;

    typename B2D_t::iterator  matAIt, matAEnd;
    typename B2DT_t::iterator matBIt;
    do
    {		
		// Using the async API, we load both chunks simultaneously, saving two
		// block synchronization operations in the process
		__syncthreads();
		matConA.nextChunkAsync();
		matConB.nextChunkAsync();
		__syncthreads();

		matAEnd = matConA.end();

		#pragma unroll
        for (matAIt = matConA.begin(), matBIt = matConB.begin();
             matAIt != matAEnd; ++matAIt, ++matBIt)
        {
            result += (*matAIt) * (*matBIt);
        }
    } while (!matConA.isDone());

	C[y * wA + x] = result;
}

template <typename T>
inline T FPRandom(const T& min, const T& max)
{
	return (((T)rand() / (T)RAND_MAX) * (max - min)) + min;
}


int main(int argc, char **argv)
{
	float *dev_A = NULL, *dev_B = NULL, 
		  *dev_naiveResult = NULL, *dev_MAPSResult = NULL, *dev_MAPSAdvResult = NULL;
	size_t matSize = MATRIX_WIDTH * MATRIX_WIDTH;
	
	// Randomize
	srand((unsigned int)time(NULL));
	
	// Create input data
	std::vector<float> host_A (matSize, 0), host_B (matSize, 0);
	for(size_t i = 0; i < matSize; ++i)
	{
		host_A[i] = FPRandom<float>(-5.0f, 5.0f);
		host_B[i] = FPRandom<float>(-5.0f, 5.0f);
	}

	// Allocate GPU buffers
	MAPS_CUDA_CHECK(cudaMalloc(&dev_A, sizeof(float) * matSize));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_B, sizeof(float) * matSize));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_naiveResult, sizeof(float) * matSize));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSResult, sizeof(float) * matSize));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSAdvResult, sizeof(float) * matSize));
	   	
	// Copy input data to GPU
	MAPS_CUDA_CHECK(cudaMemcpy(dev_A, &host_A[0], sizeof(float) * matSize, cudaMemcpyHostToDevice));
	MAPS_CUDA_CHECK(cudaMemcpy(dev_B, &host_B[0], sizeof(float) * matSize, cudaMemcpyHostToDevice));

	dim3 block_dims (BW, BW, 1);
	dim3 grid_dims (maps::RoundUp(MATRIX_WIDTH, block_dims.x), maps::RoundUp(MATRIX_WIDTH, block_dims.y), 1);

	// Run all three versions
	for(int i = 0; i < REPETITIONS; i++)
	{
		MatrixMultiplicationNaive<float> 
			<<<grid_dims, block_dims>>>(dev_A, MATRIX_WIDTH, dev_B, MATRIX_WIDTH,
										MATRIX_WIDTH, dev_naiveResult);

		MatrixMultiplicationMAPS<float, BW> 
			<<<grid_dims, block_dims>>>(dev_A, MATRIX_WIDTH, dev_B, MATRIX_WIDTH,
										MATRIX_WIDTH, dev_MAPSResult);

		MatrixMultiplicationMAPSAdvanced<float, BW> 
			<<<grid_dims, block_dims>>>(dev_A, MATRIX_WIDTH, dev_B, MATRIX_WIDTH,
										MATRIX_WIDTH, dev_MAPSAdvResult);
	}

	MAPS_CUDA_CHECK(cudaDeviceSynchronize());

	// Copy and compare the results
	std::vector<float> host_resultNaive (matSize, 0), host_resultMAPS (matSize, 0),
		               host_resultMAPSAdv (matSize, 0);
	
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultNaive[0], dev_naiveResult, sizeof(float) * matSize, cudaMemcpyDeviceToHost));
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultMAPS[0], dev_MAPSResult,   sizeof(float) * matSize, cudaMemcpyDeviceToHost));
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultMAPSAdv[0], dev_MAPSAdvResult,   sizeof(float) * matSize, cudaMemcpyDeviceToHost));

	int numErrorsMAPS = 0, numErrorsMAPSAdv = 0;
	float meanErrorMAPS = 0.0f, meanErrorMAPSAdv = 0.0f;

	// Do not compare the results in the outer borders
	for(size_t i = 0; i < matSize; ++i)
	{
		// Test Naive vs. MAPS
		if(fabs(host_resultNaive[i] - host_resultMAPS[i]) > 1e-6)
		{
			if(numErrorsMAPS == 0)
				printf("MAPS: First error in (%d, %d): %f != %f\n", i % MATRIX_WIDTH, i / MATRIX_WIDTH, 
						host_resultNaive[i], host_resultMAPS[i]);

			numErrorsMAPS++;
		}
		meanErrorMAPS += fabs(host_resultNaive[i] - host_resultMAPS[i]);

		// Test Naive vs. MAPS (async API)
		if(fabs(host_resultNaive[i] - host_resultMAPSAdv[i]) > 1e-6)
		{
			if(numErrorsMAPSAdv == 0)
				printf("MAPS (async API): First error in (%d, %d): %f != %f\n", i % MATRIX_WIDTH, i / MATRIX_WIDTH, 
						host_resultNaive[i], host_resultMAPSAdv[i]);

			numErrorsMAPSAdv++;
		}
		meanErrorMAPSAdv += fabs(host_resultNaive[i] - host_resultMAPSAdv[i]);
	}

	printf("Number of errors: Naive vs. MAPS = %d, Naive vs. MAPS (async API) = %d\n", numErrorsMAPS, numErrorsMAPSAdv);
	printf("Mean error:       Naive vs. MAPS = %f, Naive vs. MAPS (async API) = %f\n", 
		   meanErrorMAPS    / (float)matSize,
		   meanErrorMAPSAdv    / (float)matSize);

	// Free allocated data
	MAPS_CUDA_CHECK(cudaFree(dev_A));
	MAPS_CUDA_CHECK(cudaFree(dev_B));
	MAPS_CUDA_CHECK(cudaFree(dev_naiveResult));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSResult));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSAdvResult));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

	printf("Done!\n");
    
    return 0;
}
