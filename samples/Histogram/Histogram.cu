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
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <maps/maps.cuh>

#define DATA_SIZE 16384
#define NUMBER_OF_BINS 256

#define MIN_VAL 0.0f
#define MAX_VAL 255.0f

#define BW 256
#define BH 1

template<typename T, typename BinT, int BINS>
void HistogramNaive(const T *in, size_t size, BinT *hist, const T& minVal, const T& maxVal)
{
	for(size_t idx = 0; idx < size; ++idx)
	{
		// Compute the relevant bin and add to that
		BinT bin = static_cast<BinT>((float)(BINS - 1) * (float)(in[idx] - minVal) / (float)(maxVal - minVal));
		++hist[bin];
	}
}

template<typename T, typename BinT, int BINS>
__global__ void HistogramMAPS(const T *in, size_t size, BinT *outHist, T minVal, T maxVal)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	// Auto-choose best histogram algorithm
	//__shared__ typename maps::Histogram<BinT, BINS>::DevHistogram hist;
	__shared__ maps::HistogramSharedAtomic<BinT, BINS> hist;

	hist.init(outHist);

	// Compute the relevant bin and add to that
	BinT bin = static_cast<BinT>((float)(BINS - 1) * (float)(in[idx] - minVal) / (float)(maxVal - minVal));
	hist.compute(bin);

	hist.commit();
}

int main(int argc, char **argv)
{
	float *dev_data = NULL;
	std::vector<unsigned int> histNaive (NUMBER_OF_BINS, 0);
	unsigned int *dev_histMAPS = NULL;

	size_t dataSize = DATA_SIZE;
	
	// Create input data
	std::vector<float> host_data (dataSize, 0);
	for(size_t i = 0; i < dataSize; ++i)
		host_data[i] = static_cast<float>(i % NUMBER_OF_BINS);

	// Allocate GPU buffers
	MAPS_CUDA_CHECK(cudaMalloc(&dev_data, sizeof(float) * dataSize));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_histMAPS, sizeof(unsigned int) * NUMBER_OF_BINS));	
	   	
	// Copy input data to GPU
	MAPS_CUDA_CHECK(cudaMemcpy(dev_data, &host_data[0], sizeof(float) * dataSize, cudaMemcpyHostToDevice));
	MAPS_CUDA_CHECK(cudaMemset(dev_histMAPS, 0, sizeof(unsigned int) * NUMBER_OF_BINS));

	dim3 block_dims (BW, BH, 1);
	dim3 grid_dims (maps::RoundUp(dataSize, block_dims.x), maps::RoundUp(1, block_dims.y), 1);

	// Run both versions
	HistogramNaive<float, unsigned int, NUMBER_OF_BINS>(&host_data[0], dataSize, &histNaive[0], MIN_VAL, MAX_VAL);

	HistogramMAPS<float, unsigned int, NUMBER_OF_BINS><<<grid_dims, block_dims>>>(dev_data, dataSize, dev_histMAPS, MIN_VAL, MAX_VAL);

	MAPS_CUDA_CHECK(cudaDeviceSynchronize());

	// Copy and compare the results
	std::vector<unsigned int> host_histMAPS (NUMBER_OF_BINS, 0);
	
	MAPS_CUDA_CHECK(cudaMemcpy(&host_histMAPS[0], dev_histMAPS, sizeof(unsigned int) * NUMBER_OF_BINS, cudaMemcpyDeviceToHost));

	int numErrorsMAPS = 0;
	float meanErrorMAPS = 0.0f;

	// Do not compare the results in the outer borders
	for(size_t i = 0; i < NUMBER_OF_BINS; ++i)
	{
		// Test Naive vs. MAPS
		if(histNaive[i] != host_histMAPS[i])
		{
			if(numErrorsMAPS == 0)
				printf("MAPS: First error in index %d: %d != %d\n", i, 
						histNaive[i], host_histMAPS[i]);

			numErrorsMAPS++;
		}
		meanErrorMAPS += fabs((float)histNaive[i] - (float)host_histMAPS[i]);
	}

	printf("Number of errors: Naive vs. MAPS = %d\n", numErrorsMAPS);
	printf("Mean error:       Naive vs. MAPS = %f\n", meanErrorMAPS / (float)(NUMBER_OF_BINS));

	// Free allocated data
	MAPS_CUDA_CHECK(cudaFree(dev_data));
	MAPS_CUDA_CHECK(cudaFree(dev_histMAPS));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

	printf("Done!\n");
    
    return 0;
}
