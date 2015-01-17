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

#define BUFFER_SIZE 16384

#define BW 128

#define REPETITIONS 1000

// Unique ID for convolution input texture (required for working with maps::Window1DTexture)
#define BUFFER_TEXTURE_UID 2222

#define KERNEL_RADIUS 4

__constant__ float dev_convKernel[2*KERNEL_RADIUS+1];

// Simple convolution kernel
float g_convKernel[(2*KERNEL_RADIUS+1)] = 
{
	1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
};

template<int RADIUS>
__global__ void convNaive(const float *in, float *out, 
						  int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= size)
		return;

	float result = 0.0f;

	for (int kx = 0 ; kx <= 2 * RADIUS; kx++)
		result += in[x - RADIUS + kx] * dev_convKernel[kx];

	out[x] = result;
}

template<int RADIUS, int BLOCK_WIDTH>
__global__ void convMAPS(const float *in, float *out, int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x >= size)
		return;
	
	typedef maps::Window1D<float, BLOCK_WIDTH, RADIUS> window1DType;
	__shared__ window1DType wnd;

	wnd.init(in, size);
		
	float result = 0.0f;

	typename window1DType::iterator iter = wnd.begin();

	#pragma unroll
	for (int kx = 0 ; kx <= 2 * RADIUS; kx++)
	{
		result += (*iter) * dev_convKernel[kx];
		++iter;
	}

	out[x] = result;
}

template<int RADIUS, int BLOCK_WIDTH, int TEXTURE_UID>
__global__ void convMAPSTexture(float *out, int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= size)
		return;
	
	typedef maps::Window1DTexture<float, TEXTURE_UID, BLOCK_WIDTH, RADIUS> window1DType;
	__shared__ window1DType wnd;

	wnd.init();
		
	float result = 0.0f;

	typename window1DType::iterator iter = wnd.begin();

	#pragma unroll
	for (int kx = 0 ; kx <= 2 * RADIUS; kx++)
	{
		result += (*iter) * dev_convKernel[kx];
		++iter;
	}

	out[x] = result;
}

int main(int argc, char **argv)
{
	float *dev_buffer = NULL, *dev_naiveResult = NULL, 
		  *dev_MAPSResult = NULL, *dev_MAPSTexResult = NULL;
	size_t size = BUFFER_SIZE;
	
	// Create input data
	std::vector<float> host_buffer (size, 0);
	for(size_t i = 0; i < size; ++i)
		host_buffer[i] = static_cast<float>(i % 995);

	// Allocate GPU buffers
	MAPS_CUDA_CHECK(cudaMalloc(&dev_buffer, sizeof(float) * size));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_naiveResult, sizeof(float) * size));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSResult, sizeof(float) * size));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSTexResult, sizeof(float) * size));	
	   	
	// Create GPU texture

	// Set texture parameters
	typedef typename maps::UniqueTexRef1D<float>::template TexId<BUFFER_TEXTURE_UID> TexId;

    TexId::tex.addressMode[0] = cudaAddressModeClamp;
    TexId::tex.filterMode = cudaFilterModeLinear;

	// Bind texture to data
	MAPS_CUDA_CHECK(TexId::BindTexture(dev_buffer, sizeof(float) * size));


	// Copy input data to GPU
	MAPS_CUDA_CHECK(cudaMemcpyToSymbol(dev_convKernel, g_convKernel, 
									   sizeof(float) * (2*KERNEL_RADIUS+1), 
									   0, cudaMemcpyHostToDevice));
	MAPS_CUDA_CHECK(cudaMemcpy(dev_buffer, &host_buffer[0], sizeof(float) * size, 
							   cudaMemcpyHostToDevice));

	dim3 block_dims (BW, 1, 1);
	dim3 grid_dims (maps::RoundUp(size, block_dims.x), 1, 1);

	// Run all three versions
	for(int i = 0; i < REPETITIONS; i++)
	{
		convNaive<KERNEL_RADIUS> <<<grid_dims, block_dims>>>(dev_buffer, dev_naiveResult, size);

		convMAPS<KERNEL_RADIUS, BW> <<<grid_dims, block_dims>>>(dev_buffer, dev_MAPSResult, size);

		convMAPSTexture<KERNEL_RADIUS, BW, BUFFER_TEXTURE_UID> 
			<<<grid_dims, block_dims>>>(dev_MAPSTexResult, size);
	}

	MAPS_CUDA_CHECK(cudaDeviceSynchronize());

	MAPS_CUDA_CHECK(TexId::UnbindTexture());

	// Copy and compare the results
	std::vector<float> host_resultNaive (size, 0), host_resultMAPS (size, 0),
					   host_resultMAPSTex (size, 0);
	
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultNaive[0], dev_naiveResult, sizeof(float) * size, cudaMemcpyDeviceToHost));
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultMAPS[0], dev_MAPSResult, sizeof(float) * size, cudaMemcpyDeviceToHost));
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultMAPSTex[0], dev_MAPSTexResult, sizeof(float) * size, cudaMemcpyDeviceToHost));

	int numErrorsMAPS = 0, numErrorsMAPSTex = 0;
	float meanErrorMAPS = 0.0f, meanErrorMAPSTex = 0.0f;

	// Do not compare the results in the outer borders
	for(size_t x = KERNEL_RADIUS; x < size - KERNEL_RADIUS; ++x)
	{
		// Test Naive vs. MAPS
		if(fabs(host_resultNaive[x] - host_resultMAPS[x]) > 1e-6)
		{
			if(numErrorsMAPS == 0)
				printf("MAPS: First error in %d: %f != %f\n", x, 
						host_resultNaive[x], host_resultMAPS[x]);

			numErrorsMAPS++;
		}
		meanErrorMAPS += fabs(host_resultNaive[x] - host_resultMAPS[x]);

		// Test Naive vs. MAPS (Texture)
		if(fabs(host_resultNaive[x] - host_resultMAPSTex[x]) > 1e-6)
		{
			if(numErrorsMAPSTex == 0)
				printf("MAPS(Texture): First error in %d: %f != %f\n", x,
						host_resultNaive[x], host_resultMAPSTex[x]);

			numErrorsMAPSTex++;
		}
		meanErrorMAPSTex += fabs(host_resultNaive[x] - host_resultMAPSTex[x]);
	}

	printf("Number of errors: Naive vs. MAPS = %d, Naive vs. MAPS(Texture) = %d\n", numErrorsMAPS, numErrorsMAPSTex);
	printf("Mean error:       Naive vs. MAPS = %f, Naive vs. MAPS(Texture) = %f\n", 
		   meanErrorMAPS    / (float)((size - 2*KERNEL_RADIUS)), 
		   meanErrorMAPSTex / (float)((size - 2*KERNEL_RADIUS)));

	// Free allocated data
	MAPS_CUDA_CHECK(cudaFree(dev_buffer));
	MAPS_CUDA_CHECK(cudaFree(dev_naiveResult));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSResult));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSTexResult));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

	printf("Done!\n");
    
    return 0;
}
