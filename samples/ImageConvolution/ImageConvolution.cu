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

#define IMAGE_WIDTH 4096
#define IMAGE_HEIGHT 2560

#define BW 32
#define BH 32

#define REPETITIONS 100

// Unique ID for conv2 input image texture (required for working with maps::Window2DTexture)
#define IMAGE_TEXTURE_UID 1111

#define KERNEL_RADIUS 4

__constant__ float dev_convKernel[2*KERNEL_RADIUS+1][2*KERNEL_RADIUS+1];

// Simple convolution kernel
float g_convKernel[(2*KERNEL_RADIUS+1) * (2*KERNEL_RADIUS+1)] = 
{
	1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
	2.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
	3.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
	4.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
	5.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
	6.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
	7.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
	8.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
	9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f
};

template<int RADIUS>
__global__ void conv2Naive(const float *in, size_t inStride, 
						   float *out, size_t outStride, 
						   int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	float result = 0.0f;

	for (int ky = 0 ; ky <= 2 * RADIUS; ky++)
		for (int kx = 0 ; kx <= 2 * RADIUS; kx++)
			result += in[(y - RADIUS + ky) * inStride + (x - RADIUS + kx)] * dev_convKernel[ky][kx];

	out[y * outStride + x] = result;
}

template<int RADIUS, int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void conv2MAPS(const float *in, size_t inStride, 
						  float *out, size_t outStride, 
						  int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;
	
	typedef maps::Window2D<float, BLOCK_WIDTH, BLOCK_HEIGHT, RADIUS> window2DType;
	__shared__ window2DType wnd;

	wnd.init(in, width, height);
		
	float result = 0.0f;

	typename window2DType::iterator iter = wnd.begin();

	#pragma unroll
	for (int ky = 0 ; ky <= 2 * RADIUS; ky++)
		#pragma unroll
		for (int kx = 0 ; kx <= 2 * RADIUS; kx++)
		{
			result += (*iter) * dev_convKernel[ky][kx];
			++iter;
		}

	out[y * outStride + x] = result;
}

template<int RADIUS, int BLOCK_WIDTH, int BLOCK_HEIGHT, int TEXTURE_UID>
__global__ void conv2MAPSTexture(float *out, size_t outStride, 
							     int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;
	
	typedef maps::Window2DTexture<float, TEXTURE_UID, BLOCK_WIDTH, BLOCK_HEIGHT, RADIUS> window2DType;
	__shared__ window2DType wnd;

	wnd.init();
		
	float result = 0.0f;

	typename window2DType::iterator iter = wnd.begin();

	#pragma unroll
	for (int ky = 0 ; ky <= 2 * RADIUS; ky++)
		#pragma unroll
		for (int kx = 0 ; kx <= 2 * RADIUS; kx++)
		{
			result += (*iter) * dev_convKernel[ky][kx];
			++iter;
		}

	out[y * outStride + x] = result;
}

int main(int argc, char **argv)
{
	float *dev_image = NULL, *dev_naiveResult = NULL, 
		  *dev_MAPSResult = NULL, *dev_MAPSTexResult = NULL;
	size_t width = IMAGE_WIDTH, height = IMAGE_HEIGHT, imageStride = 0;
	
	// Create input data
	std::vector<float> host_image (width * height, 0);
	for(size_t i = 0; i < width * height; ++i)
		host_image[i] = static_cast<float>(i % width);

	// Allocate GPU buffers
	MAPS_CUDA_CHECK(cudaMallocPitch(&dev_image, &imageStride, sizeof(float) * width, height));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_naiveResult, sizeof(float) * width * height));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSResult, sizeof(float) * width * height));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSTexResult, sizeof(float) * width * height));	
	   	
	// Create GPU texture

	// Set texture parameters
	typedef typename maps::UniqueTexRef2D<float>::template TexId<IMAGE_TEXTURE_UID> TexId;

    TexId::tex.addressMode[0] = cudaAddressModeClamp;
    TexId::tex.addressMode[1] = cudaAddressModeClamp;
    TexId::tex.filterMode = cudaFilterModeLinear;

	// Bind texture to data
	MAPS_CUDA_CHECK(TexId::BindTexture(dev_image, width, height, imageStride));


	// Copy input data to GPU
	MAPS_CUDA_CHECK(cudaMemcpyToSymbol(dev_convKernel, g_convKernel, 
									   sizeof(float) * (2*KERNEL_RADIUS+1) * (2*KERNEL_RADIUS+1), 
									   0, cudaMemcpyHostToDevice));
	MAPS_CUDA_CHECK(cudaMemcpy2D(dev_image, imageStride, &host_image[0], sizeof(float) * width, 
								 sizeof(float) * width, height, cudaMemcpyHostToDevice));

	dim3 block_dims (BW, BH, 1);
	dim3 grid_dims (maps::RoundUp(width, block_dims.x), maps::RoundUp(height, block_dims.y), 1);

	// Run all three versions
	for(int i = 0; i < REPETITIONS; i++)
	{
		conv2Naive<KERNEL_RADIUS> <<<grid_dims, block_dims>>>(dev_image, imageStride / sizeof(float),
															  dev_naiveResult, width, 
															  width, height);

		conv2MAPS<KERNEL_RADIUS, BW, BH> <<<grid_dims, block_dims>>>(dev_image, imageStride / sizeof(float),
																	 dev_MAPSResult, width, 
																	 width, height);

		conv2MAPSTexture<KERNEL_RADIUS, BW, BH, IMAGE_TEXTURE_UID> 
			<<<grid_dims, block_dims>>>(dev_MAPSTexResult, width, width, height);
	}

	MAPS_CUDA_CHECK(cudaDeviceSynchronize());

	MAPS_CUDA_CHECK(TexId::UnbindTexture());

	// Copy and compare the results
	std::vector<float> host_resultNaive (width * height, 0), host_resultMAPS (width * height, 0),
					   host_resultMAPSTex (width * height, 0);
	
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultNaive[0], dev_naiveResult, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultMAPS[0], dev_MAPSResult, sizeof(float) * width * height, cudaMemcpyDeviceToHost));
	MAPS_CUDA_CHECK(cudaMemcpy(&host_resultMAPSTex[0], dev_MAPSTexResult, sizeof(float) * width * height, cudaMemcpyDeviceToHost));

	int numErrorsMAPS = 0, numErrorsMAPSTex = 0;
	float meanErrorMAPS = 0.0f, meanErrorMAPSTex = 0.0f;

	// Do not compare the results in the outer borders
	for(size_t y = KERNEL_RADIUS; y < height - KERNEL_RADIUS; ++y)
	{
		for(size_t x = KERNEL_RADIUS; x < width - KERNEL_RADIUS; ++x)
		{
			// Test Naive vs. MAPS
			if(fabs(host_resultNaive[y * width + x] - host_resultMAPS[y * width + x]) > 1e-6)
			{
				if(numErrorsMAPS == 0)
					printf("MAPS: First error in (%d, %d): %f != %f\n", x, y, 
						   host_resultNaive[y * width + x], host_resultMAPS[y * width + x]);

				numErrorsMAPS++;
			}
			meanErrorMAPS += fabs(host_resultNaive[y * width + x] - host_resultMAPS[y * width + x]);

			// Test Naive vs. MAPS (Texture)
			if(fabs(host_resultNaive[y * width + x] - host_resultMAPSTex[y * width + x]) > 1e-6)
			{
				if(numErrorsMAPSTex == 0)
					printf("MAPS(Texture): First error in (%d, %d): %f != %f\n", x, y, 
						   host_resultNaive[y * width + x], host_resultMAPSTex[y * width + x]);

				numErrorsMAPSTex++;
			}
			meanErrorMAPSTex += fabs(host_resultNaive[y * width + x] - host_resultMAPSTex[y * width + x]);
		}
	}

	printf("Number of errors: Naive vs. MAPS = %d, Naive vs. MAPS(Texture) = %d\n", numErrorsMAPS, numErrorsMAPSTex);
	printf("Mean error:       Naive vs. MAPS = %f, Naive vs. MAPS(Texture) = %f\n", 
		   meanErrorMAPS    / (float)((width - 2*KERNEL_RADIUS) * (height - 2*KERNEL_RADIUS)), 
		   meanErrorMAPSTex / (float)((width - 2*KERNEL_RADIUS) * (height - 2*KERNEL_RADIUS)));

	// Free allocated data
	MAPS_CUDA_CHECK(cudaFree(dev_image));
	MAPS_CUDA_CHECK(cudaFree(dev_naiveResult));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSResult));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSTexResult));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

	printf("Done!\n");
    
    return 0;
}
