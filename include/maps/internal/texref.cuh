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

#ifndef __MAPS_TEXREF_CUH_
#define __MAPS_TEXREF_CUH_

#include <cuda_runtime.h>
#include <iterator>

namespace maps
{


	/// @brief A workaround to use 1D texture references (pre-Kepler) in the library. (logic inspired by CUB library)
	/// We must wrap it with two classes, so as to avoid commas in the CUDA generated code.
	template <typename T>
	struct UniqueTexRef1D
	{
		template <int TEXTURE_UID>
		struct TexId
		{
			typedef texture<T, cudaTextureType1D, cudaReadModeElementType> TexRefType;
			static TexRefType tex;
			
			template<typename DiffType>
			static __device__ __forceinline__ T read(DiffType offset) { return tex1Dfetch(tex, offset); }

			/// Bind texture
			static __host__ cudaError_t BindTexture(const void *d_in, size_t size)
			{
				if (d_in)
				{
					cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<T>();
					tex.channelDesc = tex_desc;
					return cudaBindTexture(NULL, &tex, d_in, &tex_desc, size);
				}

				return cudaSuccess;
			}

			/// Unbind texture
			static __host__ cudaError_t UnbindTexture()
			{
				return cudaUnbindTexture(&tex);
			}
		};
	};

	template <typename T>
	template <int TEXTURE_UID>
	typename UniqueTexRef1D<T>::template TexId<TEXTURE_UID>::TexRefType UniqueTexRef1D<T>::template TexId<TEXTURE_UID>::tex = 0;

	/// @brief A workaround to use 2D texture references (pre-Kepler) in the library. (logic inspired by CUB library)
	/// We must wrap it with two classes, so as to avoid commas in the CUDA generated code.
	template <typename T>
	struct UniqueTexRef2D
	{
		template <int TEXTURE_UID>
		struct TexId
		{
			typedef texture<T, cudaTextureType2D, cudaReadModeElementType> TexRefType;
			static TexRefType tex;

			template<typename DiffType>
			static __device__ __forceinline__ T read(DiffType x, DiffType y) { return tex2D(tex, x, y); }

			/// Bind texture
			static cudaError_t BindTexture(const void *d_in, size_t width, size_t height, size_t stride)
			{
				if (d_in)
				{
					cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<T>();
					tex.channelDesc = tex_desc;
					return cudaBindTexture2D(NULL, &tex, d_in, &tex_desc, width, height, stride);
				}

				return cudaSuccess;
			}

			/// Unbind texture
			static cudaError_t UnbindTexture()
			{
				return cudaUnbindTexture(&tex);
			}
		};
	};

	template <typename T>
	template <int TEXTURE_UID>
	typename UniqueTexRef2D<T>::template TexId<TEXTURE_UID>::TexRefType UniqueTexRef2D<T>::template TexId<TEXTURE_UID>::tex = 0;

}  // namespace maps

#endif  // __MAPS_TEXREF_CUH_
