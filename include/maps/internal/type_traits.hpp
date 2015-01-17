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

#ifndef __MAPS_TYPE_TRAITS_CUH_
#define __MAPS_TYPE_TRAITS_CUH_

namespace maps
{
	// Remove qualifiers from type
	template<typename T>
	struct RemoveConst
	{
		typedef T type;
	};

	template<typename T>
	struct RemoveConst<const T>
	{
		typedef T type;
	};

	template<typename T>
	struct RemoveVolatile
	{
		typedef T type;
	};

	template<typename T>
	struct RemoveVolatile<volatile T>
	{
		typedef T type;
	};

	template<typename T>
	struct RemoveQualifiers
	{
		typedef typename RemoveConst<typename RemoveVolatile<T>::type>::type type;
	};


	//////////////////////////////////////////////////////////////////////////

	template <typename T>
	struct _IsIntegral
	{
		enum { value = false };
	};

	#ifdef __MAPS_IS_INTEGRAL_TYPE
	#error Using disallowed macro name __MAPS_IS_INTEGRAL_TYPE
	#endif

	#define __MAPS_IS_INTEGRAL_TYPE(type, val)			\
	template<>											\
	struct _IsIntegral<type>							\
	{													\
		enum { value = val };							\
	};

	__MAPS_IS_INTEGRAL_TYPE(bool, true);
	__MAPS_IS_INTEGRAL_TYPE(char, true);
	__MAPS_IS_INTEGRAL_TYPE(unsigned char, true);
	__MAPS_IS_INTEGRAL_TYPE(signed char, true);
	__MAPS_IS_INTEGRAL_TYPE(unsigned short, true);
	__MAPS_IS_INTEGRAL_TYPE(signed short, true);
	__MAPS_IS_INTEGRAL_TYPE(unsigned int, true);
	__MAPS_IS_INTEGRAL_TYPE(signed int, true);
	__MAPS_IS_INTEGRAL_TYPE(unsigned long, true);
	__MAPS_IS_INTEGRAL_TYPE(signed long, true);

	// Determines whether T is of integer type
	template <typename T>
	struct IsIntegral : public _IsIntegral<typename RemoveQualifiers<T>::type>
	{		
	};
	
	#undef __MAPS_IS_INTEGRAL_TYPE

}  // namespace maps

#endif  // __MAPS_TYPE_TRAITS_CUH_
