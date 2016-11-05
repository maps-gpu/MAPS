// MAPS - Memory Access Pattern Specification Framework
// http://maps-gpu.github.io/
// Copyright (c) 2015, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __MAPS_MACRO_HELPERS_H_
#define __MAPS_MACRO_HELPERS_H_

#ifdef _MSC_VER
#define MAPS_PRAGMA(x) __pragma(x)
#else
#define MAPS_PRAGMA(x) _Pragma(#x)
#endif


#define __MAPS_EXPAND(x) x
#define __MAPS_GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,NAME,...) NAME

#define __MAPS_FE_0(WHAT, IND)
#define __MAPS_FE_1(WHAT, IND, X) WHAT(IND, X)
#define __MAPS_FE_2(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_1(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_3(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_2(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_4(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_3(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_5(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_4(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_6(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_5(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_7(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_6(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_8(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_7(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_9(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_8(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_10(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_9(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_11(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_10(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_12(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_11(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_13(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_12(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_14(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_13(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_15(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_14(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_16(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_15(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_17(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_16(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_18(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_17(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_19(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_18(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_FE_20(WHAT, IND, X, ...) WHAT(IND, X)__MAPS_EXPAND(__MAPS_FE_19(WHAT, IND+1, __VA_ARGS__))
#define __MAPS_PP_FOR_EACH(action,...) \
  __MAPS_EXPAND(__MAPS_GET_MACRO(__VA_ARGS__,__MAPS_FE_20,__MAPS_FE_19,__MAPS_FE_18,__MAPS_FE_17,__MAPS_FE_16,__MAPS_FE_15,__MAPS_FE_14,__MAPS_FE_13,__MAPS_FE_12,__MAPS_FE_11,__MAPS_FE_10,__MAPS_FE_9,__MAPS_FE_8,__MAPS_FE_7,__MAPS_FE_6,__MAPS_FE_5,__MAPS_FE_4,__MAPS_FE_3,__MAPS_FE_2,__MAPS_FE_1,__MAPS_FE_0)(action,0, __VA_ARGS__))


#endif  // __MAPS_MACRO_HELPERS_H_
