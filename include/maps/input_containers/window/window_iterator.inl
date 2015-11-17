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

#ifndef __MAPS_WINDOW_ITERATOR_INL_
#define __MAPS_WINDOW_ITERATOR_INL_

// Don't include this file directly

namespace maps {

    // TODO: ILP

    /// @brief Internal Window ND iterator class
    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
             int WINDOW_APRON, int IPX, int IPY, int IPZ, 
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS, 
             bool MULTI_GPU>
    class WindowIterator<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                         WINDOW_APRON, IPX, IPY, IPZ, BORDERS, TEXTURE_UID, GRS,
                         MULTI_GPU> 
        : public std::iterator<std::input_iterator_tag, T>
    {
    protected:
        int m_id;
        const T *m_sParentData;

        __device__  __forceinline__ void next()
        {
            ++m_id;
        }
    public:
        __device__ WindowIterator(
            unsigned int pos, const Window<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 
                                           BLOCK_DEPTH, WINDOW_APRON, IPX, IPY,
                                           IPZ, BORDERS, TEXTURE_UID, GRS, 
                                           MULTI_GPU>& parent)
        {
            m_id = 0;
            m_sParentData = parent.m_sdata + pos;
        }

        __device__ WindowIterator(const WindowIterator& other)
        {
            m_id = other.m_id;
            m_sParentData = other.m_sParentData;
        }

        __device__  __forceinline__ void operator=(const WindowIterator& other)
        {
            m_id = other.m_id;
            m_sParentData = other.m_sParentData;
        }

        __device__ __forceinline__ int index() const
        {
            return m_id;
        }

        __device__ __forceinline__ const T& operator*() const
        {
            return m_sParentData[m_id];
        }

        __device__  __forceinline__ WindowIterator& operator++() // Prefix
        {
            next();
            return *this;
        }

        __device__  __forceinline__ WindowIterator operator++(int) // Postfix
        {
            WindowIterator temp(*this);
            next();
            return temp;
        }

        __device__  __forceinline__ bool operator==(
            const WindowIterator& a) const
        {
            return (m_sParentData + m_id) == (a.m_sParentData + a.m_id);
        }
        __device__  __forceinline__ bool operator!=(
            const WindowIterator& a) const
        {
            return (m_sParentData + m_id) != (a.m_sParentData + a.m_id);
        }
    };

    /// @brief Internal Window ND iterator class
    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
             int WINDOW_APRON, int IPX, int IPY, int IPZ, 
             BorderBehavior BORDERS, int TEXTURE_UID, GlobalReadScheme GRS, 
             bool MULTI_GPU>
    class WindowIterator<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 
                         WINDOW_APRON, IPX, IPY, IPZ, BORDERS, TEXTURE_UID, GRS,
                         MULTI_GPU> 
        : public std::iterator<std::input_iterator_tag, T>
    {
    protected:
        unsigned int m_pos;
        int m_id;
        const T *m_sParentData;
        int m_initialOffset;

        enum
        {
            XSHARED = (BLOCK_WIDTH + WINDOW_APRON * 2),
            WIND_WIDTH = (WINDOW_APRON * 2 + 1),
        };

        __device__  __forceinline__ void next()
        {
            m_id++;
            m_pos = m_initialOffset + (m_id % WIND_WIDTH) + 
                ((m_id / WIND_WIDTH)* XSHARED);
        }
    public:
        __device__ WindowIterator(
            unsigned int pos, const Window<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 
                                           BLOCK_DEPTH, WINDOW_APRON, IPX, IPY,
                                           IPZ, BORDERS, TEXTURE_UID, GRS, 
                                           MULTI_GPU>& parent)
        {
            m_pos = pos;
            m_sParentData = parent.m_sdata;
            m_id = 0;
            m_initialOffset = pos;
        }

        __device__ WindowIterator(const WindowIterator& other)
        {
            m_pos = other.m_pos;
            m_sParentData = other.m_sParentData;
            m_id = other.m_id;
            m_initialOffset = other.m_initialOffset;
        }

        __device__  __forceinline__ void operator=(const WindowIterator& a)
        {
            m_id = a.m_id;
            m_pos = a.m_pos;
            m_initialOffset = a.m_initialOffset;
            m_sParentData = a.m_sParentData;
        }

        __device__ __forceinline__ int index() const
        {
            return m_id;
        }

        __device__ __forceinline__ const T& operator*() const
        {
            return m_sParentData[m_pos];
        }

        __device__  __forceinline__ WindowIterator& operator++() // Prefix
        {
            next();
            return *this;
        }

        __device__  __forceinline__ WindowIterator operator++(int) // Postfix
        {
            WindowIterator temp(*this);
            next();
            return temp;
        }

        __device__  __forceinline__ bool operator==(
            const WindowIterator& a) const
        {
            return m_pos == a.m_pos;
        }
        __device__  __forceinline__ bool operator!=(
            const WindowIterator& a) const
        {
            return m_pos != a.m_pos;
        }
    };

    /// @brief Internal Window ND iterator class specialization for 0-sized 
    /// apron (single-valued iterator, up to ILP)
    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
             int IPX, int IPY, int IPZ, BorderBehavior BORDERS, int TEXTURE_UID,
             GlobalReadScheme GRS, bool MULTI_GPU>
    class WindowIterator<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 0, IPX, 
                         IPY, IPZ, BORDERS, TEXTURE_UID, GRS, MULTI_GPU> 
        : public std::iterator<std::input_iterator_tag, T>
    {
    protected:
        typedef Window<T, 1, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 0, IPX, 
                       IPY, IPZ, BORDERS, TEXTURE_UID, GRS, MULTI_GPU> Parent;

        int m_id;
        const Parent& m_sParent;

        __device__  __forceinline__ void next()
        {
            m_id++;
        }
    public:
        __device__ WindowIterator(unsigned int pos, const Parent& parent) : 
            m_sParent(parent)
        {
            m_id = pos;
        }

        __device__ WindowIterator(const WindowIterator& other) : 
            m_sParent(other.m_sParent)
        {
            m_id = other.m_id;
        }

        __device__ __forceinline__ int index() const
        {
            return m_id;
        }

        __device__ __forceinline__ const T& operator*() const
        {
            return m_sParent.m_regs[m_id];
        }

        __device__  __forceinline__ WindowIterator& operator++() // Prefix
        {
            next();
            return *this;
        }

        __device__  __forceinline__ WindowIterator operator++(int) // Postfix
        {
            WindowIterator temp(*this);
            next();
            return temp;
        }

        __device__  __forceinline__ bool operator==(
            const WindowIterator& a) const
        {
            return m_id == a.m_id;
        }
        __device__  __forceinline__ bool operator!=(
            const WindowIterator& a) const
        {
            return m_id != a.m_id;
        }
    };

    /// @brief Internal Window ND iterator class specialization for 0-sized 
    /// apron (single-valued iterator, up to ILP)
    template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int BLOCK_DEPTH, 
             int IPX, int IPY, int IPZ, BorderBehavior BORDERS, int TEXTURE_UID,
             GlobalReadScheme GRS, bool MULTI_GPU>
    class WindowIterator<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 0, IPX, 
                         IPY, IPZ, BORDERS, TEXTURE_UID, GRS, MULTI_GPU> 
        : public std::iterator<std::input_iterator_tag, T>
    {
    protected:
        typedef Window<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH, 0, IPX, 
                       IPY, IPZ, BORDERS, TEXTURE_UID, GRS, MULTI_GPU> Parent;

        int m_id;
        const Parent& m_sParent;

        __device__  __forceinline__ void next()
        {
            m_id++;
        }
    public:
        __device__ WindowIterator(unsigned int pos, const Parent& parent) : 
            m_sParent(parent)
        {
            m_id = pos;
        }

        __device__ WindowIterator(const WindowIterator& other) : 
            m_sParent(other.m_sParent)
        {
            m_id = other.m_id;
        }

        __device__ __forceinline__ int index() const
        {
            return m_id;
        }

        __device__ __forceinline__ const T& operator*() const
        {
            return m_sParent.m_regs[m_id];
        }

        __device__  __forceinline__ WindowIterator& operator++() // Prefix
        {
            next();
            return *this;
        }

        __device__  __forceinline__ WindowIterator operator++(int) // Postfix
        {
            WindowIterator temp(*this);
            next();
            return temp;
        }

        __device__  __forceinline__ bool operator==(
            const WindowIterator& a) const
        {
            return m_id == a.m_id;
        }
        __device__  __forceinline__ bool operator!=(
            const WindowIterator& a) const
        {
            return m_id != a.m_id;
        }
    };

}  // namespace maps

#endif  // __MAPS_WINDOW_ITERATOR_INL_
