#pragma once
#include <stdint.h>
#include "handle.h"

namespace zeno::rxmesh {

template <typename HandleT>
struct Iterator {
    using LocalT = typename HandleT::LocalT;

    __device__ Iterator(const uint16_t  local_id,
                        const LocalT*   patch_output,
                        const uint16_t* patch_offset,
                        const uint32_t  offset_size,
                        const uint32_t  patch_id,
                        const uint32_t  num_owned,
                        const uint32_t* not_owned_patch,
                        const uint16_t* not_owned_local_id,
                        int             shift = 0)
        : m_patch_output(patch_output),
          m_patch_id(patch_id),
          m_num_owned(num_owned),
          m_not_owned_patch(not_owned_patch),
          m_not_owned_local_id(not_owned_local_id),
          m_shift(shift) {
        set(local_id, offset_size, patch_offset);
    }

    Iterator(const Iterator& orig) = default;


    __device__ uint16_t size() const {
        return m_end - m_begin;
    }

    __device__ HandleT operator[](const uint16_t i) const {
        assert(m_patch_output);
        assert(i + m_begin < m_end);
        uint16_t lid = (m_patch_output[m_begin + i].id) >> m_shift;
        if (lid < m_num_owned) {
            return {m_patch_id, lid};
        } else {
            lid -= m_num_owned;
            return {m_not_owned_patch[lid], m_not_owned_local_id[lid]};
        }
    }
    __device__ HandleT operator*() const {
        assert(m_patch_output);
        return ((*this)[m_current]);
    }

    __device__ HandleT back() const {
        return ((*this)[size() - 1]);
    }
    __device__ HandleT front() const {
        return ((*this)[0]);
    }

    __device__ Iterator& operator++() {
        m_current = (m_current + 1) % size();
        return *this;
    }
    __device__ Iterator operator++(int) {
        Iterator pre(*this);
        m_current = (m_current + 1) % size();
        return pre;
    }

    __device__ Iterator& operator--() {
        m_current = (m_current == 0) ? size() - 1 : m_current - 1;
        return *this;
    }
    __device__ Iterator operator--(int) {
        Iterator pre(*this);
        m_current = (m_current == 0) ? size() - 1 : m_current - 1;
        return pre;
    }

    __device__ bool operator==(const Iterator& rhs) const {
        return rhs.m_local_id == m_local_id && rhs.m_patch_id == m_patch_id &&
               rhs.m_current == m_current;
    }
    __device__ bool operator!=(const Iterator& rhs) const {
        return !(*this == rhs);
    }


   private:
    const LocalT*   m_patch_output;
    const uint32_t  m_patch_id;
    const uint32_t* m_not_owned_patch;
    const uint16_t* m_not_owned_local_id;
    uint16_t        m_num_owned;
    uint16_t        m_local_id;
    uint16_t        m_begin;
    uint16_t        m_end;
    uint16_t        m_current;
    int             m_shift;

    __device__ void set(const uint16_t  local_id,
                        const uint32_t  offset_size,
                        const uint16_t* patch_offset) {
        m_current  = 0;
        m_local_id = local_id;
        if (offset_size == 0) {
            m_begin = patch_offset[m_local_id];
            m_end   = patch_offset[m_local_id + 1];
        } else {
            m_begin = m_local_id * offset_size;
            m_end   = (m_local_id + 1) * offset_size;
        }
        assert(m_end > m_begin);
    }
};

using ElementIterator = Iterator<ElementHandle>;

}  // namespace rxmesh