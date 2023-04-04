#pragma once

#include <memory>

namespace zeno {

template <typename T>
class SimpleList {
  typedef T* iterator;
  typedef const T* const_iterator;

  public:
    SimpleList(const size_t max_size)
        : m_max_size(max_size)
        , m_current_size(0)
    {
        m_elements = new T[max_size];
    }

    // TODO [darc] : Copy construct :

    virtual ~SimpleList() {
        delete []m_elements;
    }

    size_t size() noexcept{ return m_current_size; }
    size_t capacity() noexcept { return m_max_size; }

    bool add(const T& data) noexcept {
        if (m_current_size < m_max_size) {
            m_elements[m_current_size++] = data;
            return true;
        }
        return false;
    }

    void remove_at(size_t index) noexcept {
        if (index < m_current_size) {
            m_current_size--;
            if (index + 1 != m_current_size) {
                // not last one
                std::memmove(m_elements + index, index + 1, (m_current_size - index) * sizeof(T));
            }
        }
    }

    void remove(const T& value) noexcept {
        SimpleList<size_t> IndicesToRemove(m_max_size);
        for (size_t i = 0; i < m_current_size; ++i) {
            if (m_elements[i] == value) {
                IndicesToRemove.add(i);
            }
        }

        for (const_iterator it : IndicesToRemove) {
            remove_at(*it);
        }
    }

    T* operator[](size_t index) noexcept {
        if (index < m_current_size) {
            return m_elements[index];
        }
        return nullptr;
    }

    const T* operator[](size_t index) const noexcept {
        if (index < m_current_size) {
            return m_elements[index];
        }
        return nullptr;
    }

    iterator begin() { return m_elements; }
    const_iterator begin() const { return m_elements; }
    iterator end() { return m_elements + m_current_size; }
    const_iterator end() const { return m_elements + m_current_size; }
  private:
    size_t m_max_size;
    size_t m_current_size;
    T* m_elements;

  public:
    static SimpleList<T> from_raw(const T* array_base, const size_t size) {
        SimpleList<T> result(size);
        std::memmove(result.m_elements, array_base, size * sizeof(T));
        return result;
    }
};
}
