/* Copyright (c) 2015-2018 The Khronos Group Inc.

  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and/or associated documentation files (the
  "Materials"), to deal in the Materials without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Materials, and to
  permit persons to whom the Materials are furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Materials.

  MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
  KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
  SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
     https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
*/

/**
 * @file
 * @brief Definition of SYCL iterator classes and functions
 */
#ifndef __EXPERIMENTAL_PARALLEL_SYCL_ITERATOR___
#define __EXPERIMENTAL_PARALLEL_SYCL_ITERATOR___

#include <type_traits>
#include <typeinfo>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <CL/sycl.hpp>

namespace sycl {
namespace helpers {

template <typename T, cl::sycl::access::mode Mode>
using sycl_host_acc =
    cl::sycl::accessor<T, 1, Mode, cl::sycl::access::target::host_buffer>;
template <typename T, typename Alloc>
using sycl_buffer_1d = cl::sycl::buffer<T, 1, Alloc>;

class SyclIterator {
 protected:
  size_t pos_;

 public:
  SyclIterator(size_t pos) : pos_(pos){};

  inline size_t get_pos() const { return pos_; }

  inline void set_pos(size_t pos) { pos_ = pos; }
};

struct buffer_iterator_tag {};
struct host_accessor_iterator_tag {};

/* HostAccessorIterator
 * Iterator that access sycl-handled memory objects from
 * the host via a host accessor
 */
template <typename T, cl::sycl::access::mode Mode>
class HostAccessorIterator : public SyclIterator {
 private:
  sycl_host_acc<T, Mode> h_;

 public:
  typedef T m_type;
  typedef T value_type;
  typedef size_t difference_type;
  typedef T *pointer;
  typedef T &reference;
  typedef std::random_access_iterator_tag iterator_category;
  typedef host_accessor_iterator_tag sycl_iterator_category;

  HostAccessorIterator(sycl_host_acc<T, Mode> &h, size_t pos)
      : SyclIterator(pos), h_(h) {}

  template <typename U, cl::sycl::access::mode ModeU>
  HostAccessorIterator(const HostAccessorIterator<U, ModeU> &hI)
      : SyclIterator(hI.get_pos()), h_(hI.h_) {}

  const HostAccessorIterator<T, Mode> operator+(
      const HostAccessorIterator<T, Mode> &other) const {
    HostAccessorIterator<T, Mode> result(*this);
    result.pos_ += other.get_pos();
    return result;
  }

  difference_type operator-(const HostAccessorIterator<T, Mode> &other) const {
    return this->get_pos() == other.get_pos();
  }

  const HostAccessorIterator<T, Mode> operator+(const int &value) const {
    HostAccessorIterator<T, Mode> result(*this);
    result.pos_ += value;
    return result;
  }

  const HostAccessorIterator<T, Mode> operator-(const int &value) const {
    HostAccessorIterator<T, Mode> result(*this);
    result.pos_ -= value;
    return result;
  }

  // Prefix operator (Increment and return value)
  HostAccessorIterator<T, Mode> &operator++() {
    this->pos_++;
    return (*this);
  }

  // Postfix operator (Return value and increment)
  HostAccessorIterator<T, Mode> operator++(int i) {
    HostAccessorIterator<T, Mode> tmp(*this);
    this->pos_ += 1;
    return tmp;
  }

  reference operator*() const { return h_[this->get_pos()]; }

  pointer operator->() const { return &h_[this->get_pos()]; }
};

/* BufferIterator
 * Iterator that allows passing ranges of buffers to
 * Parallel STL calls. Cannot access the buffer from the
 * host, that requires a host accessor.
 */
template <typename T, typename Alloc>
class BufferIterator : public SyclIterator {
 protected:
  sycl_buffer_1d<T, Alloc> b_;

 public:
  typedef T m_type;
  typedef T value_type;
  typedef size_t difference_type;
  typedef Alloc allocator_type;
  typedef T *pointer;
  typedef T &reference;
  typedef std::random_access_iterator_tag iterator_category;
  typedef buffer_iterator_tag sycl_iterator_category;

  BufferIterator(sycl_buffer_1d<T, Alloc> &h, size_t pos) : SyclIterator(pos), b_(h) {}

  template <typename U>
  BufferIterator(const BufferIterator<U, Alloc> &hI)
      : SyclIterator(hI.get_pos()), b_(hI.b_) {}

  inline sycl_buffer_1d<T, Alloc> get_buffer() const { return b_; }

  const BufferIterator<T, Alloc> operator+(const BufferIterator<T, Alloc> &other) const {
    BufferIterator<T, Alloc> result(*this);
    result.pos_ += other.get_pos();
    return result;
  }

  difference_type operator-(const BufferIterator<T, Alloc> &other) const {
    return this->get_pos() - other.get_pos();
  }

  const BufferIterator<T, Alloc> operator+(const int &value) const {
    BufferIterator<T, Alloc> result(*this);
    result.pos_ += value;
    return result;
  }

  const BufferIterator<T, Alloc> operator-(const int &value) const {
    BufferIterator<T, Alloc> result(*this);
    result.pos_ -= value;
    return result;
  }

  BufferIterator<T, Alloc> &operator+=(const int &value) {
    this->pos_ += value;
    return (*this);
  }

  // Prefix operator (Increment and return value)
  BufferIterator<T, Alloc> &operator++() {
    this->pos_++;
    return (*this);
  }

  // Postfix operator (Return value and increment)
  BufferIterator<T, Alloc> operator++(int i) {
    BufferIterator<T, Alloc> tmp(*this);
    this->pos_ += 1;
    return tmp;
  }

  reference operator*() = delete;

  pointer operator->() = delete;
};

/* InputBufferIterator.
 * Read-only variant of a BufferIterator.
 */
template <typename T, typename Alloc>
class InputBufferIterator : public BufferIterator<T, Alloc> {
 public:
  typedef T m_type;
  typedef T value_type;
  typedef size_t difference_type;
  typedef T *pointer;
  typedef T &reference;
  typedef std::input_iterator_tag iterator_category;
  typedef buffer_iterator_tag sycl_iterator_category;

  InputBufferIterator(sycl_buffer_1d<T, Alloc> &h, size_t pos)
      : BufferIterator<T, Alloc>(h, pos) {}

  template <typename U>
  InputBufferIterator(const InputBufferIterator<U, Alloc> &hI)
      : BufferIterator<T, Alloc>(hI.b_, hI.get_pos()) {}

  template <typename U>
  InputBufferIterator(const BufferIterator<U, Alloc> &hI)
      : BufferIterator<T, Alloc>(hI.b_, hI.get_pos()) {}
};

//template <typename Iterator1, typename Iterator2>
inline bool operator==(SyclIterator a, SyclIterator b) {
  return a.get_pos() == b.get_pos();
}

//template <typename Iterator1, typename Iterator2>
inline bool operator!=(SyclIterator a, SyclIterator b) {
  return !(a == b);
}

template <typename T, cl::sycl::access::mode Mode>
HostAccessorIterator<T, Mode> begin(sycl_host_acc<T, Mode> &h) {
  return HostAccessorIterator<T, Mode>(h, 0);
}

template <typename T, cl::sycl::access::mode Mode>
HostAccessorIterator<T, Mode> end(sycl_host_acc<T, Mode> &h) {
  return HostAccessorIterator<T, Mode>(h, h.get_count());
}

template <typename T, typename Alloc>
BufferIterator<T, Alloc> begin(sycl_buffer_1d<T, Alloc> &h) {
  return BufferIterator<T, Alloc>(h, 0);
}

template <typename T, typename Alloc>
BufferIterator<T, Alloc> end(sycl_buffer_1d<T, Alloc> &h) {
  return BufferIterator<T, Alloc>(h, h.get_count());
}

}  // namespace helpers
}  // namespace sycl

#endif  // __EXPERIMENTAL_PARALLEL_SYCL_ITERATOR___
