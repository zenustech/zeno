#pragma once
#include <vector>

#include "zensim/math/Vec.h"

namespace zs {

  template <typename T, int Size = 8> struct RingBuffer {
    constexpr RingBuffer() : _head{1}, _tail{0}, _size{0} {}

    /// maintainence operation
    constexpr void push_back() noexcept {
      incTail();
      if (_size > Size) incHead();
    }
    constexpr void push_back(const T &element) noexcept {
      incTail();
      if (_size > Size) incHead();
      back() = element;
    }
    constexpr void push_back(T &&element) noexcept {
      incTail();
      if (_size > Size) incHead();
      back() = std::move(element);
    }
    constexpr void pop() { decTail(); }

    /// element access
    constexpr T &back() noexcept { return _buf(_tail); }
    constexpr T const &back() const noexcept { return _buf(_tail); }
    constexpr T &operator[](int index) noexcept {
      index += _head;
      index %= Size;
      return _buf[index];
    }
    constexpr T const &operator[](int index) const noexcept {
      index += _head;
      index %= Size;
      return _buf[index];
    }
    constexpr int size() { return _size; }

  protected:
    constexpr void incTail() noexcept {
      ++_tail, ++_size;
      if (_tail == Size) _tail = 0;
    }
    constexpr void decTail() noexcept {
      if (_size == 0) return;
      --_tail, --_size;
      if (_tail < 0) _tail = Size - 1;
    }
    constexpr void incHead() noexcept {
      if (_size == 0) return;
      ++_head, --_size;
      if (_head == Size) _head = 0;
    }

    vec<T, Size> _buf;
    int _head, _tail, _size;
  };

}  // namespace zs
