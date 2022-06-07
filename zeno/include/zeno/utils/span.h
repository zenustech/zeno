#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

template <class T>
struct __span_traits {
    using value_type = T;
    using reference = T &;
    using const_reference = T const &;
    using pointer = T *;
    using const_pointer = T const *;
    using iterator = T *;
    using const_iterator = T const *;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
};

template <class T>
struct __span_traits<const T> : __span_traits<T> {
    using value_type = std::remove_const_t<T>;
    using reference = T const &;
    using pointer = T const *;
    using iterator = T const *;
};

template <class T>
struct span {
    using _M_traits = __span_traits<T>;
    using value_type = typename _M_traits::value_type;
    using reference = typename _M_traits::reference;
    using const_reference = typename _M_traits::const_reference;
    using pointer = typename _M_traits::pointer;
    using const_pointer = typename _M_traits::const_pointer;
    using iterator = typename _M_traits::iterator;
    using const_iterator = typename _M_traits::const_iterator;
    using size_type = typename _M_traits::size_type;
    using difference_type = typename _M_traits::difference_type;

    pointer m_begin;
    pointer m_end;

    span() = default;

    constexpr span(pointer a_begin, pointer a_end) noexcept : m_begin{a_begin}, m_end{a_end} {
    }

    template <class U, class = std::enable_if_t<
        std::is_convertible_v<decltype(std::declval<U>().data()), pointer>
        && std::is_convertible_v<decltype(std::declval<U>().size()), size_type>
        && std::is_convertible_v<decltype(std::declval<U>().data() + std::declval<U>().size()), pointer>
        >>
    constexpr span(U &&u) noexcept : span{u.data(), u.data() + u.size()} {
    }

    constexpr iterator begin() noexcept {
        return m_begin;
    }

    constexpr iterator end() noexcept {
        return m_end;
    }

    constexpr const_iterator begin() const noexcept {
        return m_begin;
    }

    constexpr const_iterator end() const noexcept {
        return m_end;
    }

    constexpr const_iterator cbegin() const noexcept {
        return begin();
    }

    constexpr const_iterator cend() const noexcept {
        return end();
    }

    constexpr pointer data() noexcept {
        return m_begin;
    }

    constexpr const_pointer data() const noexcept {
        return m_begin;
    }

    constexpr const_pointer cdata() const noexcept {
        return data();
    }

    constexpr size_type size() const noexcept {
        return m_end - m_begin;
    }

    constexpr void remove_prefix(size_type n) noexcept {
        m_begin += n;
    }

    constexpr void remove_suffix(size_type n) noexcept {
        m_end -= n;
    }

    reference operator[](size_type n) noexcept {
        return m_begin[n];
    }

    const_reference operator[](size_type n) const noexcept {
        return m_begin[n];
    }

    reference at(size_type n) {
        if (n >= size())
            throw std::out_of_range("span::at");
        return m_begin[n];
    }

    const_reference at(size_type n) const {
        if (n >= size())
            throw std::out_of_range("span::at");
        return m_begin[n];
    }

    constexpr reference front() noexcept {
        return *m_begin;
    }

    constexpr reference back() noexcept {
        return *(m_end - 1);
    }

    constexpr const_reference front() const noexcept {
        return *m_begin;
    }

    constexpr const_reference back() const noexcept {
        return *(m_end - 1);
    }

    constexpr bool empty() const noexcept {
        return m_begin == m_end;
    }
};

