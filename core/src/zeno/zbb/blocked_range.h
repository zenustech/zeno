#pragma once


#include <zeno/common.h>
#include <cstdint>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T>
struct blocked_range {
    T _begin, _end;

    constexpr blocked_range(T const &begin, T const &end) noexcept
        : _begin(begin), _end(end)
    {}

    inline constexpr T begin() const noexcept {
        return _begin;
    }

    inline constexpr T end() const noexcept {
        return _end;
    }

    constexpr std::size_t size() const noexcept {
        return end() - begin();
    }
};


}
ZENO_NAMESPACE_END
