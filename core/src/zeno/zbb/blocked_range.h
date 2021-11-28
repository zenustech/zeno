#pragma once


#include <zeno/common.h>
#include <cstdint>
#include <thread>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zbb {


inline static auto divup(auto lhs, auto rhs) {
    return (lhs + rhs - 1) / rhs;
}


inline static std::size_t get_num_procs() {
    return std::thread::hardware_concurrency();
}


template <class T>
struct blocked_range {
    T _begin, _end;
    std::size_t _grain;

    inline constexpr blocked_range(T const &begin, T const &end, std::size_t grain) noexcept
        : _begin(begin), _end(end), _grain(grain)
    {}

    inline constexpr blocked_range(T const &begin, T const &end) noexcept
        : blocked_range(begin, end, _idivup(end - begin, 4 * get_num_procs()))
    {}

    inline constexpr T begin() const noexcept {
        return _begin;
    }

    inline constexpr T end() const noexcept {
        return _end;
    }

    inline constexpr std::size_t grain() const noexcept {
        return _grain;
    }

    inline constexpr std::size_t size() const noexcept {
        return end() - begin();
    }
};


}
ZENO_NAMESPACE_END
