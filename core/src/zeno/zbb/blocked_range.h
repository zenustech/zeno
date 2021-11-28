#pragma once


#include <zeno/common.h>
#include <cstdint>
#include <thread>
#include <utility>


ZENO_NAMESPACE_BEGIN
namespace zbb {


inline static std::size_t get_num_procs() {
    return std::thread::hardware_concurrency();
}


template <class T>
struct blocked_range {
    T _begin, _end;
    std::size_t _tid;
    std::size_t _procs, _grain;

    inline constexpr blocked_range(T const &begin, T const &end, std::size_t tid, std::size_t procs, std::size_t grain) noexcept
        : _begin(begin), _end(end), _tid(tid), _procs(procs), _grain(grain)
    {}

    inline constexpr T begin() const noexcept {
        return _begin;
    }

    inline constexpr T end() const noexcept {
        return _end;
    }

    inline constexpr std::size_t tid() const noexcept {
        return _tid;
    }

    inline constexpr std::size_t procs() const noexcept {
        return _procs;
    }

    inline constexpr std::size_t grain() const noexcept {
        return _grain;
    }

    inline constexpr std::size_t size() const noexcept {
        return end() - begin();
    }
};


template <class T>
inline constexpr blocked_range<T> make_blocked_range(T const &begin, T const &end) noexcept
{
    std::size_t procs = get_num_procs();
    std::size_t kprocs = 2 * procs;
    return {begin, end, 0, procs, (static_cast<std::size_t>(end - begin) + kprocs) / kprocs};
}

template <class T>
inline constexpr blocked_range<T> make_blocked_range(T const &begin, T const &end, std::size_t grain) noexcept
{
    std::size_t procs = get_num_procs();
    return {begin, end, 0, procs, grain};
}


}
ZENO_NAMESPACE_END
