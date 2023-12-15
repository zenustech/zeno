#pragma once

#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#include <vector>

namespace zeno {
namespace _parallel_push_back_details {

template <class T, class SizeT = std::size_t>
struct impl_push_back_1 {
    SizeT counter{};

    inline static constexpr bool is_scan_pass = true;

    void push_back(T &&t) {
        ++counter;
    }

    void push_back(T const &t) {
        ++counter;
    }

    template <class ...Ts>
    void emplace_back(Ts &&...ts) {
        ++counter;
    }
};

template <class Iter, class T, class SizeT = std::size_t>
struct impl_push_back_2 {
    Iter iter;

    inline static constexpr bool is_scan_pass = false;

    void push_back(T &&t) {
        *iter = std::move(t);
        ++iter;
    }

    void push_back(T const &t) {
        *iter = t;
        ++iter;
    }

    template <class ...Ts>
    T &emplace_back(Ts &&...ts) {
        T &ret = *iter;
        ret = T(std::forward<Ts>(ts)...);
        ++iter;
        return ret;
    }
};

}

template <class VectorT, class Index, class Func>
void parallel_push_back(VectorT &vec, Index count, Func func) {
    using namespace _parallel_push_back_details;
    using T = typename VectorT::value_type;
    using SizeT = typename VectorT::size_type;
    using Iter = typename VectorT::iterator;
    SizeT old_size = vec.size();
    std::vector<SizeT> scanres(count);
    SizeT total = parallel_exclusive_scan_sum(counter_iterator<Index>(Index{}), counter_iterator<Index>(count), scanres.begin(), [&func] (Index index) -> SizeT {
        impl_push_back_1<T, SizeT> impl{};
        func(index, impl);
        return impl.counter;
    });
    vec.resize(old_size + total);
    Iter first = vec.begin() + old_size;
    parallel_for(count, [&func, scanres = scanres.data(), first] (Index index) {
        impl_push_back_2<Iter, T, SizeT> impl{first + scanres[index]};
        func(index, impl);
    });
}

}
