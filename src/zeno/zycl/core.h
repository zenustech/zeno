#pragma once

#include <zeno/common.h>

#if defined(SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp>

ZENO_NAMESPACE_BEGIN
namespace zycl {
    using namespace cl::sycl;
}
ZENO_NAMESPACE_END

#else

#pragma message("<zeno/zycl/core.h> is using host emulated sycl, which is CPU-only and slow.")

#include <array>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace zycl {

template <int N>
struct id : std::array<size_t, N> {
    using std::array<size_t, N>::array;

    constexpr explicit(N != 1) id(size_t i)
        : std::array<size_t, N>({i}) {
        if constexpr (N != 1) {
            for (int j = 1; j < N; j++) {
                (*this)[j] = i;
            }
        }
    }

    constexpr explicit(N != 1) operator size_t() const {
        return std::get<0>(*this);
    }
};

template <int N>
struct range : id<N> {
    using id<N>::id;
};

template <int N>
struct item : range<N> {
    range<N> _M_range{};
    id<N> _M_id{};

    constexpr size_t get_range(size_t i) const {
        return _M_range[i];
    }

    constexpr size_t get_id(size_t i) const {
        return _M_id[i];
    }
};


template <int N>
struct nd_range {
    range<N> _M_global_range{};
    range<N> _M_local_range{};

    nd_range() = default;

    constexpr explicit nd_range(id<N> global_range, id<N> local_range)
        : _M_global_range(global_range), _M_local_range(local_range)
    {}

    constexpr size_t get_global_range(size_t i) const {
        return _M_global_range[i];
    }

    constexpr size_t get_local_range(size_t i) const {
        return _M_local_range[i];
    }
};

template <int N>
struct nd_item {
    range<N> _M_global_range{};
    range<N> _M_local_range{};
    id<N> _M_global_id{};
    id<N> _M_local_id{};

    constexpr operator item<N>() const {
        return {_M_global_range, _M_global_id};
    }

    constexpr size_t get_global_range(size_t i) const {
        return _M_global_range[i];
    }

    constexpr size_t get_local_range(size_t i) const {
        return _M_local_range[i];
    }

    constexpr size_t get_global_id(size_t i) const {
        return _M_global_id[i];
    }

    constexpr size_t get_local_id(size_t i) const {
        return _M_local_id[i];
    }
};


struct _M_dummy_event {
    void wait() const {}
};


template <class Acc, class T, class BinOp>
struct _M_reduction {
    Acc _M_acc;
    BinOp _M_binop;

    inline _M_reduction(Acc &&acc, T const &ident, BinOp &&binop)
        : _M_acc(std::move(acc))
        , _M_binop(std::move(binop))
    {
        _M_acc[0] = ident;
    }

    inline void combine(T const &t) {
        _M_acc[0] = _M_binop(_M_acc[0], t);
    }
};

auto reduction(auto &&acc, auto const &ident, auto &&binop) {
    return _M_reduction(std::move(acc), ident, std::move(binop));
}


template <int I, int N>
void _M_nd_range_for(range<N> const &size, id<N> &index, auto &&f) {
    if constexpr (I == N) {
        f(index);
    } else {
        for (index[I] = 0; index[I] < size[I]; index[I]++) {
            _M_nd_range_for<I + 1, N>(size, index, f);
        }
    }
}

template <int N>
void _M_nd_range_for(range<N> const &size, auto &&f) {
    id<N> index;
    _M_nd_range_for<0>(size, index, f);
}

struct handler {
    template <class = void>
    _M_dummy_event single_task(auto &&f) {
        f();
        return {};
    }

    template <class = void, int N>
    _M_dummy_event parallel_for(range<N> dim, auto &&f) {
        _M_nd_range_for(dim, [&] (id<N> idx) {
            item<N> it;
            it._M_range = dim;
            it._M_id = idx;
            f(it);
        });;
        return {};
    }

    template <class = void, int N>
    _M_dummy_event parallel_for(nd_range<N> dim, auto &&f) {
        _M_nd_range_for(dim._M_global_range, [&] (id<N> global_id) {
            nd_item<N> it;
            it._M_global_range = dim._M_global_range;
            it._M_local_range = dim._M_local_range;
            it._M_global_id = global_id;
            f(it);
        })
        return {};
    }

    template <class = void, int N>
    _M_dummy_event parallel_for(range<N> dim, auto red, auto &&f) {
        _M_nd_range_for(dim, [&] (id<N> idx) {
            item<N> it;
            it._M_range = dim;
            it._M_id = idx;
            f(it, red);
        });
        return {};
    }

    template <class = void, int N>
    _M_dummy_event parallel_for(nd_range<N> dim, auto red, auto &&f) {
        _M_nd_range_for(dim._M_global_range, [&] (id<N> global_id) {
            nd_item<N> it;
            it._M_global_range = dim._M_global_range;
            it._M_local_range = dim._M_local_range;
            it._M_global_id = global_id;
            f(it, red);
        });
        return {};
    }
};

struct default_selector {
};

struct queue {
    explicit queue(default_selector = {}) {}

    void submit(auto &&f) {
        handler h;
        f(h);
    }

    void wait() {}
};

namespace access {

enum class mode {
    read,
    write,
    read_write,
    discard_write,
    discard_read_write,
    atomic,
};

};

template <access::mode mode, class Buf, class T, int N>
struct accessor {
    Buf const &buf;

    explicit accessor(Buf const &buf) : buf(buf) {
    }

    inline T &operator[](id<N> idx) const {
        return const_cast<Buf &>(buf)._M_at(idx);
    }

    inline range<N> get_range() const {
        return buf.get_range();
    }
};

template <int N>
inline size_t _M_calc_product(range<N> const &size) {
    size_t ret = 1;
    for (int i = 0; i < N; i++) {
        ret *= size[i];
    }
    return ret;
}

template <int N>
inline size_t _M_linearize_id(range<N> const &size, id<N> const &idx) {
    size_t ret = 0;
    size_t term = 1;
    for (size_t i = 0; i < N; i++) {
        ret += term * idx[i];
        term *= size[i];
    }
    return ret;
}

template <class T, int N>
struct buffer {
    std::vector<T> _M_data;
    range<N> _M_shape;

    buffer() = default;
    buffer(buffer const &) = default;
    buffer &operator=(buffer const &) = default;
    buffer(buffer &&) = default;
    buffer &operator=(buffer &&) = default;

    explicit buffer(range<N> shape)
        : _M_shape(shape), _M_data(_M_calc_product(shape)) {
    }

    template <access::mode mode>
    auto get_access() const {
        return accessor<mode, buffer, T, N>(*this);
    }

    template <access::mode mode>
    auto get_access(handler &) const {
        return accessor<mode, buffer, T, N>(*this);
    }

    range<N> get_shape() const {
        return _M_shape;
    }

    size_t size() const {
        return _M_calc_product(_M_shape);
    }

    T &_M_at(id<N> idx) {
        return _M_data.at(_M_linearize_id(_M_shape, idx));
    }
};

}
ZENO_NAMESPACE_END

#endif
