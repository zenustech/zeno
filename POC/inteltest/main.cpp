#include <cstdio>

#if 0
#include <CL/sycl.hpp>
#else

#include <array>
#include <vector>

inline namespace __zsfakesycl {
namespace sycl {

using cl_int = int;
using cl_float = float;

template <size_t N>
struct id : std::array<size_t, N> {
    using std::array<size_t, N>::array;

    constexpr explicit(N != 1) id(size_t i)
        : std::array<size_t, N>({i}) {
    }

    constexpr explicit(N != 1) operator size_t() const {
        return std::get<0>(*this);
    }
};

template <size_t N>
struct nd_range {
    id<N> global_size{};
    id<N> local_size{};

    nd_range() = default;

    constexpr explicit nd_range(id<N> global_size, id<N> local_size)
        : global_size(global_size), local_size(local_size)
    {}

    constexpr size_t get_global_size(size_t i) const {
        return global_size[i];
    }

    constexpr size_t get_local_size(size_t i) const {
        return local_size[i];
    }
};

template <size_t N>
struct nd_item : nd_range<N> {
    id<N> global_id{};
    id<N> local_id{};

    constexpr size_t get_global_id(size_t i) const {
        return global_id[i];
    }

    constexpr size_t get_local_id(size_t i) const {
        return local_id[i];
    }
};

struct handler {
    template <size_t N, class F>
    void parallel_for(nd_range<N> dim, F &&f) {
        nd_item<N> item;
        f(item);
    }
};

struct queue {
    template <class F>
    void submit(F &&f) {
        handler h;
        f(h);
    }
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

template <access::mode mode, class Buf, class T, size_t N>
struct accessor {
    Buf const &buf;

    explicit accessor(Buf const &buf) : buf(buf) {
    }

    inline decltype(auto) operator[](id<N> idx) const {
        return buf._M_at(idx);
    }
};

template <class T, size_t N>
struct buffer {
    mutable std::vector<T> _M_data;
    id<N> _M_size;

    static size_t _M_calc_product(id<N> size) {
        size_t ret = 1;
        for (int i = 0; i < N; i++) {
            ret *= size[i];
        }
        return ret;
    }

    size_t _M_linearize_id(id<N> idx) const {
        size_t ret = 0;
        size_t term = 1;
        for (size_t i = 0; i < N; i++) {
            ret += term * idx[i];
            term *= _M_size[i];
        }
        return ret;
    }

    explicit buffer(id<N> size)
        : _M_size(size), _M_data(_M_calc_product(size)) {
    }

    template <access::mode mode>
    auto get_access() const {
        return accessor<mode, buffer, T, N>(*this);
    }

    template <access::mode mode>
    auto get_access(sycl::handler &) const {
        return accessor<mode, buffer, T, N>(*this);
    }

    id<N> size() const {
        return _M_size;
    }

    T &_M_at(id<N> idx) const {
        return _M_data.at(_M_linearize_id(idx));
    }
};

}
}

#endif


int main() {
    sycl::queue que;

    sycl::buffer<sycl::cl_int, 1> buf(32);

    que.submit([&] (sycl::handler &cgh) {
        auto d_buf = buf.get_access<sycl::access::mode::discard_write>(cgh);
        sycl::nd_range<1> dim(32, 4);
        cgh.parallel_for(dim, [=] (sycl::nd_item<1> id) {
            auto i = id.get_global_id(0);
            d_buf[i] = (sycl::cl_int)i + 1;
        });
    });

    auto h_buf = buf.get_access<sycl::access::mode::read>();
    for (int i = 0; i < 32; i++) {
        printf("%d\n", h_buf[i]);
    }

    return 0;
}
