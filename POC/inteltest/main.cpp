#include <CL/sycl.hpp>
#include <cstdio>


/*
inline namespace __zsfakesycl {
namespace sycl {

using cl_int = int;
using cl_float = float;

struct handler {
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

template <access::mode mode, class Buf>
struct accessor {
    Buf const &buf;

    explicit accessor(Buf const &buf) : buf(buf) {
    }
};

template <class T, size_t N>
struct buffer {
    explicit buffer(size_t n) {
    }

    template <access::mode mode>
    auto get_access() const {
        return accessor<mode, buffer>(this);
    }
};

}
}
*/


int main() {
    sycl::queue que;

    sycl::buffer<sycl::cl_int, 1> buf(32);

    que.submit([&] (sycl::handler &cgh) {
        auto dbuf = buf.get_access<sycl::access::mode::discard_write>(cgh);
        sycl::range<1> dim(32);
        cgh.parallel_for(dim, [=] (sycl::id<1> id) {
            dbuf[id] = (sycl::cl_int)id.get(0) + 1;
        });
    });

    auto hbuf = buf.get_access<sycl::access::mode::read>();
    for (int i = 0; i < 32; i++) {
        printf("%d\n", hbuf[i]);
    }

    return 0;
}
